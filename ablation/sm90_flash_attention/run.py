"""Fixed H100 MHA A1-A8 correctness and CUDA Graph comparison against FA4."""
import argparse
import csv
import ctypes
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))
from common import Operation, measure
from ablation.sm90_flash_attention.plot import plot_results
from ablation.sm90_flash_attention.roofline import plot_roofline

STAGES = {
    "A1": "FA2-style fused",
    "A2": "serial TMA/WGMMA",
    "A3": "K/V pipeline",
    "A4": "register redistribution",
    "A5": "WG ping-pong",
    "A6": "intra-WG overlap",
    "A7": "softmax + rescaling",
    "A8": "scheduling + epilogue",
}
SEQUENCES = (2048, 4096, 8192, 16384)
FIXED = dict(batch=4, query_heads=8, key_value_heads=8, head_dimension=128, causal=True, dtype="bf16")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, nargs="+", default=list(STAGES))
    parser.add_argument("--sequence", type=int, choices=SEQUENCES, nargs="+", default=list(SEQUENCES))
    parser.add_argument("--test", action="store_true", help="PyTorch correctness only; no FA4 dependency or timing")
    parser.add_argument("--profile-stage", choices=STAGES, help="one native call for Compute Sanitizer; skips timing/oracle")
    parser.add_argument("--library", type=Path, default=ROOT / "build/attention_ablation/libattention_ablation.so")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "profiles/attention_ablation_sm90")
    parser.add_argument("--graph-operations", type=int, default=10)
    parser.add_argument("--warmup-ms", type=int, default=1000)
    parser.add_argument("--sample-ms", type=float, default=40)
    parser.add_argument("--trials", type=int, default=35)
    return parser.parse_args()


@torch.no_grad()
def oracle(Q, K, V):
    """Chunk queries to avoid allocating a full B*H*T*T FP32 score matrix."""
    q, k, v = (x.float().transpose(1, 2) for x in (Q, K, V))
    T = q.shape[2]
    O = torch.empty_like(q)
    lse = torch.empty(q.shape[:-1], device=q.device)
    keys = torch.arange(T, device=q.device)
    for start in range(0, T, 128):
        end = min(start + 128, T)
        scores = q[:, :, start:end] @ k.transpose(-1, -2) * (128**-.5)
        causal = torch.arange(start, end, device=q.device)[:, None] >= keys[None, :]
        scores.masked_fill_(~causal, -torch.inf)
        O[:, :, start:end] = scores.softmax(-1) @ v
        lse[:, :, start:end] = scores.logsumexp(-1)
    return O.transpose(1, 2).bfloat16(), lse


def native(lib, stage, O, lse, Q, K, V, T):
    status = lib.attention_ablation_forward(stage, O.data_ptr(), lse.data_ptr(), Q.data_ptr(), K.data_ptr(), V.data_ptr(),
                                          T, torch.cuda.current_stream().cuda_stream)
    if status:
        raise RuntimeError(lib.attention_ablation_last_error().decode())
    return O, lse


def table(rows):
    headers = ["size", "dtype", "operation", "custom us", "reference (FA4) us", "reference %"]
    cells = [[f'B=4,T={r["sequence"]},H=8,D=128', "bf16", r["stage"], f'{r["custom_us"]:.2f}',
              f'{r["reference_us"]:.2f}', f'{r["reference_pct"]:.1f}'] for r in rows]
    widths = [max(len(row[i]) for row in [headers, *cells]) for i in range(len(headers))]
    def line(values):
        return "| " + " | ".join(value.ljust(width) for value, width in zip(values, widths)) + " |"
    return "\n".join([line(headers), line(["-" * width for width in widths]), *map(line, cells)]) + "\n"


def source_hashes():
    paths = sorted(Path(__file__).parent.glob("*.cu")) + sorted(Path(__file__).parent.glob("*.cuh"))
    paths += [ROOT / "kernels/attention/flash_attention/sm89_forward.cu", ROOT / "kernels/attention/flash_attention/sm90_forward.cu"]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


@torch.no_grad()
def main():
    args = arguments()
    # A1 is the only portable stage. Never publish a Hopper study from a substitute GPU.
    if torch.cuda.get_device_capability()[0] != 9 and (not args.test or args.stage != ["A1"]):
        raise RuntimeError("A2-A8 and the benchmark require H100. Local check: --test --stage A1.")
    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    lib = ctypes.CDLL(str(args.library))
    lib.attention_ablation_forward.argtypes = [ctypes.c_int] + [ctypes.c_void_p] * 5 + [ctypes.c_int, ctypes.c_void_p]
    lib.attention_ablation_forward.restype = ctypes.c_int
    lib.attention_ablation_last_error.restype = ctypes.c_char_p
    api = None if args.test or args.profile_stage else importlib.import_module("flash_attn.cute").flash_attn_func
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    rows, raw, passed = [], [], 0
    with torch.cuda.stream(stream):
        for T in args.sequence:
            Q, K, V = tuple((torch.randn(4, T, 8, 128, device="cuda") * .5).bfloat16() for _ in range(3))
            O = torch.empty_like(Q)
            lse = torch.empty(4, 8, T, device="cuda")
            if args.profile_stage:
                native(lib, int(args.profile_stage[1:]), O, lse, Q, K, V, T)
                stream.synchronize()
                continue
            expected = oracle(Q, K, V)
            functions = {stage: (lambda stage=stage: native(lib, int(stage[1:]), O, lse, Q, K, V, T)) for stage in args.stage}
            if api is not None:
                functions["FA4"] = lambda: api(Q, K, V, softmax_scale=128**-.5, causal=True, return_lse=True)[:2]
            operation = Operation(f"B=4,T={T},H=8,D=128", "bf16", "forward", functions, expected,
                                  atol=(1e-2, 1e-4), rtol=(1e-2, 1e-5))
            for name, function in functions.items():
                result = function()
                stream.synchronize()
                try:
                    operation.check(result)
                except AssertionError as error:
                    raise AssertionError(f"{name}, T={T}: {error}") from error
                passed += 1
            if torch.cuda.get_device_capability()[0] == 9:
                # Compare to the actual unchanged kernel too, but do not add another timed stage.
                result = native(lib, 9, O, lse, Q, K, V, T)
                stream.synchronize()
                operation.check(result)
            if args.test:
                continue
            samples = measure(operation, args)
            medians = {name: statistics.median(values) for name, values in samples.items()}
            for stage in args.stage:
                baseline = medians.get("A1")
                rows.append(dict(sequence=T, stage=stage, technique=STAGES[stage], custom_us=medians[stage],
                                 reference="FlashAttention-4", reference_us=medians["FA4"], reference_pct=100 * medians["FA4"] / medians[stage],
                                 speedup_a1=baseline / medians[stage] if baseline else None))
            raw.append(dict(sequence=T, samples_us=samples))
        stream.synchronize()
    if args.test:
        print(f"attention ablation: {passed} correctness checks passed")
        return
    if args.profile_stage:
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "timings.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = table(rows)
    (args.output_dir / "timings.md").write_text(report)
    smi = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version,power.limit", "--format=csv,noheader"], text=True).strip()
    metadata = dict(gpu=torch.cuda.get_device_name(), device_info=smi, torch=torch.__version__, cuda=torch.version.cuda,
                    fixed=FIXED, sequences=args.sequence, stages=args.stage, source_sha256=source_hashes(),
                    timing=dict(graph_operations=args.graph_operations, warmup_ms=args.warmup_ms, sample_ms=args.sample_ms, trials=args.trials),
                    reference_module=api.__module__, reference_path=importlib.import_module("flash_attn.cute").__file__,
                    reference_commit=os.environ.get("GPU_KERNELS_FA4_COMMIT"),
                    implementation_reference_commit="94e22c906678e5483fa0e9e24d8e787bc2c0ed4c",
                    correctness="FP32 chunked PyTorch oracle, before and after graph timing; unchanged production output also checked.", samples=raw)
    (args.output_dir / "samples.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plot_results(rows, args.output_dir / "attention_ablation.png")
    plot_roofline(rows, args.output_dir / "roofline_analytical.png")
    if args.stage == list(STAGES) and args.sequence == list(SEQUENCES):
        destination = ROOT / "assets/attention/sm90_ablation.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.output_dir / "attention_ablation.png", destination)
        shutil.copyfile(args.output_dir / "roofline_analytical.png", destination.with_name("sm90_roofline_analytical.png"))
        readme = ROOT / "README.md"
        marker = "<!-- h100-attention-ablation-figure -->"
        figure = '<img src="assets/attention/sm90_ablation.png" alt="H100 MHA A1-A8 ablation versus official FlashAttention-4" width="800">'
        readme.write_text(readme.read_text().replace(marker, figure))
    print(report, end="")


if __name__ == "__main__":
    main()
