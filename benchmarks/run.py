"""One entry point: kernel/PyTorch correctness, runtime comparison, or an Nsight launch."""
import argparse
import importlib

from common import measure, record, save_report, torch

FAMILIES = ("matmul", "flash_attention")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=(*FAMILIES, "all"),
                        type=lambda x: {"gemm": "matmul"}.get(x, x))
    parser.add_argument("--suite", choices=("quick", "full", "h100"), default="quick")
    parser.add_argument("--test", action="store_true", help="check native results against PyTorch; no timing")
    parser.add_argument("--reference", help="cublas for GEMM; pytorch, fa2, fa3, fa4, or all for FlashAttention")
    parser.add_argument("--operation", help="run only TN GEMM or FlashAttention forward")
    parser.add_argument("--profile", action="store_true", help="one uncaptured call inside cudaProfilerStart/Stop")
    parser.add_argument("--backend", choices=("custom", "reference", "all"), default="all",
                        help="backend(s) to profile")
    parser.add_argument("--graph-operations", type=int, default=10)
    parser.add_argument("--warmup-ms", type=int, default=1000)
    parser.add_argument("--sample-ms", type=int, default=20)
    parser.add_argument("--trials", type=int, default=9)
    args = parser.parse_args()
    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        families = FAMILIES if args.family == "all" else (args.family,)
        for family in families:
            module = importlib.import_module(family)
            rows, samples, count = [], [], 0
            for operation in module.cases(args, family):
                if args.operation and operation.name != args.operation:
                    continue
                if args.test and operation.expected is None:
                    raise RuntimeError("A correctness test requires a PyTorch expected result.")
                for backend, function in operation.functions.items():
                    if args.test and not args.reference and "custom" in operation.functions and backend != "custom":
                        continue
                    try:
                        result = function()
                        # Surface asynchronous kernel faults before PyTorch formats tensors.
                        # This correctness pass is outside the timed CUDA Graph replay.
                        stream.synchronize()
                        operation.check(result)
                    except AssertionError as error:
                        raise AssertionError(f"{family}: {operation.size}, {operation.dtype}, "
                                             f"{operation.name}, {backend}\n{error}") from error
                    except RuntimeError as error:
                        raise RuntimeError(f"{family}: {operation.size}, {operation.dtype}, "
                                           f"{operation.name}, {backend}\n{error}") from error
                count += 1
                if args.profile:
                    for backend, function in operation.functions.items():
                        kind = "custom" if backend == "custom" else "reference"
                        if args.backend not in ("all", kind):
                            continue
                        torch.cuda.synchronize()
                        torch.cuda.nvtx.range_push(f"{operation.size}/{operation.dtype}/{backend}/{operation.name}")
                        torch.cuda.profiler.start()
                        with torch.no_grad():
                            function()
                        torch.cuda.synchronize()
                        torch.cuda.profiler.stop()
                        torch.cuda.nvtx.range_pop()
                elif not args.test:
                    timings = measure(operation, args)
                    rows.extend(record(operation, timings))
                    samples.append(dict(size=operation.size, dtype=operation.dtype,
                                        operation=operation.name, samples_us=timings))
            stream.synchronize()
            if not count:
                parser.error("No matching operations.")
            if args.test:
                print(f"{family}: {count} correctness checks passed")
            elif not args.profile:
                save_report(family, rows, samples)


if __name__ == "__main__":
    main()
