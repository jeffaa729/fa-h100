"""PyTorch correctness checks, CUDA Graph timing, and the six-column runtime table."""
import csv
import ctypes
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
P, I, F = ctypes.c_void_p, ctypes.c_int, ctypes.c_float


def library(name):
    return ctypes.CDLL(str(ROOT / "build" / f"libgpu_kernels_{name}_bench.so"))


def bind(lib, name, signature, result=I):
    function = getattr(lib, name)
    function.argtypes, function.restype = signature, result
    return function


def checked(lib, prefix, status):
    if status:
        error = bind(lib, f"gpu_kernels_{prefix}_last_error", [], ctypes.c_char_p)
        raise RuntimeError(error().decode())


def pointers(tensors):
    return [x.data_ptr() for x in tensors]


def stream():
    return torch.cuda.current_stream().cuda_stream


def tensors(result):
    return (result,) if isinstance(result, torch.Tensor) else tuple(result)


@dataclass
class Operation:
    size: str
    dtype: str
    name: str
    functions: dict
    expected: tuple | None
    atol: float = 2e-4
    rtol: float = 2e-3

    def check(self, result):
        if self.expected is not None:
            actual = tensors(result)
            assert len(actual) == len(self.expected)
            for index, (output, expected) in enumerate(zip(actual, self.expected)):
                atol = self.atol[index] if isinstance(self.atol, tuple) else self.atol
                rtol = self.rtol[index] if isinstance(self.rtol, tuple) else self.rtol
                torch.testing.assert_close(output, expected, atol=atol, rtol=rtol)


class Graph:
    def __init__(self, function, operations):
        for _ in range(3):
            function()
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=torch.cuda.current_stream()):
            for _ in range(operations):
                self.output = function()


@torch.no_grad()
def measure(operation, args):
    graphs = {name: Graph(call, args.graph_operations)
              for name, call in operation.functions.items()}
    for graph in graphs.values():
        graph.graph.replay()
        operation.check(graph.output)
    deadline = time.perf_counter() + args.warmup_ms / 1000
    while time.perf_counter() < deadline:
        for graph in graphs.values():
            graph.graph.replay()
        torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def elapsed(graph, repeats):
        start.record()
        for _ in range(repeats):
            graph.graph.replay()
        stop.record()
        stop.synchronize()
        return start.elapsed_time(stop)

    repeats = {name: max(1, math.ceil(args.sample_ms / max(elapsed(graph, 3) / 3, 1e-6)))
               for name, graph in graphs.items()}
    samples = {name: [] for name in graphs}
    names = list(graphs)
    for trial in range(args.trials):
        for name in (names if trial % 2 == 0 else names[::-1]):
            samples[name].append(1000 * elapsed(graphs[name], repeats[name]) /
                                 (repeats[name] * args.graph_operations))
    for graph in graphs.values():
        operation.check(graph.output)
    return samples


def save_report(family, rows, samples):
    tables = []
    for reference in dict.fromkeys(row["reference"] for row in rows):
        selected = [row for row in rows if row["reference"] == reference]
        headers = ["size", "dtype", "operation", "custom us", f"reference ({reference}) us", "reference %"]
        cells = [[row["size"], row["dtype"], row["operation"],
                  "—" if row["custom_us"] is None else f'{row["custom_us"]:.2f}',
                  f'{row["reference_us"]:.2f}',
                  "—" if row["reference_pct"] is None else f'{row["reference_pct"]:.1f}']
                 for row in selected]
        widths = [max(len(line[i]) for line in [headers, *cells]) for i in range(len(headers))]
        def line(values):
            return "| " + " | ".join(value.ljust(width) for value, width in zip(values, widths)) + " |"
        tables.append("\n".join([line(headers), line(["-" * n for n in widths]), *map(line, cells)]))
    report = "\n\n".join(tables) + "\n"
    destination = ROOT / "profiles/runtime"
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / f"{family}.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (destination / f"{family}.md").write_text(report)
    (destination / f"{family}_samples.json").write_text(json.dumps({
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "timing": "CUDA Graph replay; warmup and preparation excluded; median microseconds.",
        "samples": samples}, indent=2) + "\n")
    print(report, end="")


def record(operation, samples):
    custom = statistics.median(samples["custom"]) if "custom" in samples else None
    return [dict(size=operation.size, dtype=operation.dtype, operation=operation.name,
                 custom_us=custom, reference=name, reference_us=statistics.median(values),
                 reference_pct=100 * statistics.median(values) / custom if custom else None)
            for name, values in samples.items() if name != "custom"]
