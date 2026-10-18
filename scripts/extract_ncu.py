#!/usr/bin/env python3

"""Extracts comparable timing and hardware metrics from paired Nsight Compute reports.
It prints concise tables and can overwrite one CSV and Markdown summary for a kernel family."""

import argparse
import csv
import io
import math
import re
import subprocess
from pathlib import Path


SPEED_SECTION = "GPU Speed Of Light Throughput"
MEMORY_SECTION = "Memory Workload Analysis"
LAUNCH_SECTION = "Launch Statistics"
OCCUPANCY_SECTION = "Occupancy"
COMMAND_SECTION = "Command line profiler metrics"
REFERENCE_BACKENDS = {"fp32": "cublas_fp32", "bf16": "cublas_bf16",
                      "custom_bf16": "cublas_bf16", "custom": "official",
                      "cublas_fp32": "cublas_fp32", "cublas_bf16": "cublas_bf16",
                      "official": "official"}


# These formatting helpers belong to Nsight extraction now that the old runtime
# runner is removed. Keep the existing report format and pairing unchanged.
def add_reference_percentages(rows, reference, keys, time_field="median_ms"):
    def key(row):
        return tuple(tuple(row[k]) if isinstance(row[k], (list, tuple)) else row[k] for k in keys)
    times = {(row["backend"], key(row)): float(row[time_field]) for row in rows}
    for row in rows:
        backend = reference.get(row["backend"]) if isinstance(reference, dict) else reference
        baseline = times.get((backend, key(row)))
        elapsed = float(row[time_field])
        row["reference_pct"] = (
            100 * baseline / elapsed
            if baseline is not None and math.isfinite(baseline) and baseline > 0
            and math.isfinite(elapsed) and elapsed > 0 else None)


def comparison_table(rows, keys, size, dtype, reference, *,
                     custom_backend="custom", reference_backend="reference",
                     time_field="median_ms", time_scale=1000.0):
    groups = {}
    for row in rows:
        backend = row["backend"]
        if backend not in (custom_backend, reference_backend):
            raise ValueError(f"unexpected comparison backend: {backend}")
        key = tuple(tuple(row[k]) if isinstance(row[k], (list, tuple)) else row[k] for k in keys)
        group = groups.setdefault(key, {})
        if backend in group:
            raise ValueError(f"duplicate measurement for {backend}: {key}")
        group[backend] = row
    def elapsed(row):
        value = float(row[time_field]) * time_scale if row is not None else math.nan
        return value if math.isfinite(value) and value > 0 else None
    def number(value):
        return "-" if value is None else f"{value:.2f}"
    values = []
    for group in groups.values():
        row = next(iter(group.values()))
        custom, baseline = elapsed(group.get(custom_backend)), elapsed(group.get(reference_backend))
        percent = f"{100 * baseline / custom:.1f}" if custom is not None and baseline is not None else "-"
        values.append((size(row), dtype(row), row["operation"], number(custom), number(baseline), percent))
    headers = ("size", "dtype", "operation", "custom us", f"reference ({reference}) us", "reference %")
    numeric = {0, 3, 4, 5}
    widths = [max(3, len(h), *(len(str(row[i])) for row in values)) for i, h in enumerate(headers)]
    def line(cells):
        return "|" + "|".join(
            f" {str(value):{('>' if i in numeric else '<')}{width}} "
            for i, (value, width) in enumerate(zip(cells, widths))) + "|"
    separator = "|" + "|".join(
        "-" * (width + 1) + ":" if i in numeric else ":" + "-" * (width + 1)
        for i, width in enumerate(widths)) + "|"
    return "\n".join([line(headers), separator, *(line(row) for row in values)]) + "\n"


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("ncu")
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--csv-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args()
    if len(args.reports) % 3 != 0:
        parser.error("reports must be LABEL SPEED_REPORT METRICS_REPORT triples")
    return args


def short_kernel_name(full_name):
    patterns = (
        (r"flash_attention_forward(?:_tensor_core)?_kernel", "flash_fwd"),
        (r"flash_fwd", "official_flash_fwd"),
    )
    for pattern, name in patterns:
        if re.search(pattern, full_name, re.IGNORECASE):
            return name

    if re.search(
            r"(?:matmul(?:_tensor_core_mma)?|gemm_bf16)_kernel", full_name):
        return "matmul_TN"

    match = re.search(r"([a-z0-9_]+)_forward_kernel", full_name)
    if match:
        return f"{match.group(1)}_fwd"
    match = re.search(r"([a-z0-9_]+)_kernel", full_name)
    if match:
        return match.group(1)
    return full_name.split("(", 1)[0].strip()[-60:]


def read_report(ncu, report):
    result = subprocess.run(
        [ncu, "--import", report, "--csv", "--page", "details"],
        check=True,
        capture_output=True,
        text=True,
    )
    kernels = {}
    order = []
    for row in csv.DictReader(io.StringIO(result.stdout)):
        kernel_id = row["ID"]
        if kernel_id not in kernels:
            kernels[kernel_id] = {
                "name": short_kernel_name(row["Kernel Name"]),
                "metrics": {},
            }
            order.append(kernel_id)
        metric = row["Metric Name"]
        value = row["Metric Value"]
        if metric and value:
            kernels[kernel_id]["metrics"][(row["Section Name"], metric)] = (
                value,
                row["Metric Unit"],
            )
    return [kernels[kernel_id] for kernel_id in order]


def numeric(metrics, section, metric):
    raw, unit = metrics.get((section, metric), ("nan", ""))
    try:
        value = float(raw.replace(",", ""))
    except ValueError:
        return math.nan, unit
    return value, unit


def duration_us(metrics):
    value, unit = numeric(metrics, SPEED_SECTION, "Duration")
    return value * {"ns": 0.001, "us": 1.0, "ms": 1000.0, "s": 1e6}.get(
        unit, 1.0
    )


def byte_count(metrics, metric):
    value, unit = numeric(metrics, COMMAND_SECTION, metric)
    return value * {
        "byte": 1.0,
        "Kbyte": 1e3,
        "Mbyte": 1e6,
        "Gbyte": 1e9,
    }.get(unit, 1.0)


def size_bytes(metrics, section, metric):
    value, unit = numeric(metrics, section, metric)
    return value * {
        "byte": 1.0,
        "Kbyte": 1e3,
        "Mbyte": 1e6,
        "Gbyte": 1e9,
    }.get(unit, 1.0)


def metric_value(metrics, section, metric):
    return numeric(metrics, section, metric)[0]


def parse_label(label):
    parts = label.split("/")
    if len(parts) >= 3:
        return "/".join(parts[:-2]), parts[-2], parts[-1]
    return label, "custom", "all"


def merge_reports(ncu, label, speed_path, metrics_path):
    speed = read_report(ncu, speed_path)
    hardware = read_report(ncu, metrics_path)
    if len(speed) != len(hardware):
        raise RuntimeError(
            f"report kernel count mismatch for {label}: "
            f"speed={len(speed)} metrics={len(hardware)}"
        )
    workload, backend, operation = parse_label(label)
    rows = []
    for speed_kernel, hardware_kernel in zip(speed, hardware):
        metrics = hardware_kernel["metrics"].copy()
        metrics.update(speed_kernel["metrics"])
        elapsed = duration_us(metrics)
        read_bytes = byte_count(metrics, "dram__bytes_read.sum")
        write_bytes = byte_count(metrics, "dram__bytes_write.sum")
        rows.append(
            {
                "workload": workload,
                "backend": backend,
                "operation": operation,
                "kernel": speed_kernel["name"],
                "time_us": elapsed,
                "dram_read_mb": read_bytes / 1e6,
                "dram_write_mb": write_bytes / 1e6,
                "dram_pct": metric_value(
                    metrics, SPEED_SECTION, "DRAM Throughput"
                ),
                "sm_pct": metric_value(
                    metrics, SPEED_SECTION, "Compute (SM) Throughput"
                ),
                "occupancy_pct": metric_value(
                    metrics, OCCUPANCY_SECTION, "Achieved Occupancy"
                ),
                "l1_hit_pct": metric_value(
                    metrics, MEMORY_SECTION, "L1/TEX Hit Rate"
                ),
                "l2_hit_pct": metric_value(
                    metrics, MEMORY_SECTION, "L2 Hit Rate"
                ),
                "registers": metric_value(
                    metrics, LAUNCH_SECTION, "Registers Per Thread"
                ),
                "static_smem_bytes": size_bytes(
                    metrics, LAUNCH_SECTION, "Static Shared Memory Per Block"
                ),
                "dynamic_smem_bytes": size_bytes(
                    metrics, LAUNCH_SECTION, "Dynamic Shared Memory Per Block"
                ),
                "spills": metric_value(
                    metrics, MEMORY_SECTION, "Local Memory Spilling Requests"
                ),
            }
        )
    return rows


def weighted(rows, field):
    elapsed = sum(row["time_us"] for row in rows)
    if elapsed == 0:
        return 0.0
    return sum(row[field] * row["time_us"] for row in rows) / elapsed


def totals(rows):
    grouped = {}
    for row in rows:
        key = (row["workload"], row["backend"], row["operation"])
        grouped.setdefault(key, []).append(row)

    result = []
    for (workload, backend, operation), kernels in grouped.items():
        elapsed = sum(row["time_us"] for row in kernels)
        read_mb = sum(row["dram_read_mb"] for row in kernels)
        write_mb = sum(row["dram_write_mb"] for row in kernels)
        result.append(
            {
                "workload": workload,
                "backend": backend,
                "operation": operation,
                "launches": len(kernels),
                "time_us": elapsed,
                "dram_read_mb": read_mb,
                "dram_write_mb": write_mb,
                "dram_pct": weighted(kernels, "dram_pct"),
                "sm_pct": weighted(kernels, "sm_pct"),
                "occupancy_pct": weighted(kernels, "occupancy_pct"),
                "l1_hit_pct": weighted(kernels, "l1_hit_pct"),
                "l2_hit_pct": weighted(kernels, "l2_hit_pct"),
                "max_registers": max(row["registers"] for row in kernels),
                "max_smem_bytes": max(
                    row["static_smem_bytes"] + row["dynamic_smem_bytes"]
                    for row in kernels
                ),
                "spills": sum(row["spills"] for row in kernels),
            }
        )
    return result


def result_table(summary, family=""):
    # The display is timing-only; collected hardware counters remain in the CSV.
    add_reference_percentages(summary, REFERENCE_BACKENDS, ("workload", "operation"), "time_us")
    backends = {r["backend"] for r in summary}
    if backends & {"fp32", "bf16", "custom_bf16", "cublas_fp32", "cublas_bf16"}:
        reference = "cuBLAS"
    elif "official" in backends or family == "flash_attention":
        reference = "FA-2"
    else:
        reference = "not measured"

    normalized = []
    reference_backends = {"official", "cublas_fp32", "cublas_bf16"}
    for row in summary:
        backend = row["backend"]
        if backend in {"fp32", "cublas_fp32"}:
            dtype = "fp32"
        elif backend in {"bf16", "custom_bf16", "cublas_bf16", "official"} or family == "flash_attention":
            dtype = "bf16"
        elif reference == "FA-2":
            dtype = "bf16"
        else:
            dtype = row.get("dtype", "-")
        normalized.append(dict(row, dtype=dtype,
                               backend="reference" if backend in reference_backends else "custom"))
    return comparison_table(
        normalized, ("workload", "dtype", "operation"),
        lambda r: r["workload"], lambda r: r["dtype"], reference,
        time_field="time_us", time_scale=1.0)


def print_totals(summary, family=""):
    print(result_table(summary, family), end="")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, summary, family=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result_table(summary, family), encoding="utf-8")


def main():
    args = arguments()
    rows = []
    for index in range(0, len(args.reports), 3):
        rows.extend(
            merge_reports(
                args.ncu,
                args.reports[index],
                args.reports[index + 1],
                args.reports[index + 2],
            )
        )
    summary = totals(rows)
    family = args.csv_out.stem if args.csv_out else ""
    print_totals(summary, family)
    if args.csv_out:
        write_csv(args.csv_out, summary)
    if args.markdown_out:
        write_markdown(args.markdown_out, summary, family)


if __name__ == "__main__":
    main()
