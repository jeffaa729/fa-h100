"""Minimum-I/O analytical attention roofline using saved, measured GPU times."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# NVIDIA's H100 SXM table quotes 1,979 BF16 TFLOP/s WITH sparsity.
# Our dense MMA has half that nominal ceiling; causal masking is not 2:4 sparsity.
PEAK_TFLOPS = 1979 / 2
BANDWIDTH_GBPS = 3350
SPECIFICATION = "https://www.nvidia.com/en-eu/data-center/h100/"


def analytical_work(T):
    """Fixed B=4, H=8, D=128 causal MHA; FMA counts as two FLOPs."""
    B, H, D = 4, 8, 128
    pairs = T * (T + 1) // 2
    # Two products (QK and PV), each with D multiply-adds per valid pair.
    flops = 4 * B * H * D * pairs
    # Read Q/K/V once, write O once (BF16), and write natural-log LSE (FP32).
    # No quadratic score/probability buffer is materialized by these kernels.
    minimum_bytes = 4 * B * H * T * D * 2 + B * H * T * 4
    return flops, minimum_bytes


def plot_roofline(rows, destination):
    points = []
    for row in rows:
        T = int(row["sequence"])
        flops, minimum_bytes = analytical_work(T)
        points.append(dict(sequence=T, stage=row["stage"], technique=row["technique"],
                           measured_us=float(row["custom_us"]), useful_flops=flops, minimum_io_bytes=minimum_bytes,
                           intensity=flops / minimum_bytes, tflops=flops / float(row["custom_us"]) / 1e6))
    for T in sorted({point["sequence"] for point in points}):
        row = next(row for row in rows if int(row["sequence"]) == T)
        flops, minimum_bytes = analytical_work(T)
        points.append(dict(sequence=T, stage="FA4", technique="Official FA4", measured_us=float(row["reference_us"]),
                           useful_flops=flops, minimum_io_bytes=minimum_bytes, intensity=flops / minimum_bytes,
                           tflops=flops / float(row["reference_us"]) / 1e6))

    fig, ax = plt.subplots(figsize=(10, 6.6))
    x = np.geomspace(min(point["intensity"] for point in points) / 4, max(point["intensity"] for point in points) * 1.6, 300)
    ax.loglog(x, np.minimum(PEAK_TFLOPS, BANDWIDTH_GBPS * x / 1000), color="black", linestyle="--", linewidth=2,
              label=f"Nominal roof: {PEAK_TFLOPS:g} TFLOP/s, 3.35 TB/s")
    stages = list(dict.fromkeys(point["stage"] for point in points))
    for index, stage in enumerate(stages):
        selected = sorted((point for point in points if point["stage"] == stage), key=lambda point: point["sequence"])
        official = stage == "FA4"
        ax.plot([point["intensity"] for point in selected], [point["tflops"] for point in selected],
                color="black" if official else plt.get_cmap("tab10")(index), marker="*" if official else "o",
                markersize=9 if official else 5, linewidth=1.5,
                label="Official FA4" if official else f'{stage}: {selected[0]["technique"]}')
        if official:
            for point in selected:
                ax.annotate(f'T={point["sequence"]}', (point["intensity"], point["tflops"]), xytext=(0, 12),
                            textcoords="offset points", ha="center", fontsize=8,
                            bbox=dict(facecolor="white", edgecolor="none", alpha=.8, pad=.3))
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(min(point["tflops"] for point in points) * .7, PEAK_TFLOPS * 1.3)
    ax.set_xlabel("Analytical minimum-I/O arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("Useful QK + PV throughput (TFLOP/s)")
    ax.set_title("H100 SXM · analytical minimum-I/O roofline\nBF16 causal MHA · B=4, H=8, D=128")
    ax.grid(which="both", alpha=.2)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.subplots_adjust(top=.86, bottom=.20)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    with destination.with_suffix(".csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    metadata = dict(byte_model="Minimum algorithmic I/O, not measured HBM traffic; warm graph replay can reuse L2.",
                    flop_model="4*B*H*D*T*(T+1)/2 useful causal QK/PV FLOPs; excludes softmax, masked tile work and addressing.",
                    peak_tflops=PEAK_TFLOPS, bandwidth_gbps=BANDWIDTH_GBPS, specification=SPECIFICATION,
                    roof_basis="Nominal SKU ceilings, not clock-scaled, power-limit-adjusted or measured sustainable peaks.")
    destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=ROOT / "profiles/attention_ablation_sm90")
    args = parser.parse_args()
    with (args.directory / "timings.csv").open() as file:
        rows = list(csv.DictReader(file))
    plot_roofline(rows, args.directory / "roofline_analytical.png")
