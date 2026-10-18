"""Render measured H100 stage runtimes and speedup versus A1; no estimated bandwidth."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_results(rows, destination):
    stages = list(dict.fromkeys(row["stage"] for row in rows))
    sequences = sorted({int(row["sequence"]) for row in rows})
    fig, (runtime, speedup) = plt.subplots(1, 2, figsize=(12, 4.8))
    for index, stage in enumerate(stages):
        selected = sorted((row for row in rows if row["stage"] == stage), key=lambda row: int(row["sequence"]))
        label = f'{stage}: {selected[0]["technique"]}'
        color = plt.get_cmap("tab10")(index)
        runtime.plot([int(row["sequence"]) for row in selected], [float(row["custom_us"]) for row in selected],
                     marker="o", color=color, label=label)
        if all(row.get("speedup_a1") not in (None, "") for row in selected):
            speedup.plot([int(row["sequence"]) for row in selected], [float(row["speedup_a1"]) for row in selected], marker="o", color=color)
    reference = {int(row["sequence"]): float(row["reference_us"]) for row in rows}
    runtime.plot(sequences, [reference[T] for T in sequences], color="black", marker="*", linewidth=2, label="Official FA4")
    baseline = {int(row["sequence"]): float(row["custom_us"]) for row in rows if row["stage"] == "A1"}
    if baseline:
        speedup.plot(sorted(baseline), [baseline[T] / reference[T] for T in sorted(baseline)], color="black", marker="*", linewidth=2)
    runtime.set_yscale("log")
    runtime.set_ylabel("Forward runtime (µs)")
    speedup.set_ylabel("Speedup versus A1 (×)")
    speedup.axhline(1, color="gray", linestyle="--", linewidth=.8)
    for ax in (runtime, speedup):
        ax.set_xscale("log", base=2)
        ax.set_xticks(sequences, labels=[str(T) for T in sequences])
        ax.set_xlabel("Sequence length")
        ax.grid(alpha=.2)
    fig.suptitle("H100 · BF16 causal MHA · B=4, H=8, D=128")
    handles, labels = runtime.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.subplots_adjust(top=.87, bottom=.31, wspace=.3)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    with (args.directory / "timings.csv").open() as file:
        rows = list(csv.DictReader(file))
    plot_results(rows, args.directory / "attention_ablation.png")
