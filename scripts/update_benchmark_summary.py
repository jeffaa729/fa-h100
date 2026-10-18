#!/usr/bin/env python3

"""Rebuild the timing index from saved Nsight family CSV files."""

import csv
import sys
from pathlib import Path


from extract_ncu import result_table


def main():
    result_dir = Path(sys.argv[1])
    reports = []
    for path in sorted(result_dir.glob("*.csv")):
        if path.stem not in {"matmul", "flash_attention"}:
            continue
        with path.open(newline="", encoding="utf-8") as source:
            measurements = list(csv.DictReader(source))
        reports.append(f"{path.stem}\n\n" + result_table(measurements, path.stem))
    (result_dir / "summary.md").write_text("\n".join(reports), encoding="utf-8")



if __name__ == "__main__":
    main()
