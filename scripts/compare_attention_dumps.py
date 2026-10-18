#!/usr/bin/env python3

"""Compares custom and official FlashAttention raw correctness dumps.
Each dump stores the forward output as FP32."""

import argparse
import array
import math
from pathlib import Path


def read(path):
    values = array.array("f")
    with path.open("rb") as source:
        values.fromfile(source, path.stat().st_size // values.itemsize)
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("custom", type=Path)
    parser.add_argument("official", type=Path)
    parser.add_argument("batch", type=int)
    parser.add_argument("sequence", type=int)
    parser.add_argument("heads", type=int)
    parser.add_argument("head_size", type=int)
    parser.add_argument("--tolerance", type=float, default=2.5e-2)
    args = parser.parse_args()

    custom = read(args.custom)
    official = read(args.official)
    elements = args.batch * args.sequence * args.heads * args.head_size
    expected = elements
    if len(custom) != expected or len(official) != expected:
        raise SystemExit(
            f"expected {expected} floats per dump; got "
            f"custom={len(custom)} official={len(official)}"
        )

    passed = True
    print("FlashAttention cross-implementation correctness")
    print(f"{'tensor':<10} {'max abs':>12} {'RMS':>12} {'result':>9}")
    for tensor_index, name in enumerate(("output",)):
        begin = tensor_index * elements
        end = begin + elements
        squared_error = 0.0
        maximum_error = 0.0
        for left, right in zip(custom[begin:end], official[begin:end]):
            error = abs(left - right)
            maximum_error = max(maximum_error, error)
            squared_error += error * error
        rms = math.sqrt(squared_error / elements)
        current = maximum_error <= args.tolerance
        passed &= current
        print(
            f"{name:<10} {maximum_error:>12.4e} {rms:>12.4e} "
            f"{('PASS' if current else 'FAIL'):>9}"
        )

    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
