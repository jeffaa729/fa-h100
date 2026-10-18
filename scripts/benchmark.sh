#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${GPU_KERNELS_PYTHON:-$repo_root/.venv/bin/python}"
if [[ $# == 0 || "$1" == "--help" || "$1" == "-h" ]]; then
    echo "usage: bash scripts/benchmark.sh FAMILY [quick|full|h100] [options]"
    echo "families: matmul (gemm), flash_attention, all"
    echo "options: --reference cublas|pytorch|fa2|fa3|fa4|all, --operation NAME"
    echo "matmul: cuBLAS is the only reference"
    echo "MegaMoE baseline: bash scripts/benchmark_moe.sh [2|4] [options]"
    exit 0
fi
family="$1"
shift
suite=quick
if (( $# )) && [[ "$1" != --* ]]; then suite="$1"; shift; fi
if [[ ! -x "$python_bin" ]]; then
    echo "Run uv sync --locked first." >&2
    exit 1
fi
mkdir -p "$repo_root/build"
if ! bash "$repo_root/scripts/build.sh" >"$repo_root/build/benchmark_build.log" 2>&1; then
    cat "$repo_root/build/benchmark_build.log" >&2
    exit 1
fi
options=()
reference="${GPU_KERNELS_REFERENCE_BACKEND:-}"
[[ -z "$reference" ]] || options+=(--reference "$reference")
if ! "$python_bin" "$repo_root/benchmarks/run.py" "$family" --suite "$suite" \
        --warmup-ms "${GPU_KERNELS_BENCHMARK_WARMUP_MS:-1000}" \
        --graph-operations "${GPU_KERNELS_GRAPH_OPERATIONS:-10}" \
        --trials "${GPU_KERNELS_BENCHMARK_TRIALS:-9}" \
        "${options[@]}" "$@" >"$repo_root/build/${family}_runtime.log" 2>&1; then
    cat "$repo_root/build/${family}_runtime.log" >&2
    exit 1
fi
# Keep successful output table-only; retain import warnings and errors in the log.
sed -n '/^|/p' "$repo_root/build/${family}_runtime.log"
