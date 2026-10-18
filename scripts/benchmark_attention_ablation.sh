#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${GPU_KERNELS_PYTHON:-$repo_root/.venv/bin/python}"
nvcc_bin="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
mkdir -p "$repo_root/build/attention_ablation"
build_log="$repo_root/build/attention_ablation/build.log"

if ! {
    cmake -S "$repo_root/ablation/sm90_flash_attention" -B "$repo_root/build/attention_ablation" \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="$nvcc_bin" -DCMAKE_CUDA_ARCHITECTURES=90a && \
    cmake --build "$repo_root/build/attention_ablation" -j "${GPU_KERNELS_BUILD_JOBS:-4}"
} >"$build_log" 2>&1; then
    tail -n 60 "$build_log" >&2
    exit 1
fi
"$python_bin" "$repo_root/ablation/sm90_flash_attention/run.py" "$@"
