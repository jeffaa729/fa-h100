#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${GPU_KERNELS_PYTHON:-$repo_root/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
    echo "Run uv sync --locked first." >&2
    exit 1
fi
family="${1:-all}"
(( $# == 0 )) || shift
mkdir -p "$repo_root/build"
if ! bash "$repo_root/scripts/build.sh" >"$repo_root/build/test_build.log" 2>&1; then
    cat "$repo_root/build/test_build.log" >&2
    exit 1
fi
exec "$python_bin" "$repo_root/benchmarks/run.py" "$family" --test "$@"
