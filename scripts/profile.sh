#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$repo_root/build"
report_dir="$repo_root/profiles/reports"
result_dir="$repo_root/profiles/results"
python_bin="${GPU_KERNELS_PYTHON:-$repo_root/.venv/bin/python}"
family="${1:-}"
[[ "$family" != "gemm" ]] || family=matmul
suite="${2:-quick}"
run_mode="${3:-profile}"

usage() {
    echo "usage: bash scripts/profile.sh {matmul|flash_attention} [quick|full|h100] [profile|extract]" >&2
}

if [[ "$family" == "--help" || "$family" == "-h" ]]; then
    usage
    exit 0
fi

if [[ -z "$family"
    || "$suite" != "quick" && "$suite" != "full" && "$suite" != "h100"
    || "$run_mode" != "profile" && "$run_mode" != "extract" ]]; then
    usage
    exit 1
fi

if [[ ! -x "$python_bin" ]]; then
    echo "uv environment not found; run: uv sync" >&2
    exit 1
fi

nvcc_bin="${CUDACXX:-}"
if [[ -z "$nvcc_bin" && -f "$build_dir/CMakeCache.txt" ]]; then
    nvcc_bin="$(
        sed -n 's/^CMAKE_CUDA_COMPILER:[^=]*=//p' \
            "$build_dir/CMakeCache.txt" | head -n 1
    )"
fi
if [[ -z "$nvcc_bin" && -x /usr/local/cuda/bin/nvcc ]]; then
    nvcc_bin=/usr/local/cuda/bin/nvcc
fi
if [[ -z "$nvcc_bin" ]]; then
    nvcc_bin="$(
        for candidate in /usr/local/cuda-*/bin/nvcc; do
            [[ -x "$candidate" ]] && printf '%s\n' "$candidate"
        done | sort -V | tail -n 1
    )"
fi

ncu_bin=""
for candidate in /usr/local/cuda/bin/ncu /usr/local/cuda-*/bin/ncu; do
    [[ -x "$candidate" ]] && ncu_bin="$candidate"
done
if [[ -z "$nvcc_bin" || -z "$ncu_bin" ]]; then
    echo "CUDA compiler or Nsight Compute was not found under /usr/local/cuda*" >&2
    exit 1
fi

cuda_arch="${GPU_KERNELS_CUDA_ARCH:-}"
if [[ -z "$cuda_arch" ]] && command -v nvidia-smi >/dev/null; then
    cuda_arch="$(
        nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
            | head -n 1 | tr -d '. '
    )"
fi
cuda_arch="${cuda_arch:-89}"

case "$family" in
    matmul)
        test_regex='^matmul$'
        build_targets=(gpu_kernels_operator_bench benchmark_matmul)
        ;;
    flash_attention)
        test_regex='^flash_attention$'
        build_targets=(gpu_kernels_flash_attention_bench benchmark_flash_attention)
        ;;
    *)
        usage
        exit 1
        ;;
esac

mkdir -p "$build_dir"
if ! CUDACXX="$nvcc_bin" GPU_KERNELS_CUDA_ARCH="$cuda_arch" \
        bash "$repo_root/scripts/build.sh" "${build_targets[@]}" \
        >"$build_dir/${family}_build.log" 2>&1; then
    cat "$build_dir/${family}_build.log" >&2
    exit 1
fi
if ! ctest --test-dir "$build_dir" --output-on-failure -R "$test_regex" \
        >"$build_dir/${family}_tests.log" 2>&1; then
    cat "$build_dir/${family}_tests.log" >&2
    exit 1
fi

mkdir -p "$report_dir" "$result_dir"
report_arguments=()

gpu_info="$(
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader \
        2>/dev/null | head -n 1 || true
)"
cuda_version="$($nvcc_bin --version | sed -n 's/.*release \([^,]*\).*/\1/p')"
ncu_version="$($ncu_bin --version | tail -n 1)"
flash_version="$(
    "$python_bin" -c \
        'import importlib.metadata; print(importlib.metadata.version("flash-attn"))' \
        2>/dev/null || printf 'not installed'
)"
{
    printf '# %s benchmark environment\n\n' "$family"
    printf -- '- GPU: %s\n' "${gpu_info:-unavailable}"
    printf -- '- CUDA compiler: %s\n' "${cuda_version:-unknown}"
    printf -- '- Nsight Compute: %s\n' "$ncu_version"
    printf -- '- CUDA architecture: sm_%s\n' "$cuda_arch"
    printf -- '- Build: CMake Release\n'
    printf -- '- gpu_kernels commit: %s\n' "$(git -C "$repo_root" rev-parse HEAD)"
    printf -- '- Shape suite: %s\n' "$suite"
    printf -- '- flash-attn package: %s\n' "$flash_version"
    printf -- '- flash-attn commit: %s\n' \
        "${GPU_KERNELS_FLASH_ATTN_COMMIT:-not recorded}"
} >"$result_dir/${family}_environment.md"

profile_case() {
    local label="$1"
    local stem="$2"
    local kernel_pattern="$3"
    shift 3
    local speed_report="$report_dir/${family}_${stem}_speed.ncu-rep"
    local metrics_report="$report_dir/${family}_${stem}_metrics.ncu-rep"

    if [[ "$run_mode" == "profile" ]]; then
        if ! "$ncu_bin" \
            --profile-from-start off \
            --cache-control none \
            --section SpeedOfLight \
            --kernel-name "$kernel_pattern" \
            --export "$speed_report" \
            --force-overwrite \
            "$@" >"$speed_report.log" 2>&1; then
            cat "$speed_report.log" >&2
            echo "Nsight profiling failed; counter-free comparisons are available in scripts/benchmark.sh." >&2
            return 1
        fi
        if ! "$ncu_bin" \
            --profile-from-start off \
            --cache-control all \
            --section LaunchStats \
            --section Occupancy \
            --section MemoryWorkloadAnalysis \
            --metrics dram__bytes_read.sum,dram__bytes_write.sum \
            --kernel-name "$kernel_pattern" \
            --export "$metrics_report" \
            --force-overwrite \
            "$@" >"$metrics_report.log" 2>&1; then
            cat "$metrics_report.log" >&2
            return 1
        fi
    else
        [[ -f "$speed_report" && -f "$metrics_report" ]] || {
            echo "missing report pair for $label" >&2
            exit 1
        }
    fi
    report_arguments+=("$label" "$speed_report" "$metrics_report")
}

profile_matmul() {
    local sizes=(2048)
    [[ "$suite" != "quick" ]] && sizes=(2048 4096 8192)
    local backends=(fp32 bf16 cublas_fp32 cublas_bf16)
    [[ "$cuda_arch" != 90 && "$cuda_arch" != 90a ]] || backends=(bf16 cublas_bf16)
    local size backend pattern label stem
    for size in "${sizes[@]}"; do
        for backend in "${backends[@]}"; do
            pattern='regex:.*matmul_kernel.*'
            [[ "$backend" == "bf16" ]] && \
                pattern='regex:.*(matmul_tensor_core_.*|gemm_bf16_)kernel.*'
            [[ "$backend" == cublas_* ]] && pattern='regex:.*'
            label="M=${size},N=${size},K=${size}/${backend}/TN"
            stem="${size}_${backend}_TN"
            profile_case \
                "$label" "$stem" "$pattern" \
                "$build_dir/benchmark_matmul" \
                "$size" "$size" "$size" "$backend"
        done
    done
}

official_flash_available() {
    "$python_bin" -c 'import torch; from flash_attn import flash_attn_func' \
        >/dev/null 2>&1
}

profile_flash_attention() {
    local cases=("1 512 8 128")
    if [[ "$suite" != "quick" ]]; then
        cases=()
        local batch sequence heads
        for batch in 1 4; do
            for sequence in 128 256 512 1024 2048; do
                for heads in 4 8; do
                    cases+=("$batch $sequence $heads 128")
                done
            done
        done
    fi

    local have_official=0
    if official_flash_available; then
        have_official=1
    elif [[ "${GPU_KERNELS_REQUIRE_EXTERNAL:-0}" == "1" ]]; then
        echo "PyTorch and the official flash-attn package are required" >&2
        exit 1
    else
        echo "Official flash-attn is not installed; profiling the custom kernel only." >&2
    fi

    local case_spec batch sequence heads dimension operation shape stem
    local custom_dump official_dump
    for case_spec in "${cases[@]}"; do
        read -r batch sequence heads dimension <<<"$case_spec"
        shape="B=${batch},T=${sequence},H=${heads},D=${dimension}"
        stem="B${batch}_T${sequence}_H${heads}_D${dimension}"

        if (( have_official )); then
            custom_dump="${TMPDIR:-/tmp}/gpu_kernels_${stem}_custom.bin"
            official_dump="${TMPDIR:-/tmp}/gpu_kernels_${stem}_official.bin"
            "$build_dir/benchmark_flash_attention" \
                "$batch" "$sequence" "$heads" "$dimension" forward \
                "$custom_dump" >/dev/null
            "$python_bin" "$repo_root/reference/python/flash_attention_official.py" \
                "$batch" "$sequence" "$heads" "$dimension" forward \
                "$official_dump" >/dev/null
            if ! "$python_bin" "$repo_root/scripts/compare_attention_dumps.py" \
                "$custom_dump" "$official_dump" \
                "$batch" "$sequence" "$heads" "$dimension" \
                >"$build_dir/flash_attention_comparison.log" 2>&1; then
                cat "$build_dir/flash_attention_comparison.log" >&2
                exit 1
            fi
            rm -f -- "$custom_dump" "$official_dump"
        fi

        for operation in forward; do
            profile_case \
                "$shape/custom/$operation" "${stem}_custom_${operation}" \
                'regex:flash_attention_.*_kernel' \
                "$build_dir/benchmark_flash_attention" \
                "$batch" "$sequence" "$heads" "$dimension" "$operation"
            if (( have_official )); then
                profile_case \
                    "$shape/official/$operation" \
                    "${stem}_official_${operation}" 'regex:.*flash.*' \
                    "$python_bin" \
                    "$repo_root/reference/python/flash_attention_official.py" \
                    "$batch" "$sequence" "$heads" "$dimension" "$operation"
            fi
        done
    done
}

case "$family" in
    matmul) profile_matmul ;;
    flash_attention) profile_flash_attention ;;
esac

"$python_bin" "$repo_root/scripts/extract_ncu.py" \
    --csv-out "$result_dir/${family}.csv" \
    --markdown-out "$result_dir/${family}.md" \
    "$ncu_bin" "${report_arguments[@]}"
"$python_bin" "$repo_root/scripts/update_benchmark_summary.py" "$result_dir"
