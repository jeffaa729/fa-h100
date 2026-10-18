cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()
project(gpu-kernels LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

find_package(CUDAToolkit REQUIRED)
enable_testing()

add_library(gpu_kernels STATIC
    kernels/common/cuda_common.cu
    kernels/attention/flash_attention/dispatch.cu
    kernels/attention/flash_attention/sm89_forward.cu
    kernels/attention/flash_attention/sm90_forward.cu
)
target_include_directories(gpu_kernels PUBLIC include)
set_source_files_properties(
    kernels/attention/flash_attention/sm90_forward.cu
    PROPERTIES COMPILE_OPTIONS "-lineinfo")
target_link_libraries(gpu_kernels PUBLIC CUDA::cudart CUDA::cuda_driver)
set_target_properties(gpu_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON
)

add_executable(benchmark_flash_attention EXCLUDE_FROM_ALL
    benchmarks/attention/benchmark_flash_attention.cu)
target_link_libraries(benchmark_flash_attention PRIVATE gpu_kernels)

add_library(gpu_kernels_flash_attention_bench SHARED
    benchmarks/bindings/flash_attention_bridge.cu)
target_link_libraries(gpu_kernels_flash_attention_bench PRIVATE gpu_kernels)

set(GPU_KERNELS_PYTHON "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python" CACHE FILEPATH "Python from uv sync")
if(EXISTS "${GPU_KERNELS_PYTHON}")
    add_test(NAME flash_attention COMMAND ${GPU_KERNELS_PYTHON}
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/run.py flash_attention --test)
    set_tests_properties(flash_attention PROPERTIES RESOURCE_LOCK gpu)
endif()
