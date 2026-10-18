cmake_minimum_required(VERSION 3.24)
set(CMAKE_CUDA_ARCHITECTURES 90a CACHE STRING "H100 ablation target")
project(attention_ablation LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
find_package(CUDAToolkit REQUIRED)

add_library(attention_ablation SHARED
    bridge.cu
    a1_fa2.cu
    a2_tma_wgmma.cu
    a3_kv_pipeline.cu
    a4_registers.cu
    a5_ping_pong.cu
    a6_intra_wg.cu
    a7_softmax.cu
    a8_scheduler_epilogue.cu
    ../../kernels/common/cuda_common.cu
    ../../kernels/attention/flash_attention/sm89_forward.cu
    ../../kernels/attention/flash_attention/sm90_forward.cu
)
target_include_directories(attention_ablation PRIVATE ../../include)
target_link_libraries(attention_ablation PRIVATE CUDA::cudart CUDA::cuda_driver)
target_compile_options(attention_ablation PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>)
set_target_properties(attention_ablation PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
