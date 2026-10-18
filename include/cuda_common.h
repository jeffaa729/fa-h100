#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace gpu_kernels {

void cuda_check(cudaError_t result, const char* expression, const char* file, int line);

#define CUDA_CHECK(expression) \
    ::gpu_kernels::cuda_check((expression), #expression, __FILE__, __LINE__)

void* device_malloc(std::size_t bytes);
void device_free(void* pointer);
void synchronize(cudaStream_t stream = nullptr);

}  // namespace gpu_kernels
