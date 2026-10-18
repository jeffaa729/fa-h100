// Provides CUDA error handling and allocation/synchronization helpers for Nsight launchers.
// Kernel launches report the failing CUDA call and its source location.

#include "cuda_common.h"

#include <sstream>
#include <stdexcept>

namespace gpu_kernels {

void cuda_check(cudaError_t result, const char* expression, const char* file, int line) {
    if (result == cudaSuccess) {
        return;
    }

    std::ostringstream message;
    message << "CUDA failure at " << file << ':' << line << " while evaluating " << expression << ": " << cudaGetErrorString(result);
    throw std::runtime_error(message.str());
}

void* device_malloc(std::size_t bytes) {
    void* pointer = nullptr;
    CUDA_CHECK(cudaMalloc(&pointer, bytes));
    return pointer;
}

void device_free(void* pointer) {
    CUDA_CHECK(cudaFree(pointer));
}

void synchronize(cudaStream_t stream) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
}


}  // namespace gpu_kernels
