// Exposes the native BF16 attention operator to the shared Python comparison harness without depending on PyTorch C++ headers.
// PyTorch owns the buffers and stream, while exceptions are returned as error strings instead of crossing the C ABI.

#include "flash_attention.h"

#include <exception>
#include <string>

namespace {
thread_local std::string last_error;
}

extern "C" const char* gpu_kernels_flash_last_error() {
    return last_error.c_str();
}

extern "C" int gpu_kernels_flash_forward(
    __nv_bfloat16* output, float* logsumexp,
    const __nv_bfloat16* query, const __nv_bfloat16* key, const __nv_bfloat16* value,
    int batch, int sequence, int query_heads, int key_value_heads, int dimension, float scale, cudaStream_t stream) {
    try {
        gpu_kernels::flash_attention_forward_cuda(output, logsumexp, query, key, value,
            batch, sequence, query_heads, key_value_heads, dimension, scale, stream);
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 1;
    }
}
