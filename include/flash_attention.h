#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace gpu_kernels {

// Fuses causal QK^T, online softmax, and PV without materializing the T x T
// score or probability matrices. Q/O use BF16 [B,T,Hq,128], K/V use BF16
// [B,T,Hkv,128], and the natural-log LSE uses FP32 [B,Hq,T]. Hq must be
// divisible by Hkv, and T must be a positive multiple of 64.
void flash_attention_forward_cuda(
    __nv_bfloat16* output,
    float* logsumexp,
    const __nv_bfloat16* query,
    const __nv_bfloat16* key,
    const __nv_bfloat16* value,
    int batch_size,
    int sequence_length,
    int query_heads,
    int key_value_heads,
    int head_size,
    float scale,
    cudaStream_t stream = nullptr);

}  // namespace gpu_kernels
