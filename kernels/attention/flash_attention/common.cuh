#pragma once

#include "flash_attention.h"

namespace gpu_kernels {

void flash_attention_forward_sm90_cuda(__nv_bfloat16* output, float* logsumexp, const __nv_bfloat16* query, const __nv_bfloat16* key,
                                       const __nv_bfloat16* value, int batch_size, int sequence_length, int query_heads, int key_value_heads,
                                       int head_size, float scale, cudaStream_t stream);

void flash_attention_forward_sm89_cuda(__nv_bfloat16* output, float* logsumexp, const __nv_bfloat16* query, const __nv_bfloat16* key,
                                       const __nv_bfloat16* value, int batch_size, int sequence_length, int query_heads, int key_value_heads,
                                       int head_size, float scale, cudaStream_t stream);

}  // namespace gpu_kernels
