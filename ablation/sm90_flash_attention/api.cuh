#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace gpu_kernels::fa_ablation {
using bf16 = __nv_bfloat16;
using Forward = void (*)(bf16*, float*, const bf16*, const bf16*, const bf16*, int, cudaStream_t);

#define DECLARE_STAGE(name) namespace name { void forward(bf16* O, float* lse, const bf16* Q, const bf16* K, const bf16* V, int T, cudaStream_t stream); }
DECLARE_STAGE(a1)
DECLARE_STAGE(a2)
DECLARE_STAGE(a3)
DECLARE_STAGE(a4)
DECLARE_STAGE(a5)
DECLARE_STAGE(a6)
DECLARE_STAGE(a7)
DECLARE_STAGE(a8)
#undef DECLARE_STAGE
}
