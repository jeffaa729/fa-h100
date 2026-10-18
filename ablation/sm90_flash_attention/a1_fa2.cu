// Reuse the production FA2-style mma.sync/cp.async kernel, compiled for H100.
// Call it directly: the production dispatcher would otherwise select SM90.
#include "api.cuh"
#include "../../kernels/attention/flash_attention/common.cuh"

namespace gpu_kernels::fa_ablation::a1 {
void forward(bf16* O, float* lse, const bf16* Q, const bf16* K, const bf16* V, int T, cudaStream_t stream) {
    flash_attention_forward_sm89_cuda(O, lse, Q, K, V, 4, T, 8, 8, 128, 0.08838834764831845F, stream);
}
}
