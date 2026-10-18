// A fixed BF16 causal MHA ABI: B=4, H=8, D=128, varying only sequence length.
// PyTorch owns the buffers and stream; errors remain inside this C++ boundary.
#include "api.cuh"
#include "../../kernels/attention/flash_attention/common.cuh"

#include <exception>
#include <stdexcept>
#include <string>

namespace {
thread_local std::string error_message;
}

extern "C" const char* attention_ablation_last_error() {
    return error_message.c_str();
}

extern "C" int attention_ablation_forward(int stage, __nv_bfloat16* O, float* lse, const __nv_bfloat16* Q, const __nv_bfloat16* K,
                                          const __nv_bfloat16* V, int T, cudaStream_t stream) {
    try {
        using namespace gpu_kernels::fa_ablation;
        const Forward functions[] = {a1::forward, a2::forward, a3::forward, a4::forward, a5::forward, a6::forward, a7::forward, a8::forward};
        if (stage == 9) {
            // Untimed anchor check against the unchanged production Hopper kernel.
            gpu_kernels::flash_attention_forward_sm90_cuda(O, lse, Q, K, V, 4, T, 8, 8, 128, 0.08838834764831845F, stream);
        } else {
            if (stage < 1 || stage > 8) throw std::runtime_error("stage must be A1-A8");
            functions[stage - 1](O, lse, Q, K, V, T, stream);
        }
        return 0;
    } catch (const std::exception& error) {
        error_message = error.what();
        return 1;
    }
}
