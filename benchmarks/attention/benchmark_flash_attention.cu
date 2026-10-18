// Runs one D128 BF16 causal FlashAttention workload for a same-shape comparison with the official implementation.
// Nsight measures forward; an optional raw dump supports cross-process correctness checks.

#include "cuda_common.h"
#include "flash_attention.h"

#include <cuda_profiler_api.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void dump_values(std::ofstream& output, const __nv_bfloat16* device, std::size_t elements) {
    std::vector<__nv_bfloat16> host(elements);
    CUDA_CHECK(cudaMemcpy(
        host.data(),
        device,
        elements * sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToHost));
    std::vector<float> decoded(elements);
    for (std::size_t index = 0; index < elements; ++index) {
        decoded[index] = __bfloat162float(host[index]);
    }
    output.write(
        reinterpret_cast<const char*>(decoded.data()),
        static_cast<std::streamsize>(elements * sizeof(float)));
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const int batch_size = argc > 1 ? std::atoi(argv[1]) : 2;
        const int sequence_length = argc > 2 ? std::atoi(argv[2]) : 256;
        const int heads = argc > 3 ? std::atoi(argv[3]) : 8;
        const int head_size = argc > 4 ? std::atoi(argv[4]) : 128;
        const char* operation = argc > 5 ? argv[5] : "forward";
        const char* dump_path = argc > 6 ? argv[6] : nullptr;
        if (std::strcmp(operation, "forward") != 0) {
            throw std::runtime_error("only forward is supported");
        }

        const float scale = 1.0F / std::sqrt(static_cast<float>(head_size));
        const std::size_t activations =
            static_cast<std::size_t>(batch_size) * sequence_length * heads
            * head_size;
        const std::size_t rows =
            static_cast<std::size_t>(batch_size) * heads * sequence_length;

        std::vector<__nv_bfloat16> host_query(activations);
        std::vector<__nv_bfloat16> host_key(activations);
        std::vector<__nv_bfloat16> host_value(activations);
        for (std::size_t index = 0; index < activations; ++index) {
            host_query[index] = __float2bfloat16(
                static_cast<float>(
                    static_cast<int>((index * 17) % 101) - 50) / 64.0F);
            host_key[index] = __float2bfloat16(
                static_cast<float>(
                    static_cast<int>((index * 23) % 97) - 48) / 61.0F);
            host_value[index] = __float2bfloat16(
                static_cast<float>(
                    static_cast<int>((index * 31) % 89) - 44) / 59.0F);
        }

        auto* query = static_cast<__nv_bfloat16*>(
            gpu_kernels::device_malloc(activations * sizeof(__nv_bfloat16)));
        auto* key = static_cast<__nv_bfloat16*>(
            gpu_kernels::device_malloc(activations * sizeof(__nv_bfloat16)));
        auto* value = static_cast<__nv_bfloat16*>(
            gpu_kernels::device_malloc(activations * sizeof(__nv_bfloat16)));
        auto* output = static_cast<__nv_bfloat16*>(
            gpu_kernels::device_malloc(activations * sizeof(__nv_bfloat16)));
        auto* logsumexp = static_cast<float*>(
            gpu_kernels::device_malloc(rows * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(
            query,
            host_query.data(),
            activations * sizeof(__nv_bfloat16),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            key,
            host_key.data(),
            activations * sizeof(__nv_bfloat16),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            value,
            host_value.data(),
            activations * sizeof(__nv_bfloat16),
            cudaMemcpyHostToDevice));
        auto forward = [&]() {
            gpu_kernels::flash_attention_forward_cuda(
                output,
                logsumexp,
                query,
                key,
                value,
                batch_size,
                sequence_length,
                heads,
                heads,
                head_size,
                scale);
        };
        forward();
        gpu_kernels::synchronize();
        CUDA_CHECK(cudaProfilerStart());
        forward();
        gpu_kernels::synchronize();
        CUDA_CHECK(cudaProfilerStop());

        if (dump_path != nullptr) {
            forward();
            gpu_kernels::synchronize();
            std::ofstream dump(dump_path, std::ios::binary | std::ios::trunc);
            if (!dump) {
                throw std::runtime_error(
                    std::string("cannot open dump file: ") + dump_path);
            }
            dump_values(dump, output, activations);
        }

        gpu_kernels::device_free(logsumexp);
        gpu_kernels::device_free(output);
        gpu_kernels::device_free(value);
        gpu_kernels::device_free(key);
        gpu_kernels::device_free(query);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(
            stderr, "FlashAttention benchmark failed: %s\n", error.what());
        return 1;
    }
}
