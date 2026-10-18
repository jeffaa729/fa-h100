// BF16 D128 causal Hopper forward, specialized to the fixed MHA/GQA/MQA suite.
#include "common.cuh"
#include "cuda_common.h"

#include <cuda.h>
#include <cudaTypedefs.h>

#include <cfloat>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace gpu_kernels {
namespace {

using bf16 = __nv_bfloat16;

// Causal BF16 D128 uses 128x128 query/key tiles.
// Two consumer warp groups reuse each K/V tile, owning 64 query rows each.
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int D = 128;
constexpr int HALF_D = 64;
constexpr int WGMMA_K = 16;
constexpr int STAGES = 2;
constexpr int NUM_THREADS = 384;
constexpr int GROUPS = 8;
constexpr int TMA_TILE_BYTES = BM * HALF_D * sizeof(bf16);
constexpr int KV_TRANSACTION_BYTES = 2 * TMA_TILE_BYTES;
constexpr int SMEM_ALIGNMENT = 1024;
constexpr float LOG2E = 1.4426950408889634F;

// Keep P in registers; BF16 V uses an MN-major descriptor,
// without a physical transpose. TMA and WGMMA share the SW128 layout.
struct SharedStorage {
    alignas(SMEM_ALIGNMENT) bf16 Q[2][BM * HALF_D];
    alignas(SMEM_ALIGNMENT) bf16 K[STAGES][2][BN * HALF_D];
    // Epilogue O aliases V. Their lifetimes are disjoint after all consumers finish
    // PV, so no separate 32 KiB output allocation is required.
    union {
        alignas(SMEM_ALIGNMENT) bf16 V[STAGES][2][BN * HALF_D];
        alignas(SMEM_ALIGNMENT) bf16 O[2][BM * HALF_D];
    };
};

constexpr size_t SMEM_BYTES = sizeof(SharedStorage) + SMEM_ALIGNMENT - 1;

CUtensorMap make_attention_map(const bf16* pointer, int batch_size, int sequence_length, int heads, CUtensorMapL2promotion promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE) {
    CUtensorMap map;
    void* address = const_cast<bf16*>(pointer);
    // Transfer a complete Q/K/V tile per instruction. TMA groups D into 64-value bands
    // so a single transfer produces the canonical SW128 descriptor layout.
    const uint64_t global_shape[5] = {HALF_D, static_cast<uint64_t>(sequence_length), static_cast<uint64_t>(heads), 2, static_cast<uint64_t>(batch_size)};
    const uint64_t global_stride[4] = {static_cast<uint64_t>(heads) * D * sizeof(bf16), D * sizeof(bf16), HALF_D * sizeof(bf16), static_cast<uint64_t>(sequence_length) * heads * D * sizeof(bf16)};
    const uint32_t box_shape[5] = {HALF_D, BM, 1, 2, 1};
    const uint32_t element_stride[5] = {1, 1, 1, 1, 1};
    const CUresult result = cuTensorMapEncodeTiled(
        &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, address, global_shape, global_stride, box_shape, element_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        promotion, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (result != CUDA_SUCCESS) {
        const char* name = nullptr;
        cuGetErrorName(result, &name);
        throw std::runtime_error(std::string("cuTensorMapEncodeTiled failed: ") + (name ? name : "unknown driver error"));
    }
    return map;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900

__device__ __forceinline__ uint32_t barrier_address(const uint64_t* barrier) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
}

__device__ __forceinline__ void barrier_init(uint64_t* barrier, uint32_t arrivals) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(barrier_address(barrier)), "r"(arrivals) : "memory");
}

__device__ __forceinline__ void barrier_arrive(uint64_t* barrier) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" :: "r"(barrier_address(barrier)) : "memory");
}

__device__ __forceinline__ void barrier_expect_bytes(uint64_t* barrier, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                 :: "r"(barrier_address(barrier)), "r"(bytes) : "memory");
}

__device__ __forceinline__ void barrier_wait(uint64_t* barrier, int phase) {
    // Test readiness once, then block with 0x989680 suspend ticks instead
    // of repeatedly retrying short waits.
    asm volatile(
        "{ .reg .pred done;\n"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 done, [%0], %1;\n"
        "@done bra wait_done_%=;\n"
        "wait_loop_%=:\n"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 done, [%0], %1, 0x989680;\n"
        "@!done bra wait_loop_%=;\n"
        "wait_done_%=:\n}"
        :: "r"(barrier_address(barrier)), "r"(phase) : "memory");
}

// One elected producer issues one full-tile TMA copy;
// the transaction barrier makes completion visible to both consumer groups.
template <bool Query = false>
__device__ __forceinline__ void tma_load(bf16* destination, const CUtensorMap* map, uint64_t* barrier, int token, int head, int batch) {
    // Prefer evicting Q first while retaining reused K/V in L2.
    constexpr uint64_t cache_hint = Query ? 0x12F0000000000000ULL : 0x14F0000000000000ULL;
    asm volatile("cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {0, %3, %4, 0, %5}], [%2], %6;"
        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(destination))), "l"(map), "r"(barrier_address(barrier)), "r"(token), "r"(head), "r"(batch), "l"(cache_hint) : "memory");
}

__device__ __forceinline__ void advance_stage(int& stage, int& phase) {
    if (++stage == STAGES) {
        stage = 0;
        phase ^= 1;
    }
}

__device__ __forceinline__ uint64_t encode_descriptor(uint64_t value) {
    return (value & 0x3FFFFU) >> 4U;
}

// K-major leading offset is 16 bytes; MN-major V advances one D=64 band.
template <bool MajorMN = false>
__device__ __forceinline__ uint64_t make_smem_descriptor(bf16* pointer) {
    const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
    uint64_t descriptor = encode_descriptor(address);
    descriptor |= encode_descriptor(MajorMN ? BN * 128 : 16) << 16U;
    descriptor |= encode_descriptor(1024) << 32U;
    descriptor |= 1ULL << 62U;
    return descriptor;
}

// Update only the low 32-bit descriptor word; the layout/swizzle word is invariant.
// Offsets are encoded in 16-byte units, not bytes or BF16 element indices.
__device__ __forceinline__ uint64_t advance_descriptor(uint64_t descriptor, uint32_t offset) {
    return (descriptor & 0xffffffff00000000ULL) | (static_cast<uint32_t>(descriptor) + offset);
}

__device__ __forceinline__ void warpgroup_fence() {
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void warpgroup_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}

template <int Pending>
__device__ __forceinline__ void warpgroup_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(Pending) : "memory");
}

// Compiler fences make register dependencies explicit at each asynchronous MMA
// issue site without emitting arithmetic instructions.
__device__ __forceinline__ void fence_scores(float (&scores)[GROUPS][8]) {
#pragma unroll
    for (int g = 0; g < GROUPS; ++g) {
#pragma unroll
        for (int i = 0; i < 8; ++i) asm volatile("" : "+f"(scores[g][i]) :: "memory");
    }
}

__device__ __forceinline__ void fence_probabilities(uint32_t (&P)[GROUPS][4]) {
#pragma unroll
    for (int g = 0; g < GROUPS; ++g) {
#pragma unroll
        for (int i = 0; i < 4; ++i) asm volatile("" : "+r"(P[g][i]) :: "memory");
    }
}

// QK uses shared/shared BF16 operands with FP32 accumulators.
template <int ScaleD>
__device__ __forceinline__ void wgmma_qk(float (&accumulator)[GROUPS][8], uint64_t A, uint64_t B) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, %66, 1, 1, 0, 0;"
        : "+f"(accumulator[0][0]), "+f"(accumulator[0][1]), "+f"(accumulator[0][2]), "+f"(accumulator[0][3]),
          "+f"(accumulator[0][4]), "+f"(accumulator[0][5]), "+f"(accumulator[0][6]), "+f"(accumulator[0][7]),
          "+f"(accumulator[1][0]), "+f"(accumulator[1][1]), "+f"(accumulator[1][2]), "+f"(accumulator[1][3]),
          "+f"(accumulator[1][4]), "+f"(accumulator[1][5]), "+f"(accumulator[1][6]), "+f"(accumulator[1][7]),
          "+f"(accumulator[2][0]), "+f"(accumulator[2][1]), "+f"(accumulator[2][2]), "+f"(accumulator[2][3]),
          "+f"(accumulator[2][4]), "+f"(accumulator[2][5]), "+f"(accumulator[2][6]), "+f"(accumulator[2][7]),
          "+f"(accumulator[3][0]), "+f"(accumulator[3][1]), "+f"(accumulator[3][2]), "+f"(accumulator[3][3]),
          "+f"(accumulator[3][4]), "+f"(accumulator[3][5]), "+f"(accumulator[3][6]), "+f"(accumulator[3][7]),
          "+f"(accumulator[4][0]), "+f"(accumulator[4][1]), "+f"(accumulator[4][2]), "+f"(accumulator[4][3]),
          "+f"(accumulator[4][4]), "+f"(accumulator[4][5]), "+f"(accumulator[4][6]), "+f"(accumulator[4][7]),
          "+f"(accumulator[5][0]), "+f"(accumulator[5][1]), "+f"(accumulator[5][2]), "+f"(accumulator[5][3]),
          "+f"(accumulator[5][4]), "+f"(accumulator[5][5]), "+f"(accumulator[5][6]), "+f"(accumulator[5][7]),
          "+f"(accumulator[6][0]), "+f"(accumulator[6][1]), "+f"(accumulator[6][2]), "+f"(accumulator[6][3]),
          "+f"(accumulator[6][4]), "+f"(accumulator[6][5]), "+f"(accumulator[6][6]), "+f"(accumulator[6][7]),
          "+f"(accumulator[7][0]), "+f"(accumulator[7][1]), "+f"(accumulator[7][2]), "+f"(accumulator[7][3]),
          "+f"(accumulator[7][4]), "+f"(accumulator[7][5]), "+f"(accumulator[7][6]), "+f"(accumulator[7][7])
        : "l"(A), "l"(B), "n"(ScaleD));
}

// PV uses register/shared operands: P is four BF16x2 registers per K=16 step;
// MN-major V is consumed directly from TMA-swizzled shared memory.
__device__ __forceinline__ void wgmma_pv(float (&accumulator)[GROUPS][8], const uint32_t (&A)[4], uint64_t B) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;"
        : "+f"(accumulator[0][0]), "+f"(accumulator[0][1]), "+f"(accumulator[0][2]), "+f"(accumulator[0][3]),
          "+f"(accumulator[0][4]), "+f"(accumulator[0][5]), "+f"(accumulator[0][6]), "+f"(accumulator[0][7]),
          "+f"(accumulator[1][0]), "+f"(accumulator[1][1]), "+f"(accumulator[1][2]), "+f"(accumulator[1][3]),
          "+f"(accumulator[1][4]), "+f"(accumulator[1][5]), "+f"(accumulator[1][6]), "+f"(accumulator[1][7]),
          "+f"(accumulator[2][0]), "+f"(accumulator[2][1]), "+f"(accumulator[2][2]), "+f"(accumulator[2][3]),
          "+f"(accumulator[2][4]), "+f"(accumulator[2][5]), "+f"(accumulator[2][6]), "+f"(accumulator[2][7]),
          "+f"(accumulator[3][0]), "+f"(accumulator[3][1]), "+f"(accumulator[3][2]), "+f"(accumulator[3][3]),
          "+f"(accumulator[3][4]), "+f"(accumulator[3][5]), "+f"(accumulator[3][6]), "+f"(accumulator[3][7]),
          "+f"(accumulator[4][0]), "+f"(accumulator[4][1]), "+f"(accumulator[4][2]), "+f"(accumulator[4][3]),
          "+f"(accumulator[4][4]), "+f"(accumulator[4][5]), "+f"(accumulator[4][6]), "+f"(accumulator[4][7]),
          "+f"(accumulator[5][0]), "+f"(accumulator[5][1]), "+f"(accumulator[5][2]), "+f"(accumulator[5][3]),
          "+f"(accumulator[5][4]), "+f"(accumulator[5][5]), "+f"(accumulator[5][6]), "+f"(accumulator[5][7]),
          "+f"(accumulator[6][0]), "+f"(accumulator[6][1]), "+f"(accumulator[6][2]), "+f"(accumulator[6][3]),
          "+f"(accumulator[6][4]), "+f"(accumulator[6][5]), "+f"(accumulator[6][6]), "+f"(accumulator[6][7]),
          "+f"(accumulator[7][0]), "+f"(accumulator[7][1]), "+f"(accumulator[7][2]), "+f"(accumulator[7][3]),
          "+f"(accumulator[7][4]), "+f"(accumulator[7][5]), "+f"(accumulator[7][6]), "+f"(accumulator[7][7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "l"(B));
}

__device__ __forceinline__ void issue_qk(float (&scores)[GROUPS][8], SharedStorage& shared, int stage, int consumer) {
    fence_scores(scores);
    warpgroup_fence();
    // Build each band's descriptor once, then advance by encoded K-step offsets rather
    // than repeating shared-address masking and layout-field construction.
    const uint64_t Q0 = make_smem_descriptor(shared.Q[0] + consumer * 64 * HALF_D);
    const uint64_t K0 = make_smem_descriptor(shared.K[stage][0]);
    const uint64_t Q1 = make_smem_descriptor(shared.Q[1] + consumer * 64 * HALF_D);
    const uint64_t K1 = make_smem_descriptor(shared.K[stage][1]);
    wgmma_qk<0>(scores, Q0, K0);
#pragma unroll
    for (int k = 1; k < HALF_D / WGMMA_K; ++k) {
        wgmma_qk<1>(scores, advance_descriptor(Q0, k * 2), advance_descriptor(K0, k * 2));
    }
#pragma unroll
    for (int k = 0; k < HALF_D / WGMMA_K; ++k) {
        wgmma_qk<1>(scores, advance_descriptor(Q1, k * 2), advance_descriptor(K1, k * 2));
    }
    warpgroup_commit();
    fence_scores(scores);
}

__device__ __forceinline__ void issue_pv(float (&output)[GROUPS][8], uint32_t (&P)[GROUPS][4], SharedStorage& shared, int stage) {
    fence_scores(output);
    fence_probabilities(P);
    warpgroup_fence();
    const uint64_t V = make_smem_descriptor<true>(shared.V[stage][0]);
#pragma unroll
    for (int k = 0; k < BN / WGMMA_K; ++k) {
        wgmma_pv(output, P[k], advance_descriptor(V, k * WGMMA_K * HALF_D * sizeof(bf16) / 16));
    }
    warpgroup_commit();
    fence_scores(output);
    fence_probabilities(P);
}

// Map groups of eight FP32 scores to four packed A registers,
// using round-to-nearest BF16 conversion for the PV operand.
__device__ __forceinline__ void pack_probabilities(uint32_t (&P)[GROUPS][4], const float (&scores)[GROUPS][8]) {
#pragma unroll
    for (int group = 0; group < GROUPS; ++group) {
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const __nv_bfloat162 packed = __floats2bfloat162_rn(scores[group][pair * 2], scores[group][pair * 2 + 1]);
            P[group][pair] = *reinterpret_cast<const uint32_t*>(&packed);
        }
    }
}

// Use the hardware approximate base-2 exponential for online softmax.
__device__ __forceinline__ float fast_exp2(float value) {
    float result;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
    return result;
}

template <bool First, bool Masked>
__device__ __forceinline__ void online_softmax(float (&scores)[GROUPS][8], float (&row_max)[2], float (&row_sum)[2], float (&previous_scale)[2],
                                               int query_start, int key_start, int consumer_thread, float scale_log2) {
    const int lane = consumer_thread & 31;
    const int warp = consumer_thread >> 5;
    const int top_row = warp * 16 + lane / 4;
    const int bottom_row = top_row + 8;
    // Use four independent maximum chains per row fragment, then a tree merge.
    // Seed the chains from the first four scores to avoid redundant maxima against -inf.
    float partial_max[2][4];

    //
#pragma unroll
    for (int group = 0; group < GROUPS; ++group) {
        const int column = group * 16 + 2 * (lane & 3);
        if constexpr (Masked) {
            scores[group][0] = query_start + top_row >= key_start + column ? scores[group][0] : -FLT_MAX;
            scores[group][1] = query_start + top_row >= key_start + column + 1 ? scores[group][1] : -FLT_MAX;
            scores[group][4] = query_start + top_row >= key_start + column + 8 ? scores[group][4] : -FLT_MAX;
            scores[group][5] = query_start + top_row >= key_start + column + 9 ? scores[group][5] : -FLT_MAX;
            scores[group][2] = query_start + bottom_row >= key_start + column ? scores[group][2] : -FLT_MAX;
            scores[group][3] = query_start + bottom_row >= key_start + column + 1 ? scores[group][3] : -FLT_MAX;
            scores[group][6] = query_start + bottom_row >= key_start + column + 8 ? scores[group][6] : -FLT_MAX;
            scores[group][7] = query_start + bottom_row >= key_start + column + 9 ? scores[group][7] : -FLT_MAX;
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int item = (i & 1) + (i / 2) * 4;
            partial_max[0][i] = group == 0 ? scores[group][item] : fmaxf(partial_max[0][i], scores[group][item]);
            partial_max[1][i] = group == 0 ? scores[group][item + 2] : fmaxf(partial_max[1][i], scores[group][item + 2]);
        }
    }
    float current_max[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        current_max[r] = fmaxf(fmaxf(partial_max[r][0], partial_max[r][1]), fmaxf(partial_max[r][2], partial_max[r][3]));
        // Merge the previous exponent base before the four-lane maximum reduction.
        if constexpr (!First) current_max[r] = fmaxf(current_max[r], row_max[r]);
    }
    current_max[0] = fmaxf(current_max[0], __shfl_xor_sync(0xffffffff, current_max[0], 1));
    current_max[0] = fmaxf(current_max[0], __shfl_xor_sync(0xffffffff, current_max[0], 2));
    current_max[1] = fmaxf(current_max[1], __shfl_xor_sync(0xffffffff, current_max[1], 1));
    current_max[1] = fmaxf(current_max[1], __shfl_xor_sync(0xffffffff, current_max[1], 2));

    float next_max[2] = {current_max[0], current_max[1]};
    // Keep the previous exponent base until the increase exceeds 8 in log2
    // units. P, row_sum and O retain that same base; final normalization and
    // LSE use the retained base too.
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        previous_scale[r] = 1.0F;
        if constexpr (!First) {
            const float change = (row_max[r] - next_max[r]) * scale_log2;
            if (change >= -8.0F) next_max[r] = row_max[r];
            else previous_scale[r] = fast_exp2(change);
        }
    }
    const float maximum_scaled[2] = {next_max[0] * scale_log2, next_max[1] * scale_log2};

    // Compute exponentials before summing them to separate SFU work
    // from the dependent row-sum chains.
#pragma unroll
    for (int row_id = 0; row_id < 2; ++row_id) {
#pragma unroll
        for (int group = 0; group < GROUPS; ++group) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                // Traverse one row fragment at a time.
                const int item = (i & 1) + (i / 2) * 4 + row_id * 2;
                const float exponent = __fmaf_rn(scores[group][item], scale_log2, -maximum_scaled[row_id]);
                scores[group][item] = fast_exp2(exponent);
            }
        }
    }
    // Accumulate four independent FP32 sum chains per row, then combine
    // them as a tree to shorten dependency chains.
    float partial_sum[2][4] = {};
#pragma unroll
    for (int group = 0; group < GROUPS; ++group) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int item = (i & 1) + (i / 2) * 4;
            partial_sum[0][i] = group == 0 ? scores[group][item] : partial_sum[0][i] + scores[group][item];
            partial_sum[1][i] = group == 0 ? scores[group][item + 2] : partial_sum[1][i] + scores[group][item + 2];
        }
    }
    float local_sum[2];
#pragma unroll
    for (int row = 0; row < 2; ++row) {
        local_sum[row] = (partial_sum[row][0] + partial_sum[row][1]) + (partial_sum[row][2] + partial_sum[row][3]);
    }
    row_sum[0] = (First ? 0.0F : row_sum[0] * previous_scale[0]) + local_sum[0];
    row_sum[1] = (First ? 0.0F : row_sum[1] * previous_scale[1]) + local_sum[1];
    row_max[0] = next_max[0];
    row_max[1] = next_max[1];
}

__device__ __forceinline__ void scale_output(float (&output)[GROUPS][8], const float (&factor)[2]) {
    // Use a warp-uniform rescale decision. Skip all O-register multiplications when
    // every row in this warp retained its exponent base (both factors are 1).
    if (!__any_sync(0xffffffff, factor[0] != 1.0F || factor[1] != 1.0F)) return;
#pragma unroll
    for (int group = 0; group < GROUPS; ++group) {
#pragma unroll
        for (int item = 0; item < 8; ++item) output[group][item] *= factor[(item & 2) >> 1];
    }
}

// Finalize row statistics while the final PV is outstanding.
// Row sums, reciprocals and LSE do not depend on the O accumulator registers.
__device__ __forceinline__ void finalize_rows(float (&inverse)[2], float (&lse_values)[2], const float (&row_max)[2], float (&row_sum)[2], float scale) {
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        row_sum[r] += __shfl_xor_sync(0xffffffff, row_sum[r], 1);
        row_sum[r] += __shfl_xor_sync(0xffffffff, row_sum[r], 2);
    }
    // These causal rows have positive sums; compute one approximate reciprocal
    // per row for output normalization.
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(inverse[r]) : "f"(row_sum[r]));
        lse_values[r] = row_max[r] * scale + __logf(row_sum[r]);
    }
}

// Retile register accumulators into SW128 BF16
// shared O, then use one TMA store instead of many small global stores.
__device__ __forceinline__ void store_output(SharedStorage& shared, float* lse, float (&accumulator)[GROUPS][8], const float (&inverse)[2], const float (&lse_values)[2], int batch, int head, int start, int sequence, int heads, int tid) {
    const int lane = tid & 31;
    const int row = (tid >> 5) * 16 + lane / 4;
    // Map four packed accumulator pairs to four 8x8 matrices.
    // stmatrix performs the register-to-SW128 retile
    // collectively, replacing 32 scalar shared stores with eight instructions.
#pragma unroll
    for (int g = 0; g < GROUPS; ++g) {
        uint32_t packed[4];
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const int r = pair & 1;
            const __nv_bfloat162 value = __floats2bfloat162_rn(accumulator[g][pair * 2] * inverse[r], accumulator[g][pair * 2 + 1] * inverse[r]);
            packed[pair] = *reinterpret_cast<const uint32_t*>(&value);
        }
        const int local_row = (start % BM) + (tid >> 5) * 16 + (lane & 15);
        const int column = g * 16 + (lane >> 4) * 8;
        const int offset = local_row * HALF_D + ((column % HALF_D) ^ ((local_row & 7) << 3));
        const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(shared.O[column / HALF_D] + offset));
        asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};"
            :: "r"(address), "r"(packed[0]), "r"(packed[1]), "r"(packed[2]), "r"(packed[3]) : "memory");
    }
    if ((lane & 3) == 0) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int token = start + row + r * 8;
            if (token < sequence) lse[(batch * heads + head) * sequence + token] = lse_values[r];
        }
    }
}

#endif

__global__ __launch_bounds__(NUM_THREADS) void flash_attention_forward_sm90_kernel(
    float* logsumexp, const __grid_constant__ CUtensorMap Q_map, const __grid_constant__ CUtensorMap K_map,
    const __grid_constant__ CUtensorMap V_map, const __grid_constant__ CUtensorMap O_map, int sequence_length, int query_heads, int key_value_heads,
    int section_heads, float scale) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    extern __shared__ __align__(16) unsigned char storage[];
    const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(storage));
    const uint32_t aligned = (address + SMEM_ALIGNMENT - 1) & ~(SMEM_ALIGNMENT - 1);
    auto& shared = *reinterpret_cast<SharedStorage*>(__cvta_shared_to_generic(aligned));
    __shared__ __align__(8) uint64_t query_full, k_full[STAGES], v_full[STAGES], k_empty[STAGES], v_empty[STAGES];
    if (threadIdx.x == 0) {
        // Prefetch Q/K/V/O tensor-map descriptors before pipeline initialization.
        asm volatile("prefetch.tensormap [%0];" :: "l"(&Q_map) : "memory");
        asm volatile("prefetch.tensormap [%0];" :: "l"(&K_map) : "memory");
        asm volatile("prefetch.tensormap [%0];" :: "l"(&V_map) : "memory");
        asm volatile("prefetch.tensormap [%0];" :: "l"(&O_map) : "memory");
        barrier_init(&query_full, 1);
#pragma unroll
        for (int s = 0; s < STAGES; ++s) {
            barrier_init(&k_full[s], 1);
            barrier_init(&v_full[s], 1);
            barrier_init(&k_empty[s], 2);
            barrier_init(&v_empty[s], 2);
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();
    const int group = threadIdx.x / 128;
    const int tid = threadIdx.x % 128;
    // Static scheduling processes the longest causal tiles first within L2-sized head
    // sections. GQA's adjacent query heads reuse one KV head; larger MHA
    // workloads need multiple sections instead of interleaving every head.
    // Fixed B={1,4}, Hq=8 shapes divide these power-of-two sections exactly.
    const int query_blocks = (sequence_length + BM - 1) / BM;
    const int section_tiles = section_heads * query_blocks;
    const int section = blockIdx.x / section_tiles;
    const int section_tile = blockIdx.x % section_tiles;
    const int query_block = query_blocks - 1 - section_tile / section_heads;
    const int head_batch = section * section_heads + section_tile % section_heads;
    const int start = query_block * BM;
    const int head = head_batch % query_heads;
    const int batch = head_batch / query_heads;
    const int kv_head = head / (query_heads / key_value_heads);

    // Warp specialization redistributes registers: producer 24, consumers 240.
    if (group == 0) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 24;");
        if (tid == 0) {
            barrier_expect_bytes(&query_full, 2 * TMA_TILE_BYTES);
            tma_load<true>(shared.Q[0], &Q_map, &query_full, start, head, batch);
            int stage = 0, phase = 0;
            barrier_wait(&k_empty[stage], phase);
            barrier_expect_bytes(&k_full[stage], KV_TRANSACTION_BYTES);
            tma_load(shared.K[stage][0], &K_map, &k_full[stage], start, kv_head, batch);
            // Prime K(current), then load K(next) before V(current).
            // QK gets one tile of lookahead;
            // independent empty barriers still protect both circular buffers.
            // Unroll two iterations of the non-transposed TMA K/V loop.
#pragma unroll 2
            for (int iteration = 0; iteration <= query_block; ++iteration) {
                const int token = (query_block - iteration) * BN;
                int next_stage = stage, next_phase = phase;
                advance_stage(next_stage, next_phase);
                if (iteration < query_block) {
                    barrier_wait(&k_empty[next_stage], next_phase);
                    barrier_expect_bytes(&k_full[next_stage], KV_TRANSACTION_BYTES);
                    tma_load(shared.K[next_stage][0], &K_map, &k_full[next_stage], token - BN, kv_head, batch);
                }
                barrier_wait(&v_empty[stage], phase);
                barrier_expect_bytes(&v_full[stage], KV_TRANSACTION_BYTES);
                tma_load(shared.V[stage][0], &V_map, &v_full[stage], token, kv_head, batch);
                stage = next_stage;
                phase = next_phase;
            }
        }
        return;
    }
    asm volatile("setmaxnreg.inc.sync.aligned.u32 240;");
    // Ping-pong scheduling: the first consumer self-signals,
    // then the groups alternate Tensor Core issue while their peer does softmax.
    if (group == 1) asm volatile("bar.arrive 1, 256;" ::: "memory");
    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; ++s) {
            barrier_arrive(&k_empty[s]);
            barrier_arrive(&v_empty[s]);
        }
    }
    barrier_wait(&query_full, 0);
    const int consumer = group - 1;
    const int query_start = start + consumer * 64;
    float O[GROUPS][8] = {};
    uint32_t P[GROUPS][4];
    float row_max[2] = {-FLT_MAX, -FLT_MAX}, row_sum[2] = {}, factor[2];
    const float scale_log2 = scale * LOG2E;
    int stage = 0, phase = 0;
    barrier_wait(&k_full[stage], phase);
    // The first QK MMA uses ScaleD=0 to overwrite score registers,
    // avoiding explicit CUDA-core zeroing.
    float scores[GROUPS][8];
    issue_qk(scores, shared, stage, consumer);
    warpgroup_wait<0>();
    if (tid == 0) barrier_arrive(&k_empty[stage]);
    online_softmax<true, true>(scores, row_max, row_sum, factor, query_start, start, tid, scale_log2);
    pack_probabilities(P, scores);
    int value_stage = stage, value_phase = phase;
    advance_stage(stage, phase);

    // Intra-warpgroup overlap: QK(next) precedes
    // PV(current). wait_group<1> exposes scores while PV runs alongside
    // CUDA-core softmax; wait_group<0> precedes rescaling the O registers.
    // Keep the unmasked loop rolled (unroll 1),
    // avoiding multiple large softmax/WGMMA bodies in the instruction stream.
#pragma unroll 1
    for (int iteration = 1; iteration <= query_block; ++iteration) {
        // Only consumer WG0 waits for K. Its scheduler handoff makes that acquire visible to
        // WG1; both groups still release K after their own WGMMA completes.
        if (consumer == 0) barrier_wait(&k_full[stage], phase);
        // The first QK WGMMA overwrites every score register.
        float next_scores[GROUPS][8];
        asm volatile("bar.sync %0, 256;" :: "r"(group) : "memory");
        issue_qk(next_scores, shared, stage, consumer);
        // Only consumer WG0 waits for V. Signal the peer only after issuing PV,
        // preserving the QK -> PV -> handoff order.
        if (consumer == 0) barrier_wait(&v_full[value_stage], value_phase);
        issue_pv(O, P, shared, value_stage);
        asm volatile("bar.arrive %0, 256;" :: "r"(3 - group) : "memory");
        warpgroup_wait<1>();
        if (tid == 0) barrier_arrive(&k_empty[stage]);
        online_softmax<false, false>(next_scores, row_max, row_sum, factor, query_start, (query_block - iteration) * BN, tid, scale_log2);
        warpgroup_wait<0>();
        if (tid == 0) barrier_arrive(&v_empty[value_stage]);
        // Pack next probabilities before rescaling the completed output accumulator.
        pack_probabilities(P, next_scores);
        scale_output(O, factor);
        value_stage = stage;
        value_phase = phase;
        advance_stage(stage, phase);
    }
    barrier_wait(&v_full[value_stage], value_phase);
    issue_pv(O, P, shared, value_stage);
    float inverse[2], lse_values[2];
    // Finalize row statistics before warpgroup_wait<0>, overlapping
    // the final softmax reduction with PV.
    finalize_rows(inverse, lse_values, row_max, row_sum, scale);
    warpgroup_wait<0>();
    if (tid == 0) barrier_arrive(&v_empty[value_stage]);
    // Synchronize all epilogue threads: every warp group must stop reading V
    // before shared O overwrites it.
    asm volatile("bar.sync 3, 256;" ::: "memory");
    store_output(shared, logsumexp, O, inverse, lse_values, batch, head, query_start, sequence_length, query_heads, tid);
    // Make shared writes visible and synchronize before the elected TMA store, then wait for
    // shared-source reads to finish before a CTA can release its storage.
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("bar.sync 3, 256;" ::: "memory");
    if (group == 1 && tid == 0) {
        asm volatile("cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group [%0, {0, %2, %3, 0, %4}], [%1];"
            :: "l"(&O_map), "r"(static_cast<uint32_t>(__cvta_generic_to_shared(shared.O[0]))), "r"(start), "r"(head), "r"(batch) : "memory");
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
    }
#endif
}

}  // namespace

void flash_attention_forward_sm90_cuda(bf16* output, float* logsumexp, const bf16* query, const bf16* key, const bf16* value, int batch_size,
                                       int sequence_length, int query_heads, int key_value_heads, int head_size, float scale, cudaStream_t stream) {
    // Promote Q/K/V TMA reads to 256-byte L2 fetches; retain the existing SW128
    // layout and eviction hints. Output stores keep the default promotion policy.
    constexpr auto promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
    const CUtensorMap Q_map = make_attention_map(query, batch_size, sequence_length, query_heads, promotion);
    const CUtensorMap K_map = make_attention_map(key, batch_size, sequence_length, key_value_heads, promotion);
    const CUtensorMap V_map = make_attention_map(value, batch_size, sequence_length, key_value_heads, promotion);
    const CUtensorMap O_map = make_attention_map(output, batch_size, sequence_length, query_heads);
    // Budget 32 MiB for K/V, round the KV-head capacity down to a power of two,
    // then account for GQA reuse. The fixed suite needs no residual section.
    const int kv_head_bytes = sequence_length * 2 * D * sizeof(bf16);
    const int kv_head_capacity = 32 * 1024 * 1024 / kv_head_bytes;
    int section_heads = 1;
    while (section_heads * 2 <= kv_head_capacity) section_heads *= 2;
    section_heads *= query_heads / key_value_heads;
    if (section_heads > batch_size * query_heads) section_heads = batch_size * query_heads;
    CUDA_CHECK(cudaFuncSetAttribute(flash_attention_forward_sm90_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    const dim3 grid(((sequence_length + BM - 1) / BM) * query_heads * batch_size);
    flash_attention_forward_sm90_kernel<<<grid, NUM_THREADS, SMEM_BYTES, stream>>>(
        logsumexp, Q_map, K_map, V_map, O_map, sequence_length, query_heads, key_value_heads, section_heads, scale);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace gpu_kernels
