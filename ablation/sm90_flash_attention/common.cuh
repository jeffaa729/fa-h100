// Isolated A2-A8 cumulative Hopper ablation; production sources are unchanged.
// Each stage TU selects compile-time policies; there are no runtime stage branches.
// BF16 D128 causal Hopper forward, specialized to the fixed MHA/GQA/MQA suite.
// Optimizations trace to official FlashAttention commit 94e22c906678e5483fa0e9e24d8e787bc2c0ed4c.
// FA3 paper: https://arxiv.org/abs/2407.08608, Sections 3.1 and 3.2.
// FA4 paper: https://arxiv.org/abs/2603.05451; Blackwell TMEM/tcgen05 is not ported.
// Official sources: https://github.com/Dao-AILab/flash-attention/tree/94e22c906678e5483fa0e9e24d8e787bc2c0ed4c
// FA4 Hopper source: flash_attn/cute/flash_fwd_sm90.py; no Blackwell-only TMEM.
#pragma once
#include "flash_attention.h"
#include "cuda_common.h"

#include <cuda.h>
#include <cudaTypedefs.h>

#include <cfloat>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace gpu_kernels::fa_ablation::FA_STAGE_NAMESPACE {
namespace {

constexpr int STAGE = FA_ABLATION_STAGE;

using bf16 = __nv_bfloat16;

// FA3 hopper/tile_size.h::tile_size_fwd_sm90: causal BF16 D128 uses 128x128.
// Two consumer warp groups reuse each K/V tile, owning 64 query rows each.
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int D = 128;
constexpr int HALF_D = 64;
constexpr int WGMMA_K = 16;
// A3: two-stage TMA buffers, following official Hopper pipeline_k/pipeline_v.
constexpr int STAGES = STAGE >= 3 ? 2 : 1;
constexpr int NUM_THREADS = 384;
constexpr int GROUPS = 8;
constexpr int TMA_TILE_BYTES = BM * HALF_D * sizeof(bf16);
constexpr int KV_TRANSACTION_BYTES = 2 * TMA_TILE_BYTES;
constexpr int SMEM_ALIGNMENT = 1024;
constexpr float LOG2E = 1.4426950408889634F;

// FA3 mainloop_fwd_sm90_tma_gmma_ws.hpp::TensorStorageNoTranspose:
// MmaPV_is_RS keeps P in registers; BF16 V uses an MN-major descriptor,
// without a physical transpose. TMA and WGMMA share the SW128 layout.
struct SharedStorage {
    alignas(SMEM_ALIGNMENT) bf16 Q[2][BM * HALF_D];
    alignas(SMEM_ALIGNMENT) bf16 K[STAGES][2][BN * HALF_D];
    // FA3 flash_fwd_kernel_sm90.h::SharedStorage aliases epilogue O only
    // with mainloop V. Their lifetimes are disjoint after all consumers finish
    // PV, so no separate 32 KiB output allocation is required.
    union {
        alignas(SMEM_ALIGNMENT) bf16 V[STAGES][2][BN * HALF_D];
        alignas(SMEM_ALIGNMENT) bf16 O[2][BM * HALF_D];
    };
};

constexpr size_t SMEM_BYTES = sizeof(SharedStorage) + SMEM_ALIGNMENT - 1;

CUtensorMap make_attention_map(const bf16* pointer, int batch_size, int sequence_length, int heads) {
    CUtensorMap map;
    void* address = const_cast<bf16*>(pointer);
    // FA3 mainloop::to_underlying_arguments / make_tma_copy transfers a
    // complete Q/K/V tile per instruction. TMA groups D into 64-value bands
    // so a single transfer produces the canonical SW128 descriptor layout.
    const uint64_t global_shape[5] = {HALF_D, static_cast<uint64_t>(sequence_length), static_cast<uint64_t>(heads), 2, static_cast<uint64_t>(batch_size)};
    const uint64_t global_stride[4] = {static_cast<uint64_t>(heads) * D * sizeof(bf16), D * sizeof(bf16), HALF_D * sizeof(bf16), static_cast<uint64_t>(sequence_length) * heads * D * sizeof(bf16)};
    const uint32_t box_shape[5] = {HALF_D, BM, 1, 2, 1};
    const uint32_t element_stride[5] = {1, 1, 1, 1, 1};
    const CUresult result = cuTensorMapEncodeTiled(
        &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, address, global_shape, global_stride, box_shape, element_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
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
    // FA3 consumer_wait uses PipelineTmaAsync::consumer_try_wait before its
    // blocking wait. CUTLASS arch/barrier.h::ClusterBarrier::wait supplies
    // 0x989680 suspend ticks instead of repeatedly retrying short waits.
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

// FA3 mainloop::load: one elected producer issues one full-tile TMA copy;
// the transaction barrier makes completion visible to both consumer groups.
template <bool Query = false>
__device__ __forceinline__ void tma_load(bf16* destination, const CUtensorMap* map, uint64_t* barrier, int token, int head, int batch) {
    // FA3 mainloop::load uses EVICT_FIRST for Q and EVICT_LAST for reused K/V.
    // These exact policies and the PTX cache-hint operand come from CUTLASS
    // cute/arch/copy_sm90_desc.hpp::CacheHintSm90 / SM90_TMA_LOAD_5D.
    constexpr uint64_t cache_hint = Query ? 0x12F0000000000000ULL : 0x14F0000000000000ULL;
    // A8 groups the official cache policies with scheduling and the output epilogue.
    if constexpr (STAGE >= 8) {
        asm volatile("cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {0, %3, %4, 0, %5}], [%2], %6;"
            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(destination))), "l"(map), "r"(barrier_address(barrier)), "r"(token), "r"(head), "r"(batch), "l"(cache_hint) : "memory");
    } else {
        asm volatile("cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {0, %3, %4, 0, %5}], [%2];"
            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(destination))), "l"(map), "r"(barrier_address(barrier)), "r"(token), "r"(head), "r"(batch) : "memory");
    }
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

// Official CUTLASS cute/atom/mma_traits_sm90_gmma.hpp::make_gmma_desc:
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

// CUTLASS mma_traits_sm90_gmma.hpp::DescriptorIterator::operator+ updates
// only the low 32-bit descriptor word; the layout/swizzle word is invariant.
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

// FA3 utils.h::gemm / CUTLASS mma_sm90_gmma.hpp::warpgroup_fence_operand:
// compiler fences make register lifetimes explicit at each asynchronous MMA
// issue site. These empty instructions match the official gemm wrapper.
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

// FA3 QK uses the official SS BF16 atom with FP32 accumulators.
// Source: cute/arch/mma_sm90_gmma.hpp::MMA_64x128x16_F32BF16BF16_SS.
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

// FA3 PV uses the RS atom: P is four BF16x2 registers per K=16 step;
// MN-major V is consumed directly from TMA-swizzled shared memory.
// Source: cute/arch/mma_sm90_gmma.hpp::MMA_64x128x16_F32BF16BF16_RS.
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
    // FA3's partitioned GMMA tensors use CUTLASS DescriptorIterator. Build
    // each band once, then advance by the same encoded K-step offsets rather
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

// FA3 utils.h::convert_layout_acc_Aregs and mainloop::convert_type_out:
// contiguous groups of eight FP32 scores map to four packed A registers;
// round-to-nearest BF16 conversion matches the official probability operand.
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

// FA3 softmax.h::Softmax::max_get_scale / finalize: exp2 with FMA, and
// lane-local partial sums reduced only once at the end, not every K/V tile.
// FA3 hopper/setup.py compiles softmax with --use_fast_math. fast_exp2 makes
// its hardware approximate exponent explicit without changing other kernels.
// FA3 mainloop::mma traverses backward and masks only the diagonal tile.
__device__ __forceinline__ float fast_exp2(float value) {
    // A7: official softmax.py uses fast base-2 exponentiation. Earlier stages
    // use ordinary expf in the equivalent natural-log domain.
    if constexpr (STAGE >= 7) {
        float result;
        asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
        return result;
    } else {
        return expf(value * 0.6931471805599453F);
    }
}

template <bool First, bool Masked>
__device__ __forceinline__ void online_softmax(float (&scores)[GROUPS][8], float (&row_max)[2], float (&row_sum)[2], float (&previous_scale)[2],
                                               int query_start, int key_start, int consumer_thread, float scale_log2) {
    const int lane = consumer_thread & 31;
    const int warp = consumer_thread >> 5;
    const int top_row = warp * 16 + lane / 4;
    const int bottom_row = top_row + 8;
    // Official CuTe utils.py::fmax_reduce (arch < 100) uses four independent
    // maximum chains, then a tree merge. Keep that Hopper reduction order
    // for each row fragment instead of one serial maximum across all groups.
    // Official CuTe utils.py::fmax_reduce seeds its four chains from the
    // first four scores, avoiding redundant max instructions against -inf.
    float partial_max[2][4];

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
        // Official CuTe utils.py::fmax_reduce merges init_val into the local
        // tree result before softmax.py performs its four-lane warp reduction.
        if constexpr (!First) current_max[r] = fmaxf(current_max[r], row_max[r]);
    }
    current_max[0] = fmaxf(current_max[0], __shfl_xor_sync(0xffffffff, current_max[0], 1));
    current_max[0] = fmaxf(current_max[0], __shfl_xor_sync(0xffffffff, current_max[0], 2));
    current_max[1] = fmaxf(current_max[1], __shfl_xor_sync(0xffffffff, current_max[1], 1));
    current_max[1] = fmaxf(current_max[1], __shfl_xor_sync(0xffffffff, current_max[1], 2));

    float next_max[2] = {current_max[0], current_max[1]};
    // FA4 paper Section 3.1.4 / softmax.py::SoftmaxSm100.update_row_max:
    // keep the previous exponent base until the increase exceeds 8 in log2
    // units. P, row_sum and O retain that same base; final normalization and
    // LSE use the retained base too. This arithmetic does not require TMEM.
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        previous_scale[r] = 1.0F;
        if constexpr (!First) {
            const float change = (row_max[r] - next_max[r]) * scale_log2;
            // A7: FA4 Section 3.1.4 / SoftmaxSm100.update_row_max; this is a
            // Hopper arithmetic adaptation, not the active official SM90 policy.
            if constexpr (STAGE >= 7) {
                if (change >= -8.0F) next_max[r] = row_max[r];
                else previous_scale[r] = fast_exp2(change);
            } else {
                previous_scale[r] = fast_exp2(change);
            }
        }
    }
    const float maximum_scaled[2] = {next_max[0] * scale_log2, next_max[1] * scale_log2};

    // FA3 hopper/softmax.h::Softmax::online_softmax first calls
    // scale_apply_exp2, then reduce_sum<warp_reduce=false>. Keep SFU work
    // separate from the dependent row-sum chains, as in that implementation.
#pragma unroll
    for (int row_id = 0; row_id < 2; ++row_id) {
#pragma unroll
        for (int group = 0; group < GROUPS; ++group) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                // FA3 scale_apply_exp2 traverses one row at a time (mi, ni).
                const int item = (i & 1) + (i / 2) * 4 + row_id * 2;
                const float exponent = __fmaf_rn(scores[group][item], scale_log2, -maximum_scaled[row_id]);
                scores[group][item] = fast_exp2(exponent);
            }
        }
    }
    // Official CuTe utils.py::fadd_reduce documents a commented scalar
    // four-chain/tree alternative beside its active Hopper x.reduce path.
    // This implements that documented alternative, not the active lowering,
    // reduction structure, without the SM100-only packed-f32x2 instructions.
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
    // Before A7, reduce every tile. Replicate one quarter of the row sum in
    // each lane so finalize_rows retains the same four-lane final reduction.
    if constexpr (STAGE < 7) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            local_sum[r] += __shfl_xor_sync(0xffffffff, local_sum[r], 1);
            local_sum[r] += __shfl_xor_sync(0xffffffff, local_sum[r], 2);
            local_sum[r] *= 0.25F;
        }
    }
    row_sum[0] = (First ? 0.0F : row_sum[0] * previous_scale[0]) + local_sum[0];
    row_sum[1] = (First ? 0.0F : row_sum[1] * previous_scale[1]) + local_sum[1];
    row_max[0] = next_max[0];
    row_max[1] = next_max[1];
}

__device__ __forceinline__ void scale_output(float (&output)[GROUPS][8], const float (&factor)[2]) {
    // FA4 Section 3.1.4 / flash_fwd_sm100.py::correction_loop uses a
    // warp-uniform rescale decision. Skip all O-register multiplications when
    // every row in this warp retained its exponent base (both factors are 1).
    if constexpr (STAGE >= 7) {
        if (!__any_sync(0xffffffff, factor[0] != 1.0F || factor[1] != 1.0F)) return;
    }
#pragma unroll
    for (int group = 0; group < GROUPS; ++group) {
#pragma unroll
        for (int item = 0; item < 8; ++item) output[group][item] *= factor[(item & 2) >> 1];
    }
}

// FA3 mainloop::mma calls softmax.finalize while the final PV is outstanding.
// Row sums, reciprocals and LSE do not depend on the O accumulator registers.
__device__ __forceinline__ void finalize_rows(float (&inverse)[2], float (&lse_values)[2], const float (&row_max)[2], float (&row_sum)[2], float scale) {
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        row_sum[r] += __shfl_xor_sync(0xffffffff, row_sum[r], 1);
        row_sum[r] += __shfl_xor_sync(0xffffffff, row_sum[r], 2);
    }
    // Official CuTe softmax.py::Softmax.finalize uses rcp_approx; FA3's
    // equivalent division is compiled with --use_fast_math. These fixed
    // causal rows have positive sums, so normalize with that same primitive.
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        if constexpr (STAGE >= 7) {
            asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(inverse[r]) : "f"(row_sum[r]));
        } else {
            inverse[r] = 1.0F / row_sum[r];
        }
        lse_values[r] = row_max[r] * scale + __logf(row_sum[r]);
    }
}

// FA3 epilogue_fwd.hpp::store: retile register accumulators into SW128 BF16
// shared O, then use one TMA store instead of many small global stores.
__device__ __forceinline__ void store_output(SharedStorage& shared, float* lse, float (&accumulator)[GROUPS][8], const float (&inverse)[2], const float (&lse_values)[2], int batch, int head, int start, int sequence, int heads, int tid) {
    const int lane = tid & 31;
    const int row = (tid >> 5) * 16 + lane / 4;
    // FA3 epilogue_fwd.hpp::SmemCopyAtomO selects SM90_U32x4_STSM_N.
    // CUTLASS copy_traits_sm90.hpp maps the four packed accumulator pairs
    // to four 8x8 matrices. stmatrix performs the register-to-SW128 retile
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
    bf16* output, float* logsumexp, const __grid_constant__ CUtensorMap Q_map, const __grid_constant__ CUtensorMap K_map,
    const __grid_constant__ CUtensorMap V_map, const __grid_constant__ CUtensorMap O_map, int sequence_length, int query_heads, int key_value_heads,
    int section_heads, float scale) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    extern __shared__ __align__(16) unsigned char storage[];
    const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(storage));
    const uint32_t aligned = (address + SMEM_ALIGNMENT - 1) & ~(SMEM_ALIGNMENT - 1);
    auto& shared = *reinterpret_cast<SharedStorage*>(__cvta_shared_to_generic(aligned));
    __shared__ __align__(8) uint64_t query_full, k_full[STAGES], v_full[STAGES], k_empty[STAGES], v_empty[STAGES];
    if (threadIdx.x == 0) {
        // FA3 flash_fwd_kernel_sm90.h prefetches mainloop and epilogue
        // descriptors before pipeline initialization. The exact PTX primitive
        // is CUTLASS cute/arch/copy_sm90_desc.hpp::prefetch_tma_descriptor.
        if constexpr (STAGE >= 8) {
            asm volatile("prefetch.tensormap [%0];" :: "l"(&Q_map) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"(&K_map) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"(&V_map) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"(&O_map) : "memory");
        }
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
    // Static scheduling, not a dynamic persistent work queue: use
    // FA3 tile_scheduler.hpp::DynamicPersistentTileScheduler's L2 policy and
    // FA4 tile_scheduler.py::SingleTileLPTScheduler.get_current_work / paper
    // Section 3.3: longest causal tiles first within L2-sized head
    // sections. GQA's adjacent query heads reuse one KV head; larger MHA
    // workloads need multiple sections instead of interleaving every head.
    // Fixed B={1,4}, Hq=8 shapes divide these power-of-two sections exactly.
    const int query_blocks = (sequence_length + BM - 1) / BM;
    const int section_tiles = section_heads * query_blocks;
    const int section = blockIdx.x / section_tiles;
    const int section_tile = blockIdx.x % section_tiles;
    // A8: official SingleTileLPTScheduler + L2-sized head sections. Earlier
    // stages use ordinary ascending query tiles within each batch/head.
    const int query_block = STAGE >= 8 ? query_blocks - 1 - section_tile / section_heads : blockIdx.x % query_blocks;
    const int head_batch = STAGE >= 8 ? section * section_heads + section_tile % section_heads : blockIdx.x / query_blocks;
    const int start = query_block * BM;
    const int head = head_batch % query_heads;
    const int batch = head_batch / query_heads;
    const int kv_head = head / (query_heads / key_value_heads);

    // FA3 flash_fwd_kernel_sm90.h::operator(): warp specialization and
    // dynamic register reallocation: producer 24 registers, consumers 240.
    if (group == 0) {
        if constexpr (STAGE >= 4) asm volatile("setmaxnreg.dec.sync.aligned.u32 24;");
        if (tid == 0) {
            barrier_expect_bytes(&query_full, 2 * TMA_TILE_BYTES);
            tma_load<true>(shared.Q[0], &Q_map, &query_full, start, head, batch);
            int stage = 0, phase = 0;
            if constexpr (STAGE >= 6) {
                barrier_wait(&k_empty[stage], phase);
                barrier_expect_bytes(&k_full[stage], KV_TRANSACTION_BYTES);
                tma_load(shared.K[stage][0], &K_map, &k_full[stage], start, kv_head, batch);
                // FA3 mainloop::load / IntraWGOverlap primes K(current), then
                // loads K(next) before V(current). QK gets one tile of lookahead;
                // independent empty barriers still protect both circular buffers.
                // FA3 mainloop::load uses unroll(2) for non-transposed TMA K/V.
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
            } else {
                // A2 uses one slot and does not release it until PV finishes.
                // A3 adds a second slot, allowing the producer to load ahead.
                // These are serial/pipelined variants of official load_K/load_V.
#pragma unroll 1
                for (int iteration = 0; iteration <= query_block; ++iteration) {
                    const int token = (query_block - iteration) * BN;
                    barrier_wait(&k_empty[stage], phase);
                    barrier_wait(&v_empty[stage], phase);
                    barrier_expect_bytes(&k_full[stage], KV_TRANSACTION_BYTES);
                    barrier_expect_bytes(&v_full[stage], KV_TRANSACTION_BYTES);
                    tma_load(shared.K[stage][0], &K_map, &k_full[stage], token, kv_head, batch);
                    tma_load(shared.V[stage][0], &V_map, &v_full[stage], token, kv_head, batch);
                    advance_stage(stage, phase);
                }
            }
        }
        return;
    }
    if constexpr (STAGE >= 4) asm volatile("setmaxnreg.inc.sync.aligned.u32 240;");
    // FA3 Section 3.1 ping-pong scheduling; mainloop::mma_init and
    // warp_scheduler_barrier_{sync,arrive}. The first consumer self-signals,
    // then the groups alternate Tensor Core issue while their peer does softmax.
    if constexpr (STAGE >= 5) {
        if (group == 1) asm volatile("bar.arrive 1, 256;" ::: "memory");
    }
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
    if constexpr (STAGE >= 6) {
        barrier_wait(&k_full[stage], phase);
        // FA3 mainloop::mma leaves the QK accumulator fragment uninitialized:
        // utils.h::gemm<zero_init=true> overwrites it with GMMA ScaleOut::Zero.
        // issue_qk likewise uses ScaleD=0 first, so no CUDA-core zeroing is needed.
        float scores[GROUPS][8];
        issue_qk(scores, shared, stage, consumer);
        warpgroup_wait<0>();
        if (tid == 0) barrier_arrive(&k_empty[stage]);
        online_softmax<true, true>(scores, row_max, row_sum, factor, query_start, start, tid, scale_log2);
        pack_probabilities(P, scores);
        int value_stage = stage, value_phase = phase;
        advance_stage(stage, phase);

        // FA3 Section 3.2 / mainloop::mma (IntraWGOverlap): QK(next) precedes
        // PV(current). wait_group<1> exposes scores while PV runs alongside
        // CUDA-core softmax; wait_group<0> precedes rescaling the O registers.
        // FA3 mainloop::mma keeps its unmasked fwd_step loop rolled (unroll 1),
        // avoiding multiple large softmax/WGMMA bodies in the instruction stream.
#pragma unroll 1
        for (int iteration = 1; iteration <= query_block; ++iteration) {
            // FA3 mainloop::mma / IntraWGOverlap::fwd_step waits for K only in
            // consumer WG0. Its scheduler handoff makes that acquire visible to
            // WG1; both groups still release K after their own WGMMA completes.
            if (consumer == 0) barrier_wait(&k_full[stage], phase);
            // As in FA3 fwd_step, the first QK WGMMA overwrites every score register.
            float next_scores[GROUPS][8];
            asm volatile("bar.sync %0, 256;" :: "r"(group) : "memory");
            issue_qk(next_scores, shared, stage, consumer);
            // FA3 fwd_step uses the same WG0-only wait for V. Signal the peer only
            // after issuing PV, preserving the official QK -> PV -> handoff order.
            if (consumer == 0) barrier_wait(&v_full[value_stage], value_phase);
            issue_pv(O, P, shared, value_stage);
            asm volatile("bar.arrive %0, 256;" :: "r"(3 - group) : "memory");
            warpgroup_wait<1>();
            if (tid == 0) barrier_arrive(&k_empty[stage]);
            online_softmax<false, false>(next_scores, row_max, row_sum, factor, query_start, (query_block - iteration) * BN, tid, scale_log2);
            warpgroup_wait<0>();
            if (tid == 0) barrier_arrive(&v_empty[value_stage]);
            // FA3 mainloop::mma converts P before rescaling O when
            // RescaleOBeforeGemm is false (the official D128 configuration).
            pack_probabilities(P, next_scores);
            scale_output(O, factor);
            value_stage = stage;
            value_phase = phase;
            advance_stage(stage, phase);
        }
        barrier_wait(&v_full[value_stage], value_phase);
        issue_pv(O, P, shared, value_stage);

    } else {
        // A2-A5 deliberately drain each QK/PV group. A5 only introduces the
        // inter-WG handoff; A6 separately enables official IntraWGOverlap.
#pragma unroll 1
        for (int iteration = 0; iteration <= query_block; ++iteration) {
            barrier_wait(&k_full[stage], phase);
            barrier_wait(&v_full[stage], phase);
            if constexpr (STAGE >= 5) asm volatile("bar.sync %0, 256;" :: "r"(group) : "memory");
            float scores[GROUPS][8];
            issue_qk(scores, shared, stage, consumer);
            warpgroup_wait<0>();
            const int key_start = (query_block - iteration) * BN;
            if (iteration == 0) {
                online_softmax<true, true>(scores, row_max, row_sum, factor, query_start, key_start, tid, scale_log2);
            } else {
                online_softmax<false, false>(scores, row_max, row_sum, factor, query_start, key_start, tid, scale_log2);
                scale_output(O, factor);
            }
            pack_probabilities(P, scores);
            issue_pv(O, P, shared, stage);
            if constexpr (STAGE >= 5) asm volatile("bar.arrive %0, 256;" :: "r"(3 - group) : "memory");
            warpgroup_wait<0>();
            if (tid == 0) {
                barrier_arrive(&k_empty[stage]);
                barrier_arrive(&v_empty[stage]);
            }
            advance_stage(stage, phase);
        }
    }
    float inverse[2], lse_values[2];
    // FA3 mainloop_fwd_sm90_tma_gmma_ws.hpp::mma: finalize precedes
    // warpgroup_wait<0>, overlapping the final softmax reduction with PV.
    finalize_rows(inverse, lse_values, row_max, row_sum, scale);
    warpgroup_wait<0>();
    // All stages have completed their final PV before output storage reuse.
    if constexpr (STAGE >= 8) {
        // FA3 epilogue_fwd.hpp::store first synchronizes all epilogue threads:
        // every warp group must stop reading V before shared O overwrites it.
        asm volatile("bar.sync 3, 256;" ::: "memory");
        store_output(shared, logsumexp, O, inverse, lse_values, batch, head, query_start, sequence_length, query_heads, tid);
        // FA3 epilogue_fwd.hpp::store uses fence_view_async_shared plus an
        // epilogue named barrier before the elected TMA store, then waits for
        // shared-source reads to finish before a CTA can release its storage.
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        asm volatile("bar.sync 3, 256;" ::: "memory");
        if (group == 1 && tid == 0) {
            asm volatile("cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group [%0, {0, %2, %3, 0, %4}], [%1];"
                :: "l"(&O_map), "r"(static_cast<uint32_t>(__cvta_generic_to_shared(shared.O[0]))), "r"(start), "r"(head), "r"(batch) : "memory");
            asm volatile("cp.async.bulk.commit_group;" ::: "memory");
            asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
        }
    } else {
        // A2-A7 store register-owned output pairs directly. The mapping is
        // the same GMMA accumulator layout used by official epilogue_fwd.hpp.
        const int lane = tid & 31;
        const int row = (tid >> 5) * 16 + lane / 4;
#pragma unroll
        for (int g = 0; g < GROUPS; ++g) {
#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                const int r = pair & 1;
                const int token = query_start + row + r * 8;
                const int column = g * 16 + (pair / 2) * 8 + 2 * (lane & 3);
                const __nv_bfloat162 packed = __floats2bfloat162_rn(O[g][pair * 2] * inverse[r], O[g][pair * 2 + 1] * inverse[r]);
                const int index = ((batch * sequence_length + token) * query_heads + head) * D + column;
                *reinterpret_cast<__nv_bfloat162*>(output + index) = packed;
            }
        }
        if ((lane & 3) == 0) {
#pragma unroll
            for (int r = 0; r < 2; ++r) logsumexp[(batch * query_heads + head) * sequence_length + query_start + row + r * 8] = lse_values[r];
        }
    }
#endif
}

}  // namespace

void forward(bf16* output, float* logsumexp, const bf16* query, const bf16* key, const bf16* value, int sequence_length, cudaStream_t stream) {
    constexpr int batch_size = 4, query_heads = 8, key_value_heads = 8;
    constexpr float scale = 0.08838834764831845F;
    const CUtensorMap Q_map = make_attention_map(query, batch_size, sequence_length, query_heads);
    const CUtensorMap K_map = make_attention_map(key, batch_size, sequence_length, key_value_heads);
    const CUtensorMap V_map = make_attention_map(value, batch_size, sequence_length, key_value_heads);
    const CUtensorMap O_map = make_attention_map(output, batch_size, sequence_length, query_heads);
    // FA3 tile_scheduler.hpp::DynamicPersistentTileScheduler::to_underlying_arguments:
    // budget 32 MiB for K/V, round the KV-head capacity down to a power of two,
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
        output, logsumexp, Q_map, K_map, V_map, O_map, sequence_length, query_heads, key_value_heads, section_heads, scale);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace gpu_kernels

