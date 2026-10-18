// Implements the SM89 D128 causal FlashAttention forward path with BF16 Tensor Cores and FP32 online softmax.
// QK^T, causal softmax, and P@V stay fused so the quadratic score matrix is never written to global memory.

#include "common.cuh"
#include "cuda_common.h"

#include <cfloat>
#include <stdexcept>

namespace gpu_kernels {
namespace flash_attention_sm89 {

constexpr int WARP_SIZE = 32;
constexpr int HEAD_SIZE = 128;

inline void validate_head_size(int head_size) {
    if (head_size != HEAD_SIZE) {
        throw std::runtime_error("flash attention requires head_size = 128");
    }
}

namespace tensor_core {

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int D = HEAD_SIZE;
constexpr int THREADS = 128;
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;
// QK^T produces 64 score columns, while PV produces 128 output columns.
constexpr int SCORE_N_TILES = BN / MMA_N;
constexpr int OUTPUT_N_TILES = D / MMA_N;
constexpr int HEAD_K_TILES = D / MMA_K;
constexpr int TOKEN_K_TILES = BN / MMA_K;
constexpr int VECTOR_ELEMENTS = 8;

template <int STRIDE = D>
__device__ __forceinline__ int swizzle(int offset) {
    return offset ^ (((offset / STRIDE) & 7) << 3);
}

__device__ __forceinline__ unsigned int shared_address(const void* pointer) {
    return static_cast<unsigned int>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void cp_async_bf16x8(__nv_bfloat16* destination, const __nv_bfloat16* source) {
    const unsigned int destination_address = shared_address(destination);
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(destination_address), "l"(source));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group 0;\n" ::);
}

__device__ __forceinline__ void load_matrix_x4(unsigned int (&fragment)[4], unsigned int address) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];\n"
        : "=r"(fragment[0]), "=r"(fragment[1]), "=r"(fragment[2]), "=r"(fragment[3])
        : "r"(address));
}

__device__ __forceinline__ void load_matrix_x4_transpose(unsigned int (&fragment)[4], unsigned int address) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
        "{%0, %1, %2, %3}, [%4];\n"
        : "=r"(fragment[0]), "=r"(fragment[1]), "=r"(fragment[2]), "=r"(fragment[3])
        : "r"(address));
}

__device__ __forceinline__ void mma(float (&accumulator)[4], const unsigned int (&left)[4], const unsigned int* right) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(accumulator[0]), "+f"(accumulator[1]), "+f"(accumulator[2]), "+f"(accumulator[3])
        : "r"(left[0]), "r"(left[1]), "r"(left[2]), "r"(left[3]), "r"(right[0]), "r"(right[1]));
}

__device__ __forceinline__ unsigned int pack_bf16x2(float2 values) {
    union Packed {
        __nv_bfloat162 bf16;
        unsigned int bits;
    } packed;
    packed.bf16 = __float22bfloat162_rn(values);
    return packed.bits;
}

__device__ __forceinline__ int tensor_index(int batch, int token, int head, int column, int sequence_length, int heads) {
    return ((batch * sequence_length + token) * heads + head) * D + column;
}

__device__ __forceinline__ void copy_bf16_tile(__nv_bfloat16* shared, const __nv_bfloat16* global, int batch, int first_token, int head, int sequence_length,
                                               int heads) {
    constexpr int VECTORS = BM * D / VECTOR_ELEMENTS;
    constexpr int VECTORS_PER_THREAD = VECTORS / THREADS;
    static_assert(VECTORS % THREADS == 0);
#pragma unroll
    for (int iteration = 0; iteration < VECTORS_PER_THREAD; ++iteration) {
        const int vector = threadIdx.x + iteration * THREADS;
        const int row = vector / (D / VECTOR_ELEMENTS);
        const int column = vector % (D / VECTOR_ELEMENTS) * VECTOR_ELEMENTS;
        const int logical = row * D + column;
        cp_async_bf16x8(shared + swizzle(logical), global + tensor_index(batch, first_token + row, head, column, sequence_length, heads));
    }
    cp_async_commit();
}

template <int COLUMNS = D>
__device__ __forceinline__ void load_left_fragments(unsigned int (&fragments)[COLUMNS / MMA_K][4], const __nv_bfloat16* shared, int warp_row, int lane) {
#pragma unroll
    for (int tile = 0; tile < COLUMNS / MMA_K; ++tile) {
        const int row = warp_row + lane % MMA_M;
        const int column = tile * MMA_K + lane / MMA_M * 8;
        load_matrix_x4(fragments[tile], shared_address(shared + swizzle<COLUMNS>(row * COLUMNS + column)));
    }
}

// One x4 instruction loads the B fragments for two adjacent 8-column MMA
// tiles. This halves ldmatrix issue count compared with two independent x2
// loads while preserving the same register layout consumed by mma.sync.
__device__ __forceinline__ void load_right_transposed_pair(unsigned int (&fragments)[4], const __nv_bfloat16* shared, int tile_inner, int tile_column,
                                                           int lane) {
    const int row = tile_column * MMA_N + lane / MMA_M * MMA_N + lane % MMA_N;
    const int column = tile_inner * MMA_K + (lane % MMA_M) / 8 * 8;
    load_matrix_x4(fragments, shared_address(shared + swizzle(row * D + column)));
}

__device__ __forceinline__ void load_right_pair(unsigned int (&fragments)[4], const __nv_bfloat16* shared, int tile_inner, int tile_column, int lane) {
    const int row = tile_inner * MMA_K + lane % MMA_M;
    const int column = tile_column * MMA_N + lane / MMA_M * MMA_N;
    load_matrix_x4_transpose(fragments, shared_address(shared + swizzle(row * D + column)));
}

template <int N_TILES, int K_TILES>
__device__ __forceinline__ void matrix_product_transposed_right(float (&accumulators)[N_TILES][4], const unsigned int (&left)[K_TILES][4],
                                                                const __nv_bfloat16* right, int lane) {
    static_assert(N_TILES % 2 == 0);
#pragma unroll
    for (int tile_inner = 0; tile_inner < K_TILES; ++tile_inner) {
#pragma unroll
        for (int tile_column = 0; tile_column < N_TILES; tile_column += 2) {
            unsigned int right_fragments[4];
            load_right_transposed_pair(right_fragments, right, tile_inner, tile_column, lane);
            mma(accumulators[tile_column], left[tile_inner], right_fragments);
            mma(accumulators[tile_column + 1], left[tile_inner], right_fragments + 2);
        }
    }
}

template <int N_TILES, int K_TILES>
__device__ __forceinline__ void matrix_product_right(float (&accumulators)[N_TILES][4], const unsigned int (&left)[K_TILES][4], const __nv_bfloat16* right,
                                                     int lane) {
    static_assert(N_TILES % 2 == 0);
#pragma unroll
    for (int tile_inner = 0; tile_inner < K_TILES; ++tile_inner) {
#pragma unroll
        for (int tile_column = 0; tile_column < N_TILES; tile_column += 2) {
            unsigned int right_fragments[4];
            load_right_pair(right_fragments, right, tile_inner, tile_column, lane);
            mma(accumulators[tile_column], left[tile_inner], right_fragments);
            mma(accumulators[tile_column + 1], left[tile_inner], right_fragments + 2);
        }
    }
}

__device__ __forceinline__ void store_pair(__nv_bfloat16* destination, float2 value) {
    *reinterpret_cast<__nv_bfloat162*>(destination) = __float22bfloat162_rn(value);
}


constexpr float LOG2E = 1.4426950408889634F;

__device__ __forceinline__ void pack_score_matrix(unsigned int (&fragments)[TOKEN_K_TILES][4], const float (&scores)[SCORE_N_TILES][4]) {
#pragma unroll
    for (int tile = 0; tile < TOKEN_K_TILES; ++tile) {
        const int first = tile * 2;
        fragments[tile][0] = pack_bf16x2(make_float2(scores[first][0], scores[first][1]));
        fragments[tile][1] = pack_bf16x2(make_float2(scores[first][2], scores[first][3]));
        fragments[tile][2] = pack_bf16x2(make_float2(scores[first + 1][0], scores[first + 1][1]));
        fragments[tile][3] = pack_bf16x2(make_float2(scores[first + 1][2], scores[first + 1][3]));
    }
}

template <bool FIRST_TILE, bool MASKED_TILE>
__device__ __forceinline__ void online_softmax_tile(float (&scores)[SCORE_N_TILES][4], float (&output_accumulators)[OUTPUT_N_TILES][4],
                                                    float (&row_max)[2], float (&row_sum)[2], const __nv_bfloat16* shared_value, int warp_row,
                                                    int lane, float scale_log2) {
    const int local_top_row = warp_row + lane / 4;
    const int local_bottom_row = local_top_row + 8;
    float current_max[2] = {-FLT_MAX, -FLT_MAX};
#pragma unroll
    for (int tile = 0; tile < SCORE_N_TILES; ++tile) {
        if constexpr (MASKED_TILE) {
            const int key_column = tile * MMA_N + (lane % 4) * 2;
            scores[tile][0] = local_top_row >= key_column ? scores[tile][0] : -FLT_MAX;
            scores[tile][1] = local_top_row >= key_column + 1 ? scores[tile][1] : -FLT_MAX;
            scores[tile][2] = local_bottom_row >= key_column ? scores[tile][2] : -FLT_MAX;
            scores[tile][3] = local_bottom_row >= key_column + 1 ? scores[tile][3] : -FLT_MAX;
        }
        current_max[0] = fmaxf(current_max[0], fmaxf(scores[tile][0], scores[tile][1]));
        current_max[1] = fmaxf(current_max[1], fmaxf(scores[tile][2], scores[tile][3]));
    }
    current_max[0] = fmaxf(current_max[0], __shfl_xor_sync(0xffffffff, current_max[0], 1));
    current_max[0] = fmaxf(current_max[0], __shfl_xor_sync(0xffffffff, current_max[0], 2));
    current_max[1] = fmaxf(current_max[1], __shfl_xor_sync(0xffffffff, current_max[1], 1));
    current_max[1] = fmaxf(current_max[1], __shfl_xor_sync(0xffffffff, current_max[1], 2));

    float next_max[2];
    if constexpr (FIRST_TILE) {
        next_max[0] = current_max[0];
        next_max[1] = current_max[1];
    } else {
        next_max[0] = fmaxf(row_max[0], current_max[0]);
        next_max[1] = fmaxf(row_max[1], current_max[1]);
        const float previous_scale[2] = {exp2f((row_max[0] - next_max[0]) * scale_log2),
                                         exp2f((row_max[1] - next_max[1]) * scale_log2)};
#pragma unroll
        for (int tile = 0; tile < OUTPUT_N_TILES; ++tile) {
            output_accumulators[tile][0] *= previous_scale[0];
            output_accumulators[tile][1] *= previous_scale[0];
            output_accumulators[tile][2] *= previous_scale[1];
            output_accumulators[tile][3] *= previous_scale[1];
        }
        row_sum[0] *= previous_scale[0];
        row_sum[1] *= previous_scale[1];
    }

    float local_sum[2] = {};
    const float next_max_scaled[2] = {next_max[0] * scale_log2, next_max[1] * scale_log2};
#pragma unroll
    for (int tile = 0; tile < SCORE_N_TILES; ++tile) {
        scores[tile][0] = exp2f(__fmaf_rn(scores[tile][0], scale_log2, -next_max_scaled[0]));
        scores[tile][1] = exp2f(__fmaf_rn(scores[tile][1], scale_log2, -next_max_scaled[0]));
        scores[tile][2] = exp2f(__fmaf_rn(scores[tile][2], scale_log2, -next_max_scaled[1]));
        scores[tile][3] = exp2f(__fmaf_rn(scores[tile][3], scale_log2, -next_max_scaled[1]));
        local_sum[0] += scores[tile][0] + scores[tile][1];
        local_sum[1] += scores[tile][2] + scores[tile][3];
    }
    local_sum[0] += __shfl_xor_sync(0xffffffff, local_sum[0], 1);
    local_sum[0] += __shfl_xor_sync(0xffffffff, local_sum[0], 2);
    local_sum[1] += __shfl_xor_sync(0xffffffff, local_sum[1], 1);
    local_sum[1] += __shfl_xor_sync(0xffffffff, local_sum[1], 2);
    if constexpr (FIRST_TILE) {
        row_sum[0] = local_sum[0];
        row_sum[1] = local_sum[1];
    } else {
        row_sum[0] += local_sum[0];
        row_sum[1] += local_sum[1];
    }
    row_max[0] = next_max[0];
    row_max[1] = next_max[1];

    unsigned int probability_fragments[TOKEN_K_TILES][4];
    pack_score_matrix(probability_fragments, scores);
    matrix_product_right(output_accumulators, probability_fragments, shared_value, lane);
}

template <bool VECTOR_OUTPUT>
__global__ __launch_bounds__(THREADS, 2) void flash_attention_forward_tensor_core_kernel(__nv_bfloat16* __restrict__ output, float* __restrict__ logsumexp,
                                                                                         const __nv_bfloat16* __restrict__ query,
                                                                                         const __nv_bfloat16* __restrict__ key,
                                                                                         const __nv_bfloat16* __restrict__ value, int sequence_length,
                                                                                         int query_heads, int key_value_heads, float scale) {
    __shared__ __align__(16) __nv_bfloat16 shared_query[BM * D];
    __shared__ __align__(16) __nv_bfloat16 shared_key[BN * D];
    __shared__ __align__(16) __nv_bfloat16 shared_value[BN * D];

    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int batch_head = blockIdx.y;
    const int batch = batch_head / query_heads;
    const int query_head = batch_head % query_heads;
    const int key_value_head = query_head / (query_heads / key_value_heads);
    const int first_query = blockIdx.x * BM;
    const int warp_row = warp * MMA_M;

    // Load Q and the first K tile together. Later iterations prefetch the next K
    // while softmax and P@V consume only registers and the separate V buffer.
    copy_bf16_tile(shared_query, query, batch, first_query, query_head, sequence_length, query_heads);
    copy_bf16_tile(shared_key, key, batch, first_query, key_value_head, sequence_length, key_value_heads);
    cp_async_wait();
    __syncthreads();

    unsigned int query_fragments[HEAD_K_TILES][4];
    load_left_fragments(query_fragments, shared_query, warp_row, lane);
    float output_accumulators[OUTPUT_N_TILES][4] = {};
    float row_max[2] = {-FLT_MAX, -FLT_MAX};
    float row_sum[2] = {0.0F, 0.0F};
    const float scale_log2 = scale * LOG2E;

    // FA2 processes the diagonal tile first: it is the only tile requiring a
    // causal mask and it initializes online-softmax state without rescaling O.
    copy_bf16_tile(shared_value, value, batch, first_query, key_value_head, sequence_length, key_value_heads);
    float scores[SCORE_N_TILES][4] = {};
    matrix_product_transposed_right(scores, query_fragments, shared_key, lane);
    cp_async_wait();
    __syncthreads();
    if (first_query > 0) {
        copy_bf16_tile(shared_key, key, batch, first_query - BN, key_value_head, sequence_length, key_value_heads);
    }
    online_softmax_tile<true, true>(scores, output_accumulators, row_max, row_sum, shared_value, warp_row, lane, scale_log2);
    if (first_query > 0) {
        cp_async_wait();
        __syncthreads();
    }

    // Earlier key tiles are fully visible to this query block. Keeping them in
    // a separate loop removes causal predicates and first-tile branches.
    for (int first_key = first_query - BN; first_key >= 0; first_key -= BN) {
        copy_bf16_tile(shared_value, value, batch, first_key, key_value_head, sequence_length, key_value_heads);
        float full_scores[SCORE_N_TILES][4] = {};
        matrix_product_transposed_right(full_scores, query_fragments, shared_key, lane);
        cp_async_wait();
        __syncthreads();
        const int next_key = first_key - BN;
        if (next_key >= 0) {
            copy_bf16_tile(shared_key, key, batch, next_key, key_value_head, sequence_length, key_value_heads);
        }
        online_softmax_tile<false, false>(full_scores, output_accumulators, row_max, row_sum, shared_value, warp_row, lane, scale_log2);
        if (next_key >= 0) {
            cp_async_wait();
            __syncthreads();
        }
    }

    const int local_top_query = warp_row + lane / 4;
    const int local_bottom_query = local_top_query + 8;
    const int top_query = first_query + warp_row + lane / 4;
    const int bottom_query = top_query + 8;
    const float inverse_sum[2] = {1.0F / row_sum[0], 1.0F / row_sum[1]};
#pragma unroll
    for (int tile = 0; tile < OUTPUT_N_TILES; ++tile) {
        const int column = tile * MMA_N + (lane % 4) * 2;
        const float2 top = make_float2(output_accumulators[tile][0] * inverse_sum[0], output_accumulators[tile][1] * inverse_sum[0]);
        const float2 bottom = make_float2(output_accumulators[tile][2] * inverse_sum[1], output_accumulators[tile][3] * inverse_sum[1]);
        if constexpr (VECTOR_OUTPUT) {
            // Retile scattered MMA lanes through the unused Q buffer so each
            // thread can emit aligned 16-byte global stores, as FA2 does.
            store_pair(shared_query + swizzle(local_top_query * D + column), top);
            store_pair(shared_query + swizzle(local_bottom_query * D + column), bottom);
        } else {
            // Long sequences amortize the scattered BF16 stores and benefit
            // more from avoiding the epilogue barrier and extra live state.
            store_pair(output + tensor_index(batch, top_query, query_head, column, sequence_length, query_heads), top);
            store_pair(output + tensor_index(batch, bottom_query, query_head, column, sequence_length, query_heads), bottom);
        }
    }
    if (lane % 4 == 0) {
        logsumexp[batch_head * sequence_length + top_query] = row_max[0] * scale + __logf(row_sum[0]);
        logsumexp[batch_head * sequence_length + bottom_query] = row_max[1] * scale + __logf(row_sum[1]);
    }
    if constexpr (VECTOR_OUTPUT) {
        __syncthreads();
        constexpr int OUTPUT_VECTORS_PER_THREAD = BM * D / VECTOR_ELEMENTS / THREADS;
#pragma unroll
        for (int iteration = 0; iteration < OUTPUT_VECTORS_PER_THREAD; ++iteration) {
            const int vector = threadIdx.x + iteration * THREADS;
            const int row = vector / (D / VECTOR_ELEMENTS);
            const int column = vector % (D / VECTOR_ELEMENTS) * VECTOR_ELEMENTS;
            const uint4 packed = *reinterpret_cast<const uint4*>(shared_query + swizzle(row * D + column));
            *reinterpret_cast<uint4*>(output + tensor_index(batch, first_query + row, query_head, column, sequence_length, query_heads)) = packed;
        }
    }
}

void launch_forward(__nv_bfloat16* output, float* logsumexp, const __nv_bfloat16* query, const __nv_bfloat16* key, const __nv_bfloat16* value, int batch_size,
                    int sequence_length, int query_heads, int key_value_heads, float scale, cudaStream_t stream) {
    const dim3 grid(sequence_length / BM, batch_size * query_heads);
    if (sequence_length <= 512) {
        flash_attention_forward_tensor_core_kernel<true><<<grid, THREADS, 0, stream>>>(
            output, logsumexp, query, key, value, sequence_length, query_heads, key_value_heads, scale);
    } else {
        flash_attention_forward_tensor_core_kernel<false><<<grid, THREADS, 0, stream>>>(
            output, logsumexp, query, key, value, sequence_length, query_heads, key_value_heads, scale);
    }
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace tensor_core
}  // namespace flash_attention_sm89

void flash_attention_forward_sm89_cuda(__nv_bfloat16* output, float* logsumexp, const __nv_bfloat16* query, const __nv_bfloat16* key,
                                       const __nv_bfloat16* value, int batch_size, int sequence_length, int query_heads, int key_value_heads,
                                       int head_size, float scale, cudaStream_t stream) {
    flash_attention_sm89::validate_head_size(head_size);
    if (sequence_length <= 0 || sequence_length % flash_attention_sm89::tensor_core::BM != 0) {
        throw std::runtime_error("flash attention requires T to be a positive multiple of 64");
    }
    if (query_heads <= 0 || key_value_heads <= 0 || query_heads % key_value_heads != 0) {
        throw std::runtime_error("flash attention requires query_heads to be divisible by key_value_heads");
    }
    flash_attention_sm89::tensor_core::launch_forward(output, logsumexp, query, key, value, batch_size, sequence_length, query_heads, key_value_heads,
                                                      scale, stream);
}

}  // namespace gpu_kernels
