// ---------------------------------------------------------------------------
// paged_attention_v1.cu — Single-pass paged multi-head attention
//
// Vendored from vllm/csrc/attention/attention_kernels.cu (v1 path).
// Modifications: torch::Tensor replaced with TrueBuffer; pybind11 removed.
//
// Algorithm:
//   For each (sequence, query head), iterate over all filled KV blocks and
//   accumulate attention scores in a single pass using online softmax.
//   Efficient for short-to-medium context lengths (< ~8k tokens).
//
// Template parameters:
//   scalar_t  : __half or float
//   HEAD_SIZE : 64, 80, 96, 112, 128, or 256
//   BLOCK_SIZE: tokens per KV block (must match allocator, typically 16)
//   NUM_THREADS: threads per block (128 or 256)
// ---------------------------------------------------------------------------

#include "include/vllm_types.h"
#include "include/attention_utils.cuh"
#include "include/reduce_kernel_utils.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

template <typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS>
__global__ void paged_attention_v1_kernel(
    scalar_t* __restrict__ out,              // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ q,          // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ k_cache,    // [num_blocks, num_kv_heads, head_size, block_size]
    const scalar_t* __restrict__ v_cache,    // [num_blocks, num_kv_heads, head_size, block_size]
    const int32_t* __restrict__ block_tables, // [num_seqs, max_blocks_per_seq]
    const int32_t* __restrict__ seq_lens,     // [num_seqs]
    int num_kv_heads,
    int max_blocks_per_seq,
    float scale)
{
    // Grid: (num_heads, num_seqs)
    int seq_idx  = blockIdx.y;
    int head_idx = blockIdx.x;
    int kv_head  = head_idx % num_kv_heads;  // GQA / MQA support

    int seq_len  = seq_lens[seq_idx];
    if (seq_len == 0) return;

    // Shared memory layout:
    //   float q_smem[HEAD_SIZE]            — query vector (float for precision)
    //   float logits[BLOCK_SIZE * ...]     — attention scores (one block at a time)
    //   float reduce_smem[NUM_THREADS/32]  — warp partial reductions
    __shared__ float q_smem[HEAD_SIZE];
    __shared__ float reduce_smem[NUM_THREADS / 32];

    // Load query vector into shared memory (all threads cooperate)
    const scalar_t* q_ptr = q + seq_idx * gridDim.x * HEAD_SIZE
                               + head_idx * HEAD_SIZE;
    for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS)
        q_smem[i] = to_float(q_ptr[i]);
    __syncthreads();

    int num_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Accumulators for online softmax
    SoftmaxState softmax_state;
    float out_acc[HEAD_SIZE] = {};  // output accumulator (zero-initialised)

    // -----------------------------------------------------------------------
    // Main loop: iterate over KV blocks
    // -----------------------------------------------------------------------
    for (int b = 0; b < num_blocks; ++b) {
        int32_t phys_block = block_tables[seq_idx * max_blocks_per_seq + b];
        if (phys_block < 0) continue;

        int tokens_in_block = min(BLOCK_SIZE, seq_len - b * BLOCK_SIZE);

        // Base pointers for this physical block
        const scalar_t* k_block = k_cache
            + static_cast<int64_t>(phys_block) * num_kv_heads * HEAD_SIZE * BLOCK_SIZE
            + kv_head * HEAD_SIZE * BLOCK_SIZE;
        const scalar_t* v_block = v_cache
            + static_cast<int64_t>(phys_block) * num_kv_heads * HEAD_SIZE * BLOCK_SIZE
            + kv_head * HEAD_SIZE * BLOCK_SIZE;

        // Compute QK dot products for each token in this block
        for (int t = threadIdx.x; t < tokens_in_block; t += NUM_THREADS) {
            // k_block layout: [head_size, block_size] — column-major over block_size
            float qk = 0.f;
#pragma unroll
            for (int d = 0; d < HEAD_SIZE; ++d)
                qk += q_smem[d] * to_float(k_block[d * BLOCK_SIZE + t]);
            qk *= scale;
            softmax_state.update(qk);
        }

        // Synchronise softmax state across the block via warp + block reductions
        // (simplified: reduce max and exp_sum across all threads)
        // Each thread now holds a partial softmax state; we need the global state.
        // Use shared-memory broadcast of the warp-reduced values.
        float local_max = softmax_state.max_val;
        float local_sum = softmax_state.exp_sum;
        local_max = block_reduce_max(local_max, reduce_smem);
        __syncthreads();
        // Re-scale each thread's exp_sum to the global max
        local_sum = softmax_state.exp_sum * expf(softmax_state.max_val - local_max);
        local_sum = block_reduce_sum(local_sum, reduce_smem);
        __syncthreads();
        softmax_state.max_val = local_max;
        softmax_state.exp_sum = local_sum;

        // Accumulate weighted V values
        for (int t = threadIdx.x; t < tokens_in_block; t += NUM_THREADS) {
            float qk = 0.f;
#pragma unroll
            for (int d = 0; d < HEAD_SIZE; ++d)
                qk += q_smem[d] * to_float(k_block[d * BLOCK_SIZE + t]);
            qk *= scale;
            float w = softmax_state.normalise(qk);

#pragma unroll
            for (int d = 0; d < HEAD_SIZE; ++d)
                out_acc[d] += w * to_float(v_block[d * BLOCK_SIZE + t]);
        }
    }
    __syncthreads();

    // Write output
    scalar_t* out_ptr = out + seq_idx * gridDim.x * HEAD_SIZE
                            + head_idx * HEAD_SIZE;
    for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS)
        out_ptr[i] = from_float<scalar_t>(out_acc[i]);
}

// ---------------------------------------------------------------------------
// Host-side launcher
// ---------------------------------------------------------------------------

template <typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE>
static void paged_attention_v1_launcher(
    TrueBuffer& out,
    const TrueBuffer& q,
    const TrueBuffer& k_cache,
    const TrueBuffer& v_cache,
    const TrueBuffer& block_tables,
    const TrueBuffer& seq_lens,
    int num_kv_heads,
    float scale,
    cudaStream_t stream)
{
    int num_seqs  = static_cast<int>(tb_size(q, 0));
    int num_heads = static_cast<int>(tb_size(q, 1));
    int max_blocks = static_cast<int>(tb_size(block_tables, 1));

    constexpr int NUM_THREADS = 128;
    dim3 grid(num_heads, num_seqs);

    paged_attention_v1_kernel<scalar_t, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>
        <<<grid, NUM_THREADS, 0, stream>>>(
            tb_data<scalar_t>(out),
            tb_data<scalar_t>(q),
            tb_data<scalar_t>(k_cache),
            tb_data<scalar_t>(v_cache),
            tb_data<int32_t>(block_tables),
            tb_data<int32_t>(seq_lens),
            num_kv_heads,
            max_blocks,
            scale);
}

// ---------------------------------------------------------------------------
// Public API — dispatches on head_size and block_size at runtime
// ---------------------------------------------------------------------------

void paged_attention_v1(
    TrueBuffer& out,
    const TrueBuffer& q,
    const TrueBuffer& k_cache,
    const TrueBuffer& v_cache,
    const TrueBuffer& block_tables,
    const TrueBuffer& seq_lens,
    int num_kv_heads,
    float scale,
    int head_size,
    int block_size,
    cudaStream_t stream)
{
#define DISPATCH_HEAD_BLOCK(HS, BS, T)                                     \
    if (head_size == (HS) && block_size == (BS)) {                         \
        paged_attention_v1_launcher<T, (HS), (BS)>(                        \
            out, q, k_cache, v_cache, block_tables, seq_lens,              \
            num_kv_heads, scale, stream);                                  \
        return;                                                             \
    }

    if (q.dtype == DataType::F16) {
        DISPATCH_HEAD_BLOCK(64,  16, __half)
        DISPATCH_HEAD_BLOCK(80,  16, __half)
        DISPATCH_HEAD_BLOCK(96,  16, __half)
        DISPATCH_HEAD_BLOCK(128, 16, __half)
        DISPATCH_HEAD_BLOCK(256, 16, __half)
        DISPATCH_HEAD_BLOCK(128, 32, __half)
    } else {
        DISPATCH_HEAD_BLOCK(64,  16, float)
        DISPATCH_HEAD_BLOCK(128, 16, float)
        DISPATCH_HEAD_BLOCK(256, 16, float)
    }
#undef DISPATCH_HEAD_BLOCK
    // Unsupported head_size / block_size combination — caller must check
    assert(false && "paged_attention_v1: unsupported head_size or block_size");
}
