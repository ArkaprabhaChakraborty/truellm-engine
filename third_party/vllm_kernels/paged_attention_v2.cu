// ---------------------------------------------------------------------------
// paged_attention_v2.cu — Two-pass paged attention for long contexts
//
// Vendored from vllm/csrc/attention/attention_kernels.cu (v2 path).
// Modifications: torch::Tensor replaced with TrueBuffer; pybind11 removed.
//
// Use v2 when max_context_len / BLOCK_SIZE > PARTITION_THRESHOLD (32).
// This splits the KV sequence into independent partitions, each processed
// by a separate thread block.  A second reduction pass merges them.
//
// Pass 1: each block computes partial {out, exp_sum, max_logit} for its
//         partition of the key/value sequence.
// Pass 2: one block per (seq, head) reduces across partitions.
// ---------------------------------------------------------------------------

#include "include/vllm_types.h"
#include "include/attention_utils.cuh"
#include "include/reduce_kernel_utils.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>

// How many KV blocks each partition covers.
static constexpr int PARTITION_SIZE = 32;  // blocks per partition

// ---------------------------------------------------------------------------
// Pass 1 kernel — per-partition partial attention
// ---------------------------------------------------------------------------

template <typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS>
__global__ void paged_attention_v2_pass1_kernel(
    float* __restrict__ exp_sums,            // [num_seqs, num_heads, num_partitions]
    float* __restrict__ max_logits,          // [num_seqs, num_heads, num_partitions]
    scalar_t* __restrict__ tmp_out,          // [num_seqs, num_heads, num_partitions, head_size]
    const scalar_t* __restrict__ q,          // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ k_cache,    // [num_blocks, num_kv_heads, head_size, block_size]
    const scalar_t* __restrict__ v_cache,    // [num_blocks, num_kv_heads, head_size, block_size]
    const int32_t* __restrict__ block_tables, // [num_seqs, max_blocks_per_seq]
    const int32_t* __restrict__ seq_lens,    // [num_seqs]
    int num_kv_heads,
    int max_blocks_per_seq,
    int num_partitions,
    float scale)
{
    // Grid: (num_partitions, num_heads, num_seqs)
    int seq_idx       = blockIdx.z;
    int head_idx      = blockIdx.y;
    int partition_idx = blockIdx.x;
    int kv_head       = head_idx % num_kv_heads;

    int seq_len = seq_lens[seq_idx];
    if (seq_len == 0) return;

    int block_start = partition_idx * PARTITION_SIZE;
    int block_end   = min(block_start + PARTITION_SIZE,
                          (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE);
    if (block_start >= block_end) return;

    __shared__ float q_smem[HEAD_SIZE];
    __shared__ float reduce_smem[NUM_THREADS / 32];

    const scalar_t* q_ptr = q + seq_idx * gridDim.y * HEAD_SIZE
                               + head_idx * HEAD_SIZE;
    for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS)
        q_smem[i] = to_float(q_ptr[i]);
    __syncthreads();

    SoftmaxState ss;
    float out_acc[HEAD_SIZE] = {};

    for (int b = block_start; b < block_end; ++b) {
        int32_t phys = block_tables[seq_idx * max_blocks_per_seq + b];
        if (phys < 0) continue;
        int tokens = min(BLOCK_SIZE, seq_len - b * BLOCK_SIZE);

        const scalar_t* k_blk = k_cache
            + static_cast<int64_t>(phys) * num_kv_heads * HEAD_SIZE * BLOCK_SIZE
            + kv_head * HEAD_SIZE * BLOCK_SIZE;
        const scalar_t* v_blk = v_cache
            + static_cast<int64_t>(phys) * num_kv_heads * HEAD_SIZE * BLOCK_SIZE
            + kv_head * HEAD_SIZE * BLOCK_SIZE;

        for (int t = threadIdx.x; t < tokens; t += NUM_THREADS) {
            float qk = 0.f;
#pragma unroll
            for (int d = 0; d < HEAD_SIZE; ++d)
                qk += q_smem[d] * to_float(k_blk[d * BLOCK_SIZE + t]);
            qk *= scale;
            ss.update(qk);
        }

        float lmax = block_reduce_max(ss.max_val, reduce_smem);
        __syncthreads();
        float lsum = ss.exp_sum * expf(ss.max_val - lmax);
        lsum = block_reduce_sum(lsum, reduce_smem);
        __syncthreads();
        ss.max_val = lmax;
        ss.exp_sum = lsum;

        for (int t = threadIdx.x; t < tokens; t += NUM_THREADS) {
            float qk = 0.f;
#pragma unroll
            for (int d = 0; d < HEAD_SIZE; ++d)
                qk += q_smem[d] * to_float(k_blk[d * BLOCK_SIZE + t]);
            qk *= scale;
            float w = ss.normalise(qk);
#pragma unroll
            for (int d = 0; d < HEAD_SIZE; ++d)
                out_acc[d] += w * to_float(v_blk[d * BLOCK_SIZE + t]);
        }
    }

    // Write partial results
    int64_t base = (static_cast<int64_t>(seq_idx) * gridDim.y + head_idx)
                 * num_partitions + partition_idx;
    if (threadIdx.x == 0) {
        exp_sums [base] = ss.exp_sum;
        max_logits[base] = ss.max_val;
    }

    scalar_t* tmp = tmp_out + base * HEAD_SIZE;
    for (int d = threadIdx.x; d < HEAD_SIZE; d += NUM_THREADS)
        tmp[d] = from_float<scalar_t>(out_acc[d]);
}

// ---------------------------------------------------------------------------
// Pass 2 kernel — reduce partitions
// ---------------------------------------------------------------------------

template <typename scalar_t, int HEAD_SIZE>
__global__ void paged_attention_v2_pass2_kernel(
    scalar_t* __restrict__ out,
    const float* __restrict__ exp_sums,
    const float* __restrict__ max_logits,
    const scalar_t* __restrict__ tmp_out,
    int num_partitions)
{
    // Grid: (num_heads, num_seqs)
    int seq_idx  = blockIdx.y;
    int head_idx = blockIdx.x;

    float global_max = -FLT_MAX;
    float global_sum = 0.f;

    int64_t base_ph = (static_cast<int64_t>(seq_idx) * gridDim.x + head_idx)
                    * num_partitions;

    // Find global max across partitions
    for (int p = 0; p < num_partitions; ++p)
        global_max = fmaxf(global_max, max_logits[base_ph + p]);

    // Sum re-scaled exp_sums
    for (int p = 0; p < num_partitions; ++p)
        global_sum += exp_sums[base_ph + p] * expf(max_logits[base_ph + p] - global_max);

    // Weighted sum of partial out vectors
    scalar_t* out_ptr = out + (static_cast<int64_t>(seq_idx) * gridDim.x
                              + head_idx) * HEAD_SIZE;
    for (int d = threadIdx.x; d < HEAD_SIZE; d += blockDim.x) {
        float acc = 0.f;
        for (int p = 0; p < num_partitions; ++p) {
            float w = exp_sums[base_ph + p]
                    * expf(max_logits[base_ph + p] - global_max)
                    / global_sum;
            acc += w * to_float(tmp_out[(base_ph + p) * HEAD_SIZE + d]);
        }
        out_ptr[d] = from_float<scalar_t>(acc);
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void paged_attention_v2(
    TrueBuffer& out,
    TrueBuffer& exp_sums,
    TrueBuffer& max_logits,
    TrueBuffer& tmp_out,
    const TrueBuffer& q,
    const TrueBuffer& k_cache,
    const TrueBuffer& v_cache,
    const TrueBuffer& block_tables,
    const TrueBuffer& seq_lens,
    int num_kv_heads,
    float scale,
    int head_size,
    int block_size,
    int max_context_len,
    cudaStream_t stream)
{
    int num_seqs      = static_cast<int>(tb_size(q, 0));
    int num_heads     = static_cast<int>(tb_size(q, 1));
    int max_blocks    = static_cast<int>(tb_size(block_tables, 1));
    int num_partitions = (max_context_len / block_size + PARTITION_SIZE - 1) / PARTITION_SIZE;

    constexpr int NT = 128;

    // Pass 1 — one block per (partition, head, seq)
    {
        dim3 grid(num_partitions, num_heads, num_seqs);
        if (q.dtype == DataType::F16 && head_size == 128 && block_size == 16) {
            paged_attention_v2_pass1_kernel<__half, 128, 16, NT>
                <<<grid, NT, 0, stream>>>(
                    tb_data<float>(exp_sums),
                    tb_data<float>(max_logits),
                    tb_data<__half>(tmp_out),
                    tb_data<__half>(q),
                    tb_data<__half>(k_cache),
                    tb_data<__half>(v_cache),
                    tb_data<int32_t>(block_tables),
                    tb_data<int32_t>(seq_lens),
                    num_kv_heads, max_blocks, num_partitions, scale);
        } else {
            paged_attention_v2_pass1_kernel<float, 128, 16, NT>
                <<<grid, NT, 0, stream>>>(
                    tb_data<float>(exp_sums),
                    tb_data<float>(max_logits),
                    tb_data<float>(tmp_out),
                    tb_data<float>(q),
                    tb_data<float>(k_cache),
                    tb_data<float>(v_cache),
                    tb_data<int32_t>(block_tables),
                    tb_data<int32_t>(seq_lens),
                    num_kv_heads, max_blocks, num_partitions, scale);
        }
    }

    // Pass 2 — one block per (head, seq)
    {
        dim3 grid(num_heads, num_seqs);
        if (q.dtype == DataType::F16 && head_size == 128) {
            paged_attention_v2_pass2_kernel<__half, 128>
                <<<grid, NT, 0, stream>>>(
                    tb_data<__half>(out),
                    tb_data<float>(exp_sums),
                    tb_data<float>(max_logits),
                    tb_data<__half>(tmp_out),
                    num_partitions);
        } else {
            paged_attention_v2_pass2_kernel<float, 128>
                <<<grid, NT, 0, stream>>>(
                    tb_data<float>(out),
                    tb_data<float>(exp_sums),
                    tb_data<float>(max_logits),
                    tb_data<float>(tmp_out),
                    num_partitions);
        }
    }
}
