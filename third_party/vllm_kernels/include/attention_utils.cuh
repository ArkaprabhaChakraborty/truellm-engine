#pragma once
// ---------------------------------------------------------------------------
// attention_utils.cuh — Device-side utilities shared by paged attention kernels
//
// Includes:
//   - Rotary Position Embedding (RoPE) application
//   - Softmax helper (online / numerically-stable)
//   - FP8 up-cast helpers used during attention QK scoring
// ---------------------------------------------------------------------------

#include "dtype_float16.cuh"
#include "reduce_kernel_utils.cuh"
#include <cuda_fp16.h>
#include <math.h>

// ---------------------------------------------------------------------------
// Rotary Position Embedding
//
// Applies the RoPE transform in-place to a query or key vector.
//   x      : pointer to the head's elements (interleaved pairs)
//   cos    : precomputed cos(pos * freq) for this position, half the head size
//   sin    : precomputed sin(pos * freq) for this position, half the head size
//   head_size : full dimension of one head
// ---------------------------------------------------------------------------
template <typename scalar_t>
__device__ __forceinline__ void apply_rotary_emb(
    scalar_t* __restrict__ x,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    int head_size)
{
    int half = head_size / 2;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        float x0 = to_float(x[i]);
        float x1 = to_float(x[i + half]);
        x[i]        = from_float<scalar_t>(x0 * cos[i] - x1 * sin[i]);
        x[i + half] = from_float<scalar_t>(x0 * sin[i] + x1 * cos[i]);
    }
}

// ---------------------------------------------------------------------------
// Online (single-pass) softmax helper
//
// Maintains running max and sum for numerically stable exponentiation.
// Used by paged_attention_v1 where the full sequence fits in one pass.
// ---------------------------------------------------------------------------
struct SoftmaxState {
    float max_val = -1e20f;
    float exp_sum = 0.f;

    __device__ __forceinline__ void update(float qk) {
        float new_max = fmaxf(max_val, qk);
        exp_sum = exp_sum * expf(max_val - new_max) + expf(qk - new_max);
        max_val = new_max;
    }

    // Merge two partial SoftmaxStates (for parallel reduction).
    __device__ __forceinline__ void merge(const SoftmaxState& other) {
        float new_max = fmaxf(max_val, other.max_val);
        exp_sum = exp_sum * expf(max_val - new_max)
                + other.exp_sum * expf(other.max_val - new_max);
        max_val = new_max;
    }

    __device__ __forceinline__ float normalise(float qk) const {
        return expf(qk - max_val) / exp_sum;
    }
};

// ---------------------------------------------------------------------------
// Warp-level SoftmaxState reduction
// ---------------------------------------------------------------------------
__device__ __forceinline__ SoftmaxState warp_reduce_softmax(SoftmaxState s) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        SoftmaxState other;
        other.max_val = __shfl_xor_sync(0xffffffff, s.max_val, offset);
        other.exp_sum = __shfl_xor_sync(0xffffffff, s.exp_sum, offset);
        s.merge(other);
    }
    return s;
}

// ---------------------------------------------------------------------------
// Scaled dot-product helper
// Computes qk = dot(q_vec, k_vec) * scale for one (q,k) head pair.
//   q, k   : device pointers to the head vectors
//   size   : head dimension
//   scale  : 1/sqrt(head_dim)
// ---------------------------------------------------------------------------
template <typename scalar_t>
__device__ __forceinline__ float scaled_dot(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    int size,
    float scale)
{
    float acc = 0.f;
    for (int i = 0; i < size; ++i) {
        acc += to_float(q[i]) * to_float(k[i]);
    }
    return acc * scale;
}
