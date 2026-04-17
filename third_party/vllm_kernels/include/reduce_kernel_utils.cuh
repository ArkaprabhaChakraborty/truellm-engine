#pragma once
// ---------------------------------------------------------------------------
// reduce_kernel_utils.cuh — Warp and block reduction primitives
//
// Used by paged_attention_v1/v2, layernorm, and activation kernels.
// These are device-side only (__device__ functions).
// ---------------------------------------------------------------------------

#include <cuda_fp16.h>
#include <float.h>

// ---------------------------------------------------------------------------
// Warp reductions (within one 32-thread warp)
// ---------------------------------------------------------------------------

// Sum reduction across a warp.
template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Max reduction across a warp.
template <typename T>
__device__ __forceinline__ T warp_reduce_max(T val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = max(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

// ---------------------------------------------------------------------------
// Block reductions via shared memory
// Requires shared memory buffer of size (blockDim.x / 32) elements.
// ---------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ T block_reduce_sum(T val, T* smem) {
    int lane   = threadIdx.x & 31;
    int warpid = threadIdx.x >> 5;

    val = warp_reduce_sum(val);

    if (lane == 0) smem[warpid] = val;
    __syncthreads();

    if (warpid == 0) {
        val = (threadIdx.x < (blockDim.x >> 5)) ? smem[lane] : T(0);
        val = warp_reduce_sum(val);
    }
    return val;
}

template <typename T>
__device__ __forceinline__ T block_reduce_max(T val, T* smem) {
    int lane   = threadIdx.x & 31;
    int warpid = threadIdx.x >> 5;

    val = warp_reduce_max(val);

    if (lane == 0) smem[warpid] = val;
    __syncthreads();

    if (warpid == 0) {
        val = (threadIdx.x < (blockDim.x >> 5)) ? smem[lane] : T(-FLT_MAX);
        val = warp_reduce_max(val);
    }
    return val;
}

// ---------------------------------------------------------------------------
// Warp-level prefix sum (exclusive scan)
// Used in softmax normalisation passes.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_prefix_sum(float val) {
    for (int offset = 1; offset < 32; offset <<= 1) {
        float n = __shfl_up_sync(0xffffffff, val, offset);
        if ((threadIdx.x & 31) >= offset) val += n;
    }
    return val;
}
