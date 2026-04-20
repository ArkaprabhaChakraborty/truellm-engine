#pragma once
// ---------------------------------------------------------------------------
// rope_device.cuh — Rotary Position Embedding applied in device code.
//
// Standard RoPE formula: for each pair (x_{2i}, x_{2i+1}):
//   out_{2i}   = x_{2i}   * cos(θ_i * pos) - x_{2i+1} * sin(θ_i * pos)
//   out_{2i+1} = x_{2i+1} * cos(θ_i * pos) + x_{2i}   * sin(θ_i * pos)
//
// cos_cache / sin_cache: [max_seq_len, head_dim/2] F32, pre-computed.
// head_dim must be even.
//
// Called per-head inside the megakernel QK computation.
// ---------------------------------------------------------------------------

#pragma once
#include <cuda_fp16.h>
#include <cstdint>

// Apply RoPE to a single head of head_dim F16 values in-place.
// pos: absolute sequence position.
// cos/sin: [max_seq_len, head_dim/2] F32.
__device__ __forceinline__
void rope_apply_f16(__half* x, int head_dim, int pos,
                    const float* cos_cache, const float* sin_cache)
{
    const int half_dim = head_dim / 2;
    for (int i = threadIdx.x; i < half_dim; i += blockDim.x) {
        float xi   = __half2float(x[i]);
        float xi1  = __half2float(x[i + half_dim]);
        float c    = cos_cache[pos * half_dim + i];
        float s    = sin_cache[pos * half_dim + i];
        x[i]          = __float2half(xi * c - xi1 * s);
        x[i + half_dim] = __float2half(xi1 * c + xi * s);
    }
}
