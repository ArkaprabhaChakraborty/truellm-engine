// ---------------------------------------------------------------------------
// vq_cache.cu — Vector-quantized KV cache compression (engine-internal)
// ---------------------------------------------------------------------------

#include "vq_cache.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>

namespace truellm {

// ---------------------------------------------------------------------------
// vq_encode_kernel
//
// Each thread handles one (token, head) pair.
// Exhaustive L2 search over codebook — practical for codebook_size ≤ 256.
// ---------------------------------------------------------------------------
extern "C" __global__ void vq_encode_kernel(
    const __half* __restrict__ d_kv,
    const __half* __restrict__ d_book,
    uint16_t*    __restrict__ d_idx,
    int seq_len, int n_kv_heads, int head_dim, int codebook_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * n_kv_heads;
    if (tid >= total) return;

    int tok  = tid / n_kv_heads;
    int head = tid % n_kv_heads;

    const __half* vec  = d_kv   + (tok * n_kv_heads + head) * head_dim;
    float best_dist = FLT_MAX;
    int   best_idx  = 0;

    for (int c = 0; c < codebook_size; ++c) {
        const __half* entry = d_book + c * head_dim;
        float dist = 0.f;
        for (int d = 0; d < head_dim; ++d) {
            float diff = __half2float(vec[d]) - __half2float(entry[d]);
            dist += diff * diff;
        }
        if (dist < best_dist) { best_dist = dist; best_idx = c; }
    }
    d_idx[tok * n_kv_heads + head] = static_cast<uint16_t>(best_idx);
}

// ---------------------------------------------------------------------------
// vq_decode_kernel
// ---------------------------------------------------------------------------
extern "C" __global__ void vq_decode_kernel(
    __half*        __restrict__ d_kv_out,
    const __half*  __restrict__ d_book,
    const uint16_t* __restrict__ d_idx,
    int seq_len, int n_kv_heads, int head_dim)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * n_kv_heads;
    if (tid >= total) return;

    int tok  = tid / n_kv_heads;
    int head = tid % n_kv_heads;

    int codebook_idx = static_cast<int>(d_idx[tok * n_kv_heads + head]);
    const __half* entry = d_book + codebook_idx * head_dim;
    __half* out_vec = d_kv_out  + (tok * n_kv_heads + head) * head_dim;

    for (int d = 0; d < head_dim; ++d)
        out_vec[d] = entry[d];
}

// ---------------------------------------------------------------------------
// Host wrappers
// ---------------------------------------------------------------------------
void vq_encode(
    const __half* d_kv,
    const __half* d_book,
    uint16_t*     d_idx,
    int seq_len, int n_kv_heads, int head_dim, int codebook_size,
    cudaStream_t stream)
{
    int total = seq_len * n_kv_heads;
    int threads = 256;
    int blocks  = (total + threads - 1) / threads;
    vq_encode_kernel<<<blocks, threads, 0, stream>>>(
        d_kv, d_book, d_idx, seq_len, n_kv_heads, head_dim, codebook_size);
}

void vq_decode(
    __half*        d_kv,
    const __half*  d_book,
    const uint16_t* d_idx,
    int seq_len, int n_kv_heads, int head_dim,
    cudaStream_t stream)
{
    int total = seq_len * n_kv_heads;
    int threads = 256;
    int blocks  = (total + threads - 1) / threads;
    vq_decode_kernel<<<blocks, threads, 0, stream>>>(
        d_kv, d_book, d_idx, seq_len, n_kv_heads, head_dim);
}

} // namespace truellm
