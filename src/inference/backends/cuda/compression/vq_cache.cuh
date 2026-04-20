#pragma once
// ---------------------------------------------------------------------------
// vq_cache.cuh — Vector-quantized KV cache compression
//
// Moved from truellm-native-plugins/plugins/kv_compression_kernel/.
// Now compiled as part of truellm_backend_cuda (first-party engine component).
// External plugin authors who need VQ-style compression should use
// host->get_kv_cache_tensor() via the plugin vtable instead.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace truellm {

struct VqConfig;  // defined in config_schema.h — forward declaration avoids
                  // pulling the full header into every .cu compilation unit

// Max codebook entries (uint16 index → fits 65536, we cap at 1024 in practice)
static constexpr int VQ_MAX_CODEBOOK = 1024;

// ---------------------------------------------------------------------------
// Device kernels (declared here, defined in vq_cache.cu)
// ---------------------------------------------------------------------------

// vq_encode_kernel: for each KV vector in [seq_len × n_kv_heads × head_dim],
// find the nearest codebook entry by L2 distance and store the index.
//   d_kv     : input  [seq_len, n_kv_heads, head_dim] F16
//   d_book   : codebook [codebook_size, head_dim] F16
//   d_idx    : output  [seq_len, n_kv_heads] uint16
//   seq_len, n_kv_heads, head_dim, codebook_size as grid-stride dimensions
extern "C" __global__ void vq_encode_kernel(
    const __half* __restrict__ d_kv,
    const __half* __restrict__ d_book,
    uint16_t*    __restrict__ d_idx,
    int seq_len, int n_kv_heads, int head_dim, int codebook_size);

// vq_decode_kernel: replace each KV vector with its codebook entry.
//   d_kv_out : output [seq_len, n_kv_heads, head_dim] F16 (can alias d_kv)
//   d_book   : codebook [codebook_size, head_dim] F16
//   d_idx    : indices  [seq_len, n_kv_heads] uint16
extern "C" __global__ void vq_decode_kernel(
    __half*       __restrict__ d_kv_out,
    const __half* __restrict__ d_book,
    const uint16_t* __restrict__ d_idx,
    int seq_len, int n_kv_heads, int head_dim);

// ---------------------------------------------------------------------------
// Host wrappers
// ---------------------------------------------------------------------------

// vq_encode: launch vq_encode_kernel and return immediately (async on stream).
// d_idx must be pre-allocated: seq_len × n_kv_heads × sizeof(uint16_t).
void vq_encode(
    const __half* d_kv,
    const __half* d_book,
    uint16_t*     d_idx,
    int seq_len, int n_kv_heads, int head_dim, int codebook_size,
    cudaStream_t stream);

// vq_decode: replace KV tensor entries in-place using codebook lookup.
void vq_decode(
    __half*        d_kv,        // in-place
    const __half*  d_book,
    const uint16_t* d_idx,
    int seq_len, int n_kv_heads, int head_dim,
    cudaStream_t stream);

} // namespace truellm
