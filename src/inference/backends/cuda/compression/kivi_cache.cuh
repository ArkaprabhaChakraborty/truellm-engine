#pragma once
// ---------------------------------------------------------------------------
// kivi_cache.cuh — KiVi channel-wise outlier + INT4 group-quantized KV cache
//
// Moved from truellm-native-plugins/plugins/kv_compression_kernel/.
// Compressed layout per KV vector (head_dim elements):
//   [n_outliers × (channel_idx: uint8, value: F16)]   — high-magnitude channels
//   [n_groups × (INT4 × group_size, scale: F16, zero: F16)]  — residual channels
//
// Compression ratio: head_dim=128, n_outliers=8, group_size=16 →
//   original: 128 × 2 = 256 bytes
//   compressed: 8×3 + (120/16)×(8+4) = 24 + 90 = 114 bytes  (~55% of original)
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace truellm {

// Hard upper bounds dictated by the on-stack arrays in kivi_compress_kernel /
// kivi_decompress_kernel.  `is_outlier[]`, `residuals[]`, and `result[]` are
// sized to KIVI_MAX_HEAD_DIM; `outlier_idx[]` / `outlier_mag[]` are sized to
// KIVI_MAX_OUTLIERS.  The host wrappers MUST clamp inputs to these limits;
// the kernels also clamp defensively so an out-of-range call degrades to a
// no-op rather than corrupting the stack.
constexpr int KIVI_MAX_HEAD_DIM = 128;
constexpr int KIVI_MAX_OUTLIERS = 64;

// Returns the number of bytes for one compressed KV vector.
// n_residual = head_dim - n_outliers (must be divisible by group_size)
inline __host__ __device__ size_t kivi_compressed_bytes(
    int head_dim, int n_outliers, int group_size)
{
    int n_res    = head_dim - n_outliers;
    int n_groups = (n_res + group_size - 1) / group_size;
    // outlier section: each outlier is (uint8 channel idx + F16 value) = 3 bytes
    size_t outlier_bytes = static_cast<size_t>(n_outliers) * 3;
    // residual section: each group has group_size INT4 values packed as
    // group_size/2 bytes, plus F16 scale + F16 zero = 4 bytes
    size_t residual_bytes = static_cast<size_t>(n_groups) * (group_size / 2 + 4);
    return outlier_bytes + residual_bytes;
}

// ---------------------------------------------------------------------------
// Device kernels (defined in kivi_cache.cu)
// ---------------------------------------------------------------------------

// kivi_compress_kernel: compress one KV vector per thread.
//   d_kv_in  : [n_vecs, head_dim] F16
//   d_out    : [n_vecs, kivi_compressed_bytes(head_dim, n_outliers, group_size)] uint8
extern "C" __global__ void kivi_compress_kernel(
    const __half* __restrict__ d_kv_in,
    uint8_t*      __restrict__ d_out,
    int n_vecs, int head_dim, int n_outliers, int group_size);

// kivi_decompress_kernel: reconstruct F16 KV vectors from compressed layout.
extern "C" __global__ void kivi_decompress_kernel(
    const uint8_t* __restrict__ d_in,
    __half*        __restrict__ d_kv_out,
    int n_vecs, int head_dim, int n_outliers, int group_size);

// ---------------------------------------------------------------------------
// Host wrappers
// ---------------------------------------------------------------------------

// Returns true iff (head_dim, n_outliers, group_size) fits the kernels'
// fixed-size on-stack buffers.  Hosts should call this before invoking
// kivi_compress / kivi_decompress; out-of-range inputs cause both wrappers
// to become no-ops.  Exposed for unit tests and for KvCompressionPass::init
// to decide whether to enable KiVi.
bool kivi_params_supported(int head_dim, int n_outliers, int group_size);

void kivi_compress(
    const __half* d_kv_in,
    uint8_t*      d_out,
    int n_vecs, int head_dim, int n_outliers, int group_size,
    cudaStream_t stream);

void kivi_decompress(
    const uint8_t* d_in,
    __half*        d_kv_out,
    int n_vecs, int head_dim, int n_outliers, int group_size,
    cudaStream_t stream);

} // namespace truellm
