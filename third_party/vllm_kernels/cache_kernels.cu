// ---------------------------------------------------------------------------
// cache_kernels.cu — KV cache management for paged attention
//
// Vendored from vllm/csrc/cache_kernels.cu.
// Modifications: torch::Tensor replaced with TrueBuffer; pybind11 removed.
//
// Operations:
//   reshape_and_cache       — write new K,V into the correct paged block slots
//   reshape_and_cache_flash — flash-attention-compatible layout variant
//   copy_blocks             — copy entire blocks between physical block IDs
//   swap_blocks             — swap blocks between GPU and host-pinned memory
//   convert_fp8             — quantise an F16 block to FP8 in-place (sm_89+)
// ---------------------------------------------------------------------------

#include "include/vllm_types.h"
#include "include/dtype_float16.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// ---------------------------------------------------------------------------
// reshape_and_cache
//
// Writes freshly-computed key and value vectors into the KV cache block table.
//
// KV cache storage layout (two contiguous halves):
//   K pool (first half):  [num_blocks, num_kv_heads, head_size, block_size]
//   V pool (second half): [num_blocks, num_kv_heads, head_size, block_size]
//
//   k_cache_dev  = kv_cache_dev
//   v_cache_dev  = kv_cache_dev + num_blocks * num_kv_heads * head_size * block_size * elem_size
//
// This layout matches what paged_attention_v1/v2 kernels read:
//   k_block = k_cache + block_id * num_kv_heads * head_size * block_size
//             + kv_head * head_size * block_size
//   element access: k_block[dim * block_size + block_offset]
//
// key / value input layout:
//   [num_tokens, num_kv_heads, head_size]
//
// slot_mapping:
//   [num_tokens]  — physical slot = block_id * block_size + block_offset
//
// num_total_blocks is needed to compute the V pool offset.
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void reshape_and_cache_kernel(
    const scalar_t* __restrict__ key,          // [num_tokens, num_kv_heads, head_size]
    const scalar_t* __restrict__ value,        // [num_tokens, num_kv_heads, head_size]
    scalar_t* __restrict__ kv_cache,           // [num_total_blocks*2, num_kv_heads, head_size, block_size]
    const int32_t* __restrict__ slot_mapping,  // [num_tokens]
    int num_kv_heads,
    int head_size,
    int block_size,
    int num_total_blocks,
    int num_tokens)
{
    // Each thread handles one (token, head, dim) element.
    int64_t idx   = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(num_tokens) * num_kv_heads * head_size;
    if (idx >= total) return;

    int elem  = static_cast<int>(idx % head_size);
    int head  = static_cast<int>((idx / head_size) % num_kv_heads);
    int token = static_cast<int>(idx / (head_size * num_kv_heads));

    int32_t slot = slot_mapping[token];
    if (slot < 0) return; // padded token

    int block_id  = slot / block_size;
    int block_off = slot % block_size;

    // K layout: [num_blocks, num_kv_heads, head_size, block_size]
    //   stride per block      = num_kv_heads * head_size * block_size
    //   stride per head       = head_size * block_size
    //   stride per dim        = block_size
    //   stride per block_off  = 1
    int64_t block_stride = static_cast<int64_t>(num_kv_heads) * head_size * block_size;
    int64_t k_idx = static_cast<int64_t>(block_id) * block_stride
                  + static_cast<int64_t>(head) * head_size * block_size
                  + static_cast<int64_t>(elem) * block_size
                  + block_off;

    // V pool starts after all K blocks
    int64_t v_offset = static_cast<int64_t>(num_total_blocks) * block_stride;

    // Input index: [token, head, elem]
    int64_t in_idx = static_cast<int64_t>(token) * (num_kv_heads * head_size)
                   + static_cast<int64_t>(head) * head_size + elem;

    kv_cache[k_idx]            = key[in_idx];
    kv_cache[v_offset + k_idx] = value[in_idx];
}

// ---------------------------------------------------------------------------
// copy_blocks
//
// Bulk-copies complete blocks from a list of (src, dst) block ID pairs.
// Used for prefix caching and beam search.
//
// block_mapping:  [num_pairs, 2]  — {src_block_id, dst_block_id}
// kv_cache:       [num_blocks*2, num_kv_heads, head_size, block_size]
//                  first num_blocks entries = K pool, next = V pool
// gridDim.y = num_pairs * 2  — first half handles K pool, second half V pool
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void copy_blocks_kernel(
    scalar_t* __restrict__ kv_cache,
    const int32_t* __restrict__ block_mapping,
    int num_pairs,
    int half_stride,      // num_kv_heads * head_size * block_size  (per logical block, one pool)
    int num_half_blocks)  // num_blocks (entries per pool half)
{
    // blockIdx.y in [0, num_pairs)        → K pool copy
    // blockIdx.y in [num_pairs, 2*num_pairs) → V pool copy
    int pair_idx  = blockIdx.y % num_pairs;
    int64_t base  = (blockIdx.y >= num_pairs)
                  ? static_cast<int64_t>(num_half_blocks) * half_stride
                  : 0LL;

    int src = block_mapping[pair_idx * 2];
    int dst = block_mapping[pair_idx * 2 + 1];

    int elem = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (elem >= half_stride) return;

    kv_cache[base + static_cast<int64_t>(dst) * half_stride + elem] =
        kv_cache[base + static_cast<int64_t>(src) * half_stride + elem];
}

// ---------------------------------------------------------------------------
// swap_blocks (GPU ↔ host-pinned memory)
//
// Used for sequence eviction.  Copies full blocks between device memory
// (kv_cache) and a host-pinned buffer.
//
// direction: 0 = GPU→host,  1 = host→GPU
// block_mapping: [num_pairs, 2]  src/dst indices into respective buffers
// ---------------------------------------------------------------------------
__global__ void swap_blocks_kernel(
    void* __restrict__ gpu_buf,
    void* __restrict__ host_buf,
    const int32_t* __restrict__ block_mapping,
    int num_pairs,
    int block_bytes,
    int direction)
{
    int pair_idx = blockIdx.y;
    if (pair_idx >= num_pairs) return;

    int src = block_mapping[pair_idx * 2];
    int dst = block_mapping[pair_idx * 2 + 1];

    int byte = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (byte >= block_bytes) return;

    char* gpu  = static_cast<char*>(gpu_buf);
    char* host = static_cast<char*>(host_buf);

    if (direction == 0) {
        // GPU → host
        host[static_cast<int64_t>(dst) * block_bytes + byte] =
            gpu[static_cast<int64_t>(src) * block_bytes + byte];
    } else {
        // host → GPU
        gpu[static_cast<int64_t>(dst) * block_bytes + byte] =
            host[static_cast<int64_t>(src) * block_bytes + byte];
    }
}

// ---------------------------------------------------------------------------
// convert_fp8 (Ada Lovelace sm_89+ only)
//
// Converts an F16 KV cache block to FP8 E4M3 in-place.
// This halves the VRAM footprint of the KV cache.
// ---------------------------------------------------------------------------
#if defined(TRUELLM_CUDA_SM89_PLUS)
#include "include/quantization/fp8/fp8_utils.cuh"

__global__ void convert_fp8_kernel(
    __nv_fp8_e4m3* __restrict__ out,
    const __half* __restrict__ in,
    int64_t n_elements,
    float scale)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        out[idx] = half_to_fp8e4m3(in[idx], scale);
    }
#endif
}
#endif // TRUELLM_CUDA_SM89_PLUS

// ---------------------------------------------------------------------------
// Host-side launch wrappers
// ---------------------------------------------------------------------------

static constexpr int CACHE_BLOCK = 256;

void reshape_and_cache(
    const TrueBuffer& key,
    const TrueBuffer& value,
    TrueBuffer& kv_cache,
    const TrueBuffer& slot_mapping,
    cudaStream_t stream)
{
    int num_tokens       = static_cast<int>(tb_size(key, 0));
    int num_kv_heads     = static_cast<int>(tb_size(key, 1));
    int head_size        = static_cast<int>(tb_size(key, 2));
    // kv_cache shape: [num_blocks*2, num_kv_heads, head_size, block_size]
    int num_total_blocks = static_cast<int>(tb_size(kv_cache, 0)) / 2;
    int block_size       = static_cast<int>(tb_size(kv_cache, 3));

    int64_t total = static_cast<int64_t>(num_tokens) * num_kv_heads * head_size;
    int grid = static_cast<int>((total + CACHE_BLOCK - 1) / CACHE_BLOCK);

    if (key.dtype == DataType::F16) {
        reshape_and_cache_kernel<__half><<<grid, CACHE_BLOCK, 0, stream>>>(
            tb_data<__half>(key), tb_data<__half>(value),
            tb_data<__half>(kv_cache),
            tb_data<int32_t>(slot_mapping),
            num_kv_heads, head_size, block_size, num_total_blocks, num_tokens);
    } else {
        reshape_and_cache_kernel<float><<<grid, CACHE_BLOCK, 0, stream>>>(
            tb_data<float>(key), tb_data<float>(value),
            tb_data<float>(kv_cache),
            tb_data<int32_t>(slot_mapping),
            num_kv_heads, head_size, block_size, num_total_blocks, num_tokens);
    }
}

void copy_blocks(
    TrueBuffer& kv_cache,
    const TrueBuffer& block_mapping,
    cudaStream_t stream)
{
    int num_pairs       = static_cast<int>(tb_size(block_mapping, 0));
    // kv_cache shape: [num_blocks*2, num_kv_heads, head_size, block_size]
    int num_half_blocks = static_cast<int>(tb_size(kv_cache, 0)) / 2;
    int half_stride     = static_cast<int>(
        tb_size(kv_cache, 1) * tb_size(kv_cache, 2) * tb_size(kv_cache, 3));

    // gridDim.y = num_pairs*2: first num_pairs rows copy K pool, next V pool
    dim3 grid(
        (half_stride + CACHE_BLOCK - 1) / CACHE_BLOCK,
        num_pairs * 2);

    if (kv_cache.dtype == DataType::F16) {
        copy_blocks_kernel<__half><<<grid, CACHE_BLOCK, 0, stream>>>(
            tb_data<__half>(kv_cache),
            tb_data<int32_t>(block_mapping),
            num_pairs, half_stride, num_half_blocks);
    } else {
        copy_blocks_kernel<float><<<grid, CACHE_BLOCK, 0, stream>>>(
            tb_data<float>(kv_cache),
            tb_data<int32_t>(block_mapping),
            num_pairs, half_stride, num_half_blocks);
    }
}

void swap_blocks(
    TrueBuffer& kv_cache,
    TrueBuffer& host_cache,
    const TrueBuffer& block_mapping,
    int direction,
    cudaStream_t stream)
{
    int num_pairs = static_cast<int>(tb_size(block_mapping, 0));
    int block_bytes = static_cast<int>(
        tb_size(kv_cache, 1) * tb_size(kv_cache, 2) *
        tb_size(kv_cache, 3) * tb_size(kv_cache, 4))
        * static_cast<int>(2 /* F16 bytes */);

    dim3 grid(
        (block_bytes + CACHE_BLOCK - 1) / CACHE_BLOCK,
        num_pairs);

    swap_blocks_kernel<<<grid, CACHE_BLOCK, 0, stream>>>(
        kv_cache.data, host_cache.data,
        tb_data<int32_t>(block_mapping),
        num_pairs, block_bytes, direction);
}

#if defined(TRUELLM_CUDA_SM89_PLUS)
void convert_fp8(TrueBuffer& out, const TrueBuffer& in, float scale, cudaStream_t stream)
{
    int64_t n = tb_numel(in);
    int grid = static_cast<int>((n + CACHE_BLOCK - 1) / CACHE_BLOCK);
    convert_fp8_kernel<<<grid, CACHE_BLOCK, 0, stream>>>(
        reinterpret_cast<__nv_fp8_e4m3*>(out.data),
        tb_data<__half>(in),
        n, scale);
}
#endif
