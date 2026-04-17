// ---------------------------------------------------------------------------
// fp8_gemm.cu — FP8 quantisation helpers for Ada Lovelace / Hopper (sm_89+)
//
// Vendored from vllm/csrc/quantization/fp8/fp8_utils.cu (simplified).
// Modifications: torch::Tensor replaced with TrueBuffer; pybind11 removed.
//
// Only compiled when TRUELLM_CUDA_SM89_PLUS is defined (CMake sets this
// when CMAKE_CUDA_ARCHITECTURES contains an arch >= 89).
//
// Exported functions:
//   scaled_fp8_quant  — quantise an F16 tensor to FP8 E4M3 with scale factor
//   scaled_fp8_dequant — dequantise FP8 E4M3 back to F16
// ---------------------------------------------------------------------------

#include "include/vllm_types.h"
#include "include/dtype_float16.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#if defined(TRUELLM_CUDA_SM89_PLUS)
#include "include/quantization/fp8/fp8_utils.cuh"

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// Per-tensor quantisation: one scale factor for the entire tensor.
__global__ void scaled_fp8_quant_kernel(
    __nv_fp8_e4m3* __restrict__ out,
    const __half* __restrict__ in,
    int64_t n_elements,
    float scale)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n_elements)
        out[idx] = half_to_fp8e4m3(in[idx], scale);
}

__global__ void scaled_fp8_dequant_kernel(
    __half* __restrict__ out,
    const __nv_fp8_e4m3* __restrict__ in,
    int64_t n_elements,
    float inv_scale)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n_elements)
        out[idx] = fp8e4m3_to_half(in[idx], inv_scale);
}

// Per-token quantisation: one scale per row (token).
// out   : [num_tokens, hidden]  FP8
// scale : [num_tokens]          per-token scale
// in    : [num_tokens, hidden]  F16
__global__ void per_token_fp8_quant_kernel(
    __nv_fp8_e4m3* __restrict__ out,
    float* __restrict__ scale_out,
    const __half* __restrict__ in,
    int num_tokens,
    int hidden_size)
{
    int token = blockIdx.x;
    if (token >= num_tokens) return;

    // Find max abs value in the row
    const __half* row_in = in + token * hidden_size;
    float amax = 0.f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x)
        amax = fmaxf(amax, fabsf(__half2float(row_in[i])));

    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset));

    // Broadcast via shared memory for block-level reduce
    __shared__ float smem[32];
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = amax;
    __syncthreads();
    if (threadIdx.x == 0) {
        float gmax = 0.f;
        for (int w = 0; w < (blockDim.x >> 5); ++w)
            gmax = fmaxf(gmax, smem[w]);
        // FP8 E4M3 max representable value is 448.0
        scale_out[token] = gmax / 448.0f;
        smem[0] = scale_out[token];
    }
    __syncthreads();
    float scale = smem[0];
    if (scale == 0.f) scale = 1.f;

    __nv_fp8_e4m3* row_out = out + token * hidden_size;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x)
        row_out[i] = half_to_fp8e4m3(row_in[i], 1.f / scale);
}

// ---------------------------------------------------------------------------
// Host-side wrappers
// ---------------------------------------------------------------------------

static constexpr int FP8_BLOCK = 256;

void scaled_fp8_quant(TrueBuffer& out, const TrueBuffer& in, float scale,
                      cudaStream_t stream)
{
    int64_t n = tb_numel(in);
    int grid  = static_cast<int>((n + FP8_BLOCK - 1) / FP8_BLOCK);
    scaled_fp8_quant_kernel<<<grid, FP8_BLOCK, 0, stream>>>(
        reinterpret_cast<__nv_fp8_e4m3*>(out.data),
        tb_data<__half>(in),
        n, scale);
}

void scaled_fp8_dequant(TrueBuffer& out, const TrueBuffer& in, float inv_scale,
                        cudaStream_t stream)
{
    int64_t n = tb_numel(in);
    int grid  = static_cast<int>((n + FP8_BLOCK - 1) / FP8_BLOCK);
    scaled_fp8_dequant_kernel<<<grid, FP8_BLOCK, 0, stream>>>(
        tb_data<__half>(out),
        reinterpret_cast<const __nv_fp8_e4m3*>(in.data),
        n, inv_scale);
}

void per_token_fp8_quant(TrueBuffer& out, TrueBuffer& scale_out,
                         const TrueBuffer& in, cudaStream_t stream)
{
    int num_tokens  = static_cast<int>(tb_size(in, 0));
    int hidden_size = static_cast<int>(tb_size(in, 1));
    per_token_fp8_quant_kernel<<<num_tokens, FP8_BLOCK, 0, stream>>>(
        reinterpret_cast<__nv_fp8_e4m3*>(out.data),
        tb_data<float>(scale_out),
        tb_data<__half>(in),
        num_tokens, hidden_size);
}

#else

// ---------------------------------------------------------------------------
// Stubs for non-sm_89+ builds — never called; prevent linker errors.
// ---------------------------------------------------------------------------
void scaled_fp8_quant(TrueBuffer&, const TrueBuffer&, float, cudaStream_t) {}
void scaled_fp8_dequant(TrueBuffer&, const TrueBuffer&, float, cudaStream_t) {}
void per_token_fp8_quant(TrueBuffer&, TrueBuffer&, const TrueBuffer&, cudaStream_t) {}

#endif // TRUELLM_CUDA_SM89_PLUS
