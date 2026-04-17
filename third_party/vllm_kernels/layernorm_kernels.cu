// ---------------------------------------------------------------------------
// layernorm_kernels.cu — RMSNorm and LayerNorm kernels
//
// Vendored from vllm/csrc/layernorm_kernels.cu.
// Modifications: torch::Tensor replaced with TrueBuffer; pybind11 removed.
//
// Kernels:
//   rms_norm              — RMSNorm for Llama / Mistral / Qwen family
//   fused_add_rms_norm    — residual add + RMSNorm in one pass
//   layer_norm            — standard LayerNorm (GPT-2 / BERT family)
// ---------------------------------------------------------------------------

#include "include/vllm_types.h"
#include "include/dtype_float16.cuh"
#include "include/reduce_kernel_utils.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ---------------------------------------------------------------------------
// RMSNorm kernel
//
// out[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i]
//
// One block per row (token).  Threads cover the hidden dimension.
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void rms_norm_kernel(
    scalar_t* __restrict__ out,           // [num_tokens, hidden_size]
    const scalar_t* __restrict__ x,       // [num_tokens, hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    int hidden_size,
    float eps)
{
    extern __shared__ float smem[];  // [blockDim.x / 32] for warp partials

    int token_idx = blockIdx.x;
    const scalar_t* row_x   = x   + token_idx * hidden_size;
    scalar_t*       row_out = out + token_idx * hidden_size;

    // Compute sum of squares across the hidden dimension
    float sum_sq = 0.f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = to_float(row_x[i]);
        sum_sq += v * v;
    }
    sum_sq = block_reduce_sum(sum_sq, smem);
    __syncthreads();

    float rms_inv = rsqrtf(sum_sq / static_cast<float>(hidden_size) + eps);

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        row_out[i] = from_float<scalar_t>(
            to_float(row_x[i]) * rms_inv * to_float(weight[i]));
    }
}

// ---------------------------------------------------------------------------
// Fused residual + RMSNorm kernel
//
// residual[i] += x[i]
// out[i] = residual[i] / sqrt(mean(residual^2) + eps) * weight[i]
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void fused_add_rms_norm_kernel(
    scalar_t* __restrict__ out,           // [num_tokens, hidden_size]
    scalar_t* __restrict__ residual,      // [num_tokens, hidden_size]  (in-place updated)
    const scalar_t* __restrict__ x,       // [num_tokens, hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    int hidden_size,
    float eps)
{
    extern __shared__ float smem[];

    int token_idx      = blockIdx.x;
    scalar_t* row_res  = residual + token_idx * hidden_size;
    scalar_t* row_out  = out      + token_idx * hidden_size;
    const scalar_t* row_x = x    + token_idx * hidden_size;

    float sum_sq = 0.f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float r = to_float(row_res[i]) + to_float(row_x[i]);
        row_res[i] = from_float<scalar_t>(r);   // update residual in-place
        sum_sq += r * r;
    }
    sum_sq = block_reduce_sum(sum_sq, smem);
    __syncthreads();

    float rms_inv = rsqrtf(sum_sq / static_cast<float>(hidden_size) + eps);

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        row_out[i] = from_float<scalar_t>(
            to_float(row_res[i]) * rms_inv * to_float(weight[i]));
    }
}

// ---------------------------------------------------------------------------
// LayerNorm kernel (mean-variance normalisation, for GPT-2 / BERT style)
//
// out[i] = (x[i] - mean) / sqrt(var + eps) * weight[i] + bias[i]
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void layer_norm_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    int hidden_size,
    float eps)
{
    extern __shared__ float smem[];

    int token_idx       = blockIdx.x;
    const scalar_t* row = x   + token_idx * hidden_size;
    scalar_t* row_out   = out + token_idx * hidden_size;

    // Mean
    float sum = 0.f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x)
        sum += to_float(row[i]);
    sum = block_reduce_sum(sum, smem);
    __syncthreads();
    float mean = sum / static_cast<float>(hidden_size);

    // Variance
    float var = 0.f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float d = to_float(row[i]) - mean;
        var += d * d;
    }
    var = block_reduce_sum(var, smem);
    __syncthreads();
    float inv_std = rsqrtf(var / static_cast<float>(hidden_size) + eps);

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float norm = (to_float(row[i]) - mean) * inv_std;
        row_out[i] = from_float<scalar_t>(
            norm * to_float(weight[i]) + to_float(bias[i]));
    }
}

// ---------------------------------------------------------------------------
// Host-side launch helpers
// ---------------------------------------------------------------------------

static constexpr int NORM_THREADS = 256;

static int smem_bytes(int block_threads) {
    return (block_threads / 32) * static_cast<int>(sizeof(float));
}

void rms_norm(TrueBuffer& out, const TrueBuffer& x, const TrueBuffer& weight,
              float eps, cudaStream_t stream)
{
    int num_tokens  = static_cast<int>(tb_size(x, 0));
    int hidden_size = static_cast<int>(tb_size(x, 1));

    if (x.dtype == DataType::F16) {
        rms_norm_kernel<__half><<<num_tokens, NORM_THREADS,
                                  smem_bytes(NORM_THREADS), stream>>>(
            tb_data<__half>(out), tb_data<__half>(x),
            tb_data<__half>(weight), hidden_size, eps);
    } else {
        rms_norm_kernel<float><<<num_tokens, NORM_THREADS,
                                 smem_bytes(NORM_THREADS), stream>>>(
            tb_data<float>(out), tb_data<float>(x),
            tb_data<float>(weight), hidden_size, eps);
    }
}

void fused_add_rms_norm(TrueBuffer& out, TrueBuffer& residual,
                        const TrueBuffer& x, const TrueBuffer& weight,
                        float eps, cudaStream_t stream)
{
    int num_tokens  = static_cast<int>(tb_size(x, 0));
    int hidden_size = static_cast<int>(tb_size(x, 1));

    if (x.dtype == DataType::F16) {
        fused_add_rms_norm_kernel<__half><<<num_tokens, NORM_THREADS,
                                            smem_bytes(NORM_THREADS), stream>>>(
            tb_data<__half>(out), tb_data<__half>(residual),
            tb_data<__half>(x),   tb_data<__half>(weight),
            hidden_size, eps);
    } else {
        fused_add_rms_norm_kernel<float><<<num_tokens, NORM_THREADS,
                                           smem_bytes(NORM_THREADS), stream>>>(
            tb_data<float>(out), tb_data<float>(residual),
            tb_data<float>(x),   tb_data<float>(weight),
            hidden_size, eps);
    }
}

void layer_norm(TrueBuffer& out, const TrueBuffer& x,
                const TrueBuffer& weight, const TrueBuffer& bias,
                float eps, cudaStream_t stream)
{
    int num_tokens  = static_cast<int>(tb_size(x, 0));
    int hidden_size = static_cast<int>(tb_size(x, 1));

    if (x.dtype == DataType::F16) {
        layer_norm_kernel<__half><<<num_tokens, NORM_THREADS,
                                    smem_bytes(NORM_THREADS), stream>>>(
            tb_data<__half>(out), tb_data<__half>(x),
            tb_data<__half>(weight), tb_data<__half>(bias),
            hidden_size, eps);
    } else {
        layer_norm_kernel<float><<<num_tokens, NORM_THREADS,
                                   smem_bytes(NORM_THREADS), stream>>>(
            tb_data<float>(out), tb_data<float>(x),
            tb_data<float>(weight), tb_data<float>(bias),
            hidden_size, eps);
    }
}
