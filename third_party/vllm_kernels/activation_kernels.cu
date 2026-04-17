// ---------------------------------------------------------------------------
// activation_kernels.cu — Fused activation functions
//
// Vendored from vllm/csrc/activation_kernels.cu.
// Modifications: torch::Tensor replaced with TrueBuffer; pybind11 removed.
//
// Kernels:
//   silu_and_mul        — SiLU(gate) * up  (SwiGLU / Llama FFN)
//   gelu_and_mul        — GELU(gate) * up  (Phi-3 / Falcon-H1 FFN)
//   gelu_tanh_and_mul   — GELU-tanh approx * up (GPT-2 style)
// ---------------------------------------------------------------------------

#include "include/vllm_types.h"
#include "include/dtype_float16.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ---------------------------------------------------------------------------
// Device activation functions
// ---------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ T silu(T x) {
    float xf = to_float(x);
    return from_float<T>(xf / (1.f + expf(-xf)));
}

template <typename T>
__device__ __forceinline__ T gelu_erf(T x) {
    // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    float xf = to_float(x);
    return from_float<T>(0.5f * xf * (1.f + erff(xf * 0.70710678118f)));
}

template <typename T>
__device__ __forceinline__ T gelu_tanh(T x) {
    // GELU-tanh approximation used by GPT-2 / BERT
    // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    float xf = to_float(x);
    float c  = 0.79788456f * (xf + 0.044715f * xf * xf * xf);
    return from_float<T>(0.5f * xf * (1.f + tanhf(c)));
}

// ---------------------------------------------------------------------------
// Kernel: element-wise gate(activation) * up
//
// Input layout: out[i] = act(gate[i]) * up[i]
//   out  : [num_tokens, d]
//   gate : [num_tokens, d]  (first half of the fused projection)
//   up   : [num_tokens, d]  (second half of the fused projection)
//
// In practice, many models store gate and up concatenated in a single
// [num_tokens, 2*d] tensor.  The caller slices before calling here.
// ---------------------------------------------------------------------------

template <typename scalar_t, typename ActFn>
__global__ void act_and_mul_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ gate,
    const scalar_t* __restrict__ up,
    int64_t n_elements)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        ActFn fn;
        out[idx] = from_float<scalar_t>(
            to_float(fn(gate[idx])) * to_float(up[idx]));
    }
}

// Functor wrappers for template dispatch
template <typename T> struct SiluFn   { __device__ T operator()(T x) const { return silu(x); }   };
template <typename T> struct GeluFn   { __device__ T operator()(T x) const { return gelu_erf(x); } };
template <typename T> struct GeluTanhFn { __device__ T operator()(T x) const { return gelu_tanh(x); } };

// ---------------------------------------------------------------------------
// Host-side launch wrappers
// ---------------------------------------------------------------------------

static constexpr int BLOCK_SIZE_ACT = 256;

template <typename scalar_t, template <typename> class ActFn>
static void launch_act_and_mul(
    TrueBuffer& out,
    const TrueBuffer& gate,
    const TrueBuffer& up,
    cudaStream_t stream)
{
    int64_t n = tb_numel(gate);
    int grid  = static_cast<int>((n + BLOCK_SIZE_ACT - 1) / BLOCK_SIZE_ACT);
    act_and_mul_kernel<scalar_t, ActFn<scalar_t>>
        <<<grid, BLOCK_SIZE_ACT, 0, stream>>>(
            tb_data<scalar_t>(out),
            tb_data<scalar_t>(gate),
            tb_data<scalar_t>(up),
            n);
}

// ---------------------------------------------------------------------------
// Public API (called by cuda_engine.cpp)
// ---------------------------------------------------------------------------

void silu_and_mul(TrueBuffer& out, const TrueBuffer& gate, const TrueBuffer& up,
                  cudaStream_t stream)
{
    if (gate.dtype == DataType::F16) {
        launch_act_and_mul<__half, SiluFn>(out, gate, up, stream);
    } else {
        launch_act_and_mul<float, SiluFn>(out, gate, up, stream);
    }
}

void gelu_and_mul(TrueBuffer& out, const TrueBuffer& gate, const TrueBuffer& up,
                  cudaStream_t stream)
{
    if (gate.dtype == DataType::F16) {
        launch_act_and_mul<__half, GeluFn>(out, gate, up, stream);
    } else {
        launch_act_and_mul<float, GeluFn>(out, gate, up, stream);
    }
}

void gelu_tanh_and_mul(TrueBuffer& out, const TrueBuffer& gate, const TrueBuffer& up,
                       cudaStream_t stream)
{
    if (gate.dtype == DataType::F16) {
        launch_act_and_mul<__half, GeluTanhFn>(out, gate, up, stream);
    } else {
        launch_act_and_mul<float, GeluTanhFn>(out, gate, up, stream);
    }
}
