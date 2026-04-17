#pragma once
// ---------------------------------------------------------------------------
// vllm_types.h — TrueBuffer shim replacing torch::Tensor / at::Tensor
//
// The vendored .cu kernel files were originally written against PyTorch's
// C++ API.  This header provides drop-in replacements for the small surface
// of that API that the kernels actually use, backed by truellm::TrueBuffer.
//
// Rules:
//   - No #include <torch/...> or <ATen/...> anywhere in this tree.
//   - No link to libtorch or libc10.
//   - Every accessor that was runtime-dispatched in PyTorch is resolved at
//     template-instantiation time here.
// ---------------------------------------------------------------------------

#include <truellm/types.h>

#include <cassert>
#include <cstdint>
#include <cstddef>
#include <type_traits>

// ---------------------------------------------------------------------------
// TORCH_CHECK / AT_DISPATCH replacements
// ---------------------------------------------------------------------------

// In vLLM kernels, TORCH_CHECK is used for precondition assertions.
// Map it to a TrueLLM-flavoured assert so it compiles without libtorch.
#ifndef TORCH_CHECK
#  define TORCH_CHECK(cond, ...)  assert((cond))
#endif

// c10::Error is only thrown by TORCH_CHECK — never actually caught in .cu
// files, so a minimal stub is enough.
namespace c10 {
struct Error {
    const char* msg;
    explicit Error(const char* m) : msg(m) {}
};
} // namespace c10

// ---------------------------------------------------------------------------
// Minimal c10::cuda::CUDAStream stub
// The vLLM kernels pass streams to CUDA runtime calls; in some signatures
// they use c10::cuda::CUDAStream rather than raw cudaStream_t.
// ---------------------------------------------------------------------------
#if defined(__CUDACC__) || defined(TRUELLM_HAS_CUDA)
#  include <cuda_runtime.h>
#  include <cuda_fp16.h>    // __half type (not always pulled in by cuda_runtime.h)
namespace c10 {
namespace cuda {
struct CUDAStream {
    cudaStream_t stream;
    explicit CUDAStream(cudaStream_t s = nullptr) : stream(s) {}
    cudaStream_t stream_() const { return stream; }
    // Implicit conversion so callers can pass CUDAStream where cudaStream_t is expected
    operator cudaStream_t() const { return stream; }
};
inline CUDAStream getCurrentCUDAStream(int /*device_id*/ = 0) {
    cudaStream_t s = nullptr;
    return CUDAStream(s);
}
} // namespace cuda
} // namespace c10
#endif // __CUDACC__

// ---------------------------------------------------------------------------
// scalar_t helpers — map truellm::DataType to C++ types
// ---------------------------------------------------------------------------
namespace truellm {
namespace vk { // vk = vllm_kernels namespace

// Template that maps a DataType enum value to a C++ scalar type.
// Specialisations below cover the types the kernels actually instantiate.
template <DataType DT> struct scalar_type_of;
template <> struct scalar_type_of<DataType::F32>  { using type = float;       };
template <> struct scalar_type_of<DataType::F16>  { using type = __half;      };

#if defined(__CUDA_ARCH__) && defined(__CUDACC__)
// BF16 and FP8 only available in device code / NVCC
template <> struct scalar_type_of<DataType::BF16> { using type = __nv_bfloat16; };
#endif

// ---------------------------------------------------------------------------
// TrueBuffer accessor helpers
// These replace the .data_ptr<T>(), .sizes(), .strides() calls.
// ---------------------------------------------------------------------------

// Return a typed device pointer from a TrueBuffer.
template <typename T>
inline T* tb_data(const TrueBuffer& buf) {
    return reinterpret_cast<T*>(buf.data);
}

// Shape accessor — replaces tensor.size(dim).
inline int64_t tb_size(const TrueBuffer& buf, int dim) {
    assert(dim >= 0 && dim < buf.ndim);
    return buf.shape[dim];
}

// Stride accessor — replaces tensor.stride(dim).
inline int64_t tb_stride(const TrueBuffer& buf, int dim) {
    assert(dim >= 0 && dim < buf.ndim);
    return buf.stride[dim];
}

// Total number of elements.
inline int64_t tb_numel(const TrueBuffer& buf) {
    int64_t n = 1;
    for (int i = 0; i < buf.ndim; ++i) n *= buf.shape[i];
    return n;
}

// True when the buffer lives on a CUDA device.
inline bool tb_is_cuda(const TrueBuffer& buf) {
    return buf.device_id >= 0;
}

// Size in bytes of one element given a DataType.
inline size_t tb_element_size(DataType dt) {
    switch (dt) {
        case DataType::F32:  return 4;
        case DataType::F16:  return 2;
        case DataType::BF16: return 2;
        case DataType::I32:  return 4;
        case DataType::I16:  return 2;
        case DataType::I8:   return 1;
        default:             return 1;
    }
}

// Construct a 4-D TrueBuffer view (no allocation — data pointer is shared).
inline TrueBuffer tb_view4d(void* data, DataType dtype, int device_id,
                             int64_t d0, int64_t d1, int64_t d2, int64_t d3)
{
    TrueBuffer b{};
    b.data      = data;
    b.dtype     = dtype;
    b.device_id = device_id;
    b.ndim      = 4;
    b.shape[0]  = d0; b.shape[1] = d1; b.shape[2] = d2; b.shape[3] = d3;
    // Row-major contiguous strides
    b.stride[3] = 1;
    b.stride[2] = d3;
    b.stride[1] = d2 * d3;
    b.stride[0] = d1 * d2 * d3;
    b.size_bytes = static_cast<size_t>(d0 * d1 * d2 * d3) * tb_element_size(dtype);
    return b;
}

inline TrueBuffer tb_view3d(void* data, DataType dtype, int device_id,
                             int64_t d0, int64_t d1, int64_t d2)
{
    return tb_view4d(data, dtype, device_id, d0, d1, d2, 1);
}

inline TrueBuffer tb_view2d(void* data, DataType dtype, int device_id,
                             int64_t d0, int64_t d1)
{
    return tb_view4d(data, dtype, device_id, d0, d1, 1, 1);
}

inline TrueBuffer tb_view1d(void* data, DataType dtype, int device_id, int64_t d0)
{
    return tb_view4d(data, dtype, device_id, d0, 1, 1, 1);
}

} // namespace vk
} // namespace truellm

// Bring helpers into global scope so .cu files that use them without
// a namespace qualifier compile without modification.
using truellm::vk::tb_data;
using truellm::vk::tb_size;
using truellm::vk::tb_stride;
using truellm::vk::tb_numel;
using truellm::vk::tb_is_cuda;
using truellm::TrueBuffer;
using truellm::DataType;
