#pragma once
// ---------------------------------------------------------------------------
// dtype_float16.cuh — F16 / BF16 arithmetic helpers
//
// Provides consistent __half and __nv_bfloat16 arithmetic on Turing (sm_75)
// and later.  On sm_75 some __half2 intrinsics require explicit includes.
// ---------------------------------------------------------------------------

#include <cuda_fp16.h>
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#  include <cuda_bf16.h>
#endif

// ---------------------------------------------------------------------------
// Generic to-float and from-float conversions so kernels can be templated
// on scalar_t without specialising every arithmetic op.
// ---------------------------------------------------------------------------

template <typename T> __device__ __forceinline__ float to_float(T v);
template <typename T> __device__ __forceinline__ T from_float(float v);

template <> __device__ __forceinline__ float to_float<float>(float v)  { return v; }
template <> __device__ __forceinline__ float to_float<__half>(__half v){ return __half2float(v); }
template <> __device__ __forceinline__ float from_float<float>(float v){ return v; }
template <> __device__ __forceinline__ __half from_float<__half>(float v){ return __float2half(v); }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template <> __device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v){
    return __bfloat162float(v);
}
template <> __device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v){
    return __float2bfloat16(v);
}
#endif

// ---------------------------------------------------------------------------
// Safe fmaxf / fminf wrappers on device
// ---------------------------------------------------------------------------
__device__ __forceinline__ float device_fmax(float a, float b) { return fmaxf(a, b); }
__device__ __forceinline__ float device_fmin(float a, float b) { return fminf(a, b); }
