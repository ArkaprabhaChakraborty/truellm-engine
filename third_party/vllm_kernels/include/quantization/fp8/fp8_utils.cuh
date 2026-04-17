#pragma once
// ---------------------------------------------------------------------------
// fp8_utils.cuh — FP8 E4M3 / E5M2 type helpers (Ada Lovelace sm_89+)
//
// cuda_fp8.h is included under TRUELLM_CUDA_SM89_PLUS (not __CUDA_ARCH__)
// because NVCC compiles kernels for every arch in the build list and the
// host-side cudafe pass also needs the type declarations.
//
// The device conversion helpers use only float arithmetic so they compile
// for any target architecture — no __CUDA_ARCH__ >= 890 guard needed.
// ---------------------------------------------------------------------------

#include <cuda_fp16.h>
#include <cstdint>

#if defined(TRUELLM_CUDA_SM89_PLUS)
#  include <cuda_fp8.h>   // __nv_fp8_e4m3 / __nv_fp8_e5m2

// Convenience aliases used throughout the kernel files.
using fp8_e4m3 = __nv_fp8_e4m3;
using fp8_e5m2 = __nv_fp8_e5m2;

// ---------------------------------------------------------------------------
// Scalar conversions — available for all device architectures.
// The fp8 types are defined in cuda_fp8.h and support construction from float
// on any architecture (hardware acceleration only on sm_89+).
// ---------------------------------------------------------------------------

__device__ __forceinline__ float fp8e4m3_to_float(fp8_e4m3 v) {
    return static_cast<float>(v);
}

__device__ __forceinline__ float fp8e5m2_to_float(fp8_e5m2 v) {
    return static_cast<float>(v);
}

__device__ __forceinline__ fp8_e4m3 float_to_fp8e4m3(float v, float scale = 1.0f) {
    return static_cast<fp8_e4m3>(v * scale);
}

__device__ __forceinline__ fp8_e5m2 float_to_fp8e5m2(float v, float scale = 1.0f) {
    return static_cast<fp8_e5m2>(v * scale);
}

// ---------------------------------------------------------------------------
// F16 ↔ FP8 bulk conversions (used by cache_kernels and fp8_gemm)
// ---------------------------------------------------------------------------

__device__ __forceinline__ fp8_e4m3 half_to_fp8e4m3(__half v, float scale = 1.0f) {
    return float_to_fp8e4m3(__half2float(v), scale);
}

__device__ __forceinline__ __half fp8e4m3_to_half(fp8_e4m3 v, float inv_scale = 1.0f) {
    return __float2half(fp8e4m3_to_float(v) * inv_scale);
}

#else // !TRUELLM_CUDA_SM89_PLUS
// Stub types for builds that don't include sm_89+ in the architecture list.
struct fp8_e4m3 { uint8_t bits; };
struct fp8_e5m2 { uint8_t bits; };
#endif // TRUELLM_CUDA_SM89_PLUS
