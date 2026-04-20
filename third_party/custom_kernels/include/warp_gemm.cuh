#pragma once
// ---------------------------------------------------------------------------
// warp_gemm.cuh — Register-tiled F16 GEMM for use inside the megakernel.
//
// Two paths selected at compile time:
//   SM_80+ path: WMMA 16×16×16 tiles via mma.h (Ampere+).
//   SM_70 path:  Scalar F16 loop via __half arithmetic (Turing fallback).
//
// All weight loads use __ldg() (read-only cache) for weight-stationary access.
// __syncwarp() after every shared memory write to avoid warp-divergent reads.
//
// Conventions:
//   A: [M, K] row-major   (activations — written to shared memory)
//   B: [N, K] row-major   (weights — read via __ldg; transposed layout)
//   C: [M, N] row-major   (output — accumulated in registers, written once)
//
// Single-token decode path: M = 1 (one activation row).
// Batch prefill path: M > 1 (WMMA handles multiple rows natively).
//
// Usage:
//   warp_gemm_f16(C, A, B, M, N, K, smem_workspace, smem_bytes)
//
// smem_workspace must be at least warp_gemm_smem_bytes(M, K, N) bytes.
// ---------------------------------------------------------------------------

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#  include <mma.h>
#  define TRUELLM_HAS_WMMA 1
#else
#  define TRUELLM_HAS_WMMA 0
#endif

// ---------------------------------------------------------------------------
// Helper: load a K-element row of A (F16) into shared memory.
// All threads in a warp cooperate. 'smem' must be at least K * sizeof(__half).
// ---------------------------------------------------------------------------
__device__ __forceinline__
void load_row_to_smem(const __half* __restrict__ src, __half* smem, int K)
{
    for (int k = threadIdx.x; k < K; k += warpSize)
        smem[k] = src[k];
    __syncwarp();
}

// ---------------------------------------------------------------------------
// warp_gemm_f16 — single (M=1) row GEMV or small GEMM.
//
// For M=1 this reduces to a GEMV: C[n] = sum_k A[k] * B[n*K + k]
// Compatible with both SM_70 and SM_80 paths.
//
// A:  [M, K] F16 row-major (M=1 for decode path)
// B:  [N, K] F16 row-major (weight matrix, transposed)
// C:  [M, N] F16 output
// smem: temporary shared buffer of at least K * sizeof(__half) bytes
// ---------------------------------------------------------------------------
__device__ __forceinline__
void warp_gemm_f16(
    __half* __restrict__       C,
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    int M, int N, int K,
    __half* smem)
{
#if TRUELLM_HAS_WMMA
    // WMMA path (sm_80+): use nvcuda::wmma for 16×16×16 tiles.
    // For M=1 and small M we fall through to the scalar path — WMMA tiles
    // require M to be a multiple of 16. The conditional selects the fast path.
    if (M >= 16 && N % 16 == 0 && K % 16 == 0) {
        using namespace nvcuda;
        // Tile across N in chunks of 16.
        for (int ni = 0; ni < N; ni += 16) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);
            for (int ki = 0; ki < K; ki += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
                wmma::load_matrix_sync(a_frag, A + ki,           K);
                wmma::load_matrix_sync(b_frag, B + ni * K + ki,  K);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            // Store accumulated result back to C as F16.
            // Use a shared buffer to convert float → half.
            __shared__ float tmp_store[16 * 16];
            wmma::store_matrix_sync(tmp_store, c_frag, 16, wmma::mem_row_major);
            for (int r = 0; r < 16 && r < M; ++r)
                for (int c = threadIdx.x; c < 16; c += warpSize)
                    C[r * N + ni + c] = __float2half(tmp_store[r * 16 + c]);
        }
        return;
    }
#endif

    // Scalar path (sm_70 fallback, or M < 16):
    // Each thread computes one output element C[row][col].
    // Threads are assigned columns; rows loop serially for small M.
    for (int row = 0; row < M; ++row) {
        // Load row of A into smem.
        load_row_to_smem(A + row * K, smem, K);
        // Each thread handles one or more columns of B.
        for (int col = threadIdx.x; col < N; col += warpSize) {
            float acc = 0.0f;
            const __half* b_row = B + col * K;
            for (int k = 0; k < K; ++k) {
                acc += __half2float(smem[k]) * __half2float(__ldg(&b_row[k]));
            }
            C[row * N + col] = __float2half(acc);
        }
        __syncwarp();
    }
}

// Return the shared memory needed for one warp_gemm_f16 call.
__host__ __device__ __forceinline__
size_t warp_gemm_smem_bytes(int /*M*/, int K, int /*N*/)
{
    return static_cast<size_t>(K) * sizeof(__half);
}
