// ---------------------------------------------------------------------------
// bench_smem_occupancy.cpp
//
// Measures the effect of shared memory allocation on SM occupancy and
// throughput for the megakernel's primary compute patterns:
//
//   BM_WarpGemmSmem  — warp_gemm_f16() throughput at varying K (smem usage).
//   BM_SmemOccupancy — kernel occupancy as smem per block increases.
//
// The goal is to confirm that the S_TILE=8 choice in megakernel.cu keeps
// smem usage below the per-SM occupancy threshold on both sm_70 and sm_80+.
//
// Build: cmake -DTRUELLM_BUILD_BENCHMARKS=ON -DTRUELLM_HAS_CUDA=ON ...
// Run:   ./build/bin/truellm_cuda_benchmarks --benchmark_filter=BM_Smem
// ---------------------------------------------------------------------------

#include <benchmark/benchmark.h>

#ifdef TRUELLM_HAS_CUDA

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <vector>
#include <cstring>

// Pull in the warp_gemm helper directly so we can measure it in isolation.
// The include uses __device__ functions, so we must compile this .cpp as a
// .cu file if using nvcc. When using clang-cuda it compiles as-is.
// The CMakeLists renames this file to .cu in the CUDA benchmark target.
#include "../../../../third_party/custom_kernels/include/warp_gemm.cuh"

#define CUDA_CHECK(e) do { \
    cudaError_t _e = (e); \
    if (_e != cudaSuccess) \
        throw std::runtime_error(cudaGetErrorString(_e)); \
} while(0)

// ---------------------------------------------------------------------------
// Kernel: invoke warp_gemm_f16 N times per thread block.
// Measures raw GEMM throughput for a single (1 × N) × (N × K) matmul.
// ---------------------------------------------------------------------------
__global__ void kernel_warp_gemm_bench(__half* C, const __half* A, const __half* B,
                                        int N, int K, int iters,
                                        __half* smem_ws)
{
    for (int i = 0; i < iters; ++i)
        warp_gemm_f16(C, A, B, 1, N, K, smem_ws);
}

// ---------------------------------------------------------------------------
// Kernel: allocate smem_bytes of shared memory per block, do nothing.
// Used with cudaOccupancyMaxActiveBlocksPerMultiprocessor to measure how
// occupancy degrades as smem grows.
// ---------------------------------------------------------------------------
template <int SMEM_BYTES>
__global__ void kernel_smem_probe()
{
    extern __shared__ char smem[];
    if (threadIdx.x == 0) smem[0] = 1;
}

static void BM_WarpGemmSmem(benchmark::State& state)
{
    const int K = static_cast<int>(state.range(0)); // hidden dim
    const int N = 128; // output dim (fixed; comparable across K)

    __half *d_A = nullptr, *d_B = nullptr, *d_C = nullptr, *d_ws = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, static_cast<size_t>(K) * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_B, static_cast<size_t>(N * K) * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_C, static_cast<size_t>(N) * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_ws, static_cast<size_t>(K) * sizeof(__half)));
    CUDA_CHECK(cudaMemset(d_A, 0, K * sizeof(__half)));
    CUDA_CHECK(cudaMemset(d_B, 0, N * K * sizeof(__half)));

    const int iters = 100;
    for (auto _ : state) {
        kernel_warp_gemm_bench<<<1, 32>>>(d_C, d_A, d_B, N, K, iters, d_ws);
        CUDA_CHECK(cudaDeviceSynchronize());
        benchmark::DoNotOptimize(d_C);
    }
    state.SetItemsProcessed(
        state.iterations() * static_cast<int64_t>(iters) * 2 * N * K);
    state.SetBytesProcessed(
        state.iterations() * static_cast<int64_t>(iters) * N * K * sizeof(__half));

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_ws);
}

// Occupancy probe: measure active blocks/SM for various smem allocations.
static void BM_SmemOccupancy(benchmark::State& state)
{
    const int smem_kb = static_cast<int>(state.range(0));
    const int smem_bytes = smem_kb * 1024;
    const int threads = 128;

    // Use cudaOccupancyMaxActiveBlocksPerMultiprocessor to query occupancy.
    // We pick kernel_smem_probe<0> and pass smem_bytes as the dynamic smem arg.
    int max_blocks = 0;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks,
        (const void*)kernel_smem_probe<0>,
        threads,
        smem_bytes);
    if (err != cudaSuccess) {
        state.SkipWithMessage(cudaGetErrorString(err));
        return;
    }

    // Record occupancy as a benchmark counter (not time-based).
    for (auto _ : state) {
        benchmark::DoNotOptimize(max_blocks);
    }
    state.counters["max_blocks_per_sm"] = max_blocks;
    state.counters["smem_kb"]           = smem_kb;
}

// K = {512, 1024, 2048, 4096}: typical hidden dims for 7B–70B models.
BENCHMARK(BM_WarpGemmSmem)
    ->Arg(512)->Arg(1024)->Arg(2048)->Arg(4096)
    ->Unit(benchmark::kMicrosecond);

// smem_kb from 4 KB to 96 KB (A100 maximum per block is 164 KB).
BENCHMARK(BM_SmemOccupancy)
    ->Arg(4)->Arg(8)->Arg(16)->Arg(32)->Arg(48)->Arg(64)->Arg(96)
    ->Unit(benchmark::kNanosecond);

#else

static void BM_NoCuda_Smem(benchmark::State& state)
{
    state.SkipWithMessage("Built without CUDA");
}
BENCHMARK(BM_NoCuda_Smem);

#endif // TRUELLM_HAS_CUDA
