// ---------------------------------------------------------------------------
// bench_sync_overhead.cpp
//
// Measures the overhead of grid.sync() cooperative barriers vs cudaDeviceSynchronize
// and cudaStreamSynchronize at varying grid sizes.
//
// Tests the claim in cooperative_barrier.cuh that unconditional grid.sync() has
// lower overhead than re-launching a kernel per barrier.
//
// Build: cmake -DTRUELLM_BUILD_BENCHMARKS=ON -DTRUELLM_HAS_CUDA=ON ...
// Run:   ./build/bin/truellm_cuda_benchmarks --benchmark_filter=BM_Sync
// ---------------------------------------------------------------------------

#include <benchmark/benchmark.h>

#ifdef TRUELLM_HAS_CUDA

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstring>

namespace cg = cooperative_groups;

#define CUDA_CHECK(e) do { \
    cudaError_t _e = (e); \
    if (_e != cudaSuccess) \
        throw std::runtime_error(cudaGetErrorString(_e)); \
} while(0)

// ---------------------------------------------------------------------------
// Kernel: N cooperative grid barriers with no-op work between them.
// ---------------------------------------------------------------------------
__global__ void kernel_grid_sync_noop(int n_barriers, volatile int* out)
{
    cg::grid_group g = cg::this_grid();
    for (int i = 0; i < n_barriers; ++i) {
        g.sync();
        if (threadIdx.x == 0 && blockIdx.x == 0)
            *out = i;
    }
}

// ---------------------------------------------------------------------------
// Kernel: N separate dummy launches (no cooperative groups).
// Used to measure baseline launch overhead when doing N re-launches.
// ---------------------------------------------------------------------------
__global__ void kernel_dummy(volatile int* out, int val)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = val;
}

static int g_num_sms = 0;
static int* g_dev_out = nullptr;

static void setup_bench_device()
{
    if (g_dev_out) return;
    CUDA_CHECK(cudaDeviceGetAttribute(&g_num_sms,
               cudaDevAttrMultiProcessorCount, 0));
    CUDA_CHECK(cudaMalloc(&g_dev_out, sizeof(int)));
}

// ---------------------------------------------------------------------------
// BM_GridSync_NBarriers: cooperative grid.sync() N times per launch.
// state.range(0) = number of barriers per kernel launch
// state.range(1) = number of thread blocks (factor of num_sms)
// ---------------------------------------------------------------------------
static void BM_GridSync_NBarriers(benchmark::State& state)
{
    setup_bench_device();
    const int n_barriers = static_cast<int>(state.range(0));
    const int n_blocks   = static_cast<int>(state.range(1)) * g_num_sms;
    const int n_threads  = 128;

    // Verify cooperative launch is supported.
    int supports_coop = 0;
    cudaDeviceGetAttribute(&supports_coop,
                           cudaDevAttrCooperativeLaunch, 0);
    if (!supports_coop) {
        state.SkipWithMessage("Cooperative launch not supported on this device");
        return;
    }

    void* args[] = { &n_barriers, &g_dev_out };
    for (auto _ : state) {
        CUDA_CHECK(cudaLaunchCooperativeKernel(
            (const void*)kernel_grid_sync_noop,
            dim3(n_blocks), dim3(n_threads),
            args, 0, nullptr));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    state.SetLabel(std::to_string(n_blocks) + " blocks");
    state.counters["barriers_per_launch"] = n_barriers;
}

// ---------------------------------------------------------------------------
// BM_MultiLaunch_NKernels: N separate kernel launches + cudaDeviceSynchronize.
// Equivalent work to N cooperative barriers but using separate launches.
// state.range(0) = number of launches (= barriers to compare against)
// state.range(1) = number of thread blocks (factor of num_sms)
// ---------------------------------------------------------------------------
static void BM_MultiLaunch_NKernels(benchmark::State& state)
{
    setup_bench_device();
    const int n_launches = static_cast<int>(state.range(0));
    const int n_blocks   = static_cast<int>(state.range(1)) * g_num_sms;
    const int n_threads  = 128;

    for (auto _ : state) {
        for (int i = 0; i < n_launches; ++i) {
            int val = i;
            kernel_dummy<<<n_blocks, n_threads>>>(g_dev_out, val);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    state.SetLabel(std::to_string(n_blocks) + " blocks");
    state.counters["launches"] = n_launches;
}

// barriers = {4, 16, 32}, blocks_per_sm = {1, 2}
BENCHMARK(BM_GridSync_NBarriers)
    ->Args({4,  1})->Args({4,  2})
    ->Args({16, 1})->Args({16, 2})
    ->Args({32, 1})->Args({32, 2})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_MultiLaunch_NKernels)
    ->Args({4,  1})->Args({4,  2})
    ->Args({16, 1})->Args({16, 2})
    ->Args({32, 1})->Args({32, 2})
    ->Unit(benchmark::kMicrosecond);

#else

static void BM_NoCuda_Sync(benchmark::State& state)
{
    state.SkipWithMessage("Built without CUDA");
}
BENCHMARK(BM_NoCuda_Sync);

#endif // TRUELLM_HAS_CUDA
