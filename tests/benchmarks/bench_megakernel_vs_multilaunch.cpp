// ---------------------------------------------------------------------------
// bench_megakernel_vs_multilaunch.cpp
//
// Compares the cooperative megakernel forward pass against the legacy
// multi-launch ggml+vllm path on identical sequences.
//
// Build: cmake -DTRUELLM_BUILD_BENCHMARKS=ON -DTRUELLM_HAS_CUDA=ON ...
// Run:   ./build/bin/truellm_cuda_benchmarks \
//            --benchmark_filter=BM_MegakernelVsMultiLaunch
//
// Environment:
//   TRUELLM_BENCH_MODEL — path to a GGUF model (required, or benchmarks skip)
//   TRUELLM_BENCH_GPU   — CUDA device id (default: 0)
// ---------------------------------------------------------------------------

#include <benchmark/benchmark.h>

#ifdef TRUELLM_HAS_CUDA

#include "backends/cuda/cuda_engine.h"
#include <truellm/config_schema.h>
#include <truellm/types.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <string>
#include <vector>

static const char* model_path()
{
    const char* p = std::getenv("TRUELLM_BENCH_MODEL");
    return p ? p : "";
}

static int bench_gpu()
{
    const char* p = std::getenv("TRUELLM_BENCH_GPU");
    return p ? std::atoi(p) : 0;
}

static truellm::InferenceConfig make_cfg(bool use_megakernel)
{
    truellm::InferenceConfig c;
    c.context_size              = 512;
    c.backend                   = "cuda";
    c.cuda_engine.use_megakernel = use_megakernel;
    c.cuda_engine.enforce_eager = true;
    c.cuda_engine.paged_kv_block_size = 16;
    return c;
}

static truellm::HardwareConfig make_hw()
{
    truellm::HardwareConfig h;
    h.gpu.device_id = bench_gpu();
    return h;
}

// Shared engine instances — lazily initialised so model load cost is not
// included in per-iteration timing.
static truellm::CudaEngine* g_legacy_engine = nullptr;
static truellm::CudaEngine* g_mk_engine     = nullptr;

static void ensure_engines()
{
    if (g_legacy_engine) return;
    const char* mp = model_path();
    if (!mp || mp[0] == '\0') return;

    truellm::SamplingConfig sc;
    sc.temperature = 0.0f;
    sc.seed        = 42;

    g_legacy_engine = new truellm::CudaEngine(make_cfg(false), make_hw(), sc);
    g_legacy_engine->load_model(mp);

    g_mk_engine = new truellm::CudaEngine(make_cfg(true), make_hw(), sc);
    g_mk_engine->load_model(mp);
}

static void BM_Legacy_MultiLaunch(benchmark::State& state)
{
    ensure_engines();
    if (!g_legacy_engine || !g_legacy_engine->is_model_loaded()) {
        state.SkipWithMessage("No model loaded (set TRUELLM_BENCH_MODEL)");
        return;
    }
    const int prompt_len = static_cast<int>(state.range(0));
    std::vector<int32_t> toks(prompt_len, 42);

    truellm::GenerateRequest req;
    req.tokens     = toks;
    req.max_tokens = 1;

    for (auto _ : state) {
        auto r = g_legacy_engine->generate(req);
        benchmark::DoNotOptimize(r);
        cudaDeviceSynchronize();
    }
    state.SetItemsProcessed(state.iterations() * prompt_len);
    state.SetLabel("legacy ggml+vllm");
}

static void BM_Megakernel_CoopLaunch(benchmark::State& state)
{
    ensure_engines();
    if (!g_mk_engine || !g_mk_engine->is_model_loaded()) {
        state.SkipWithMessage("No model loaded (set TRUELLM_BENCH_MODEL)");
        return;
    }
    const int prompt_len = static_cast<int>(state.range(0));
    std::vector<int32_t> toks(prompt_len, 42);

    truellm::GenerateRequest req;
    req.tokens     = toks;
    req.max_tokens = 1;

    for (auto _ : state) {
        auto r = g_mk_engine->generate(req);
        benchmark::DoNotOptimize(r);
        cudaDeviceSynchronize();
    }
    state.SetItemsProcessed(state.iterations() * prompt_len);
    state.SetLabel("cooperative megakernel");
}

BENCHMARK(BM_Legacy_MultiLaunch)
    ->Arg(1)->Arg(8)->Arg(32)->Arg(128)->Arg(512)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_Megakernel_CoopLaunch)
    ->Arg(1)->Arg(8)->Arg(32)->Arg(128)->Arg(512)
    ->Unit(benchmark::kMicrosecond);

#else

// Stub so the translation unit is not empty when CUDA is absent.
static void BM_NoCuda(benchmark::State& state)
{
    state.SkipWithMessage("Built without CUDA");
}
BENCHMARK(BM_NoCuda);

#endif // TRUELLM_HAS_CUDA
