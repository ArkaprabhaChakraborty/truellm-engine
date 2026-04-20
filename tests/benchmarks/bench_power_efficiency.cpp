// ---------------------------------------------------------------------------
// bench_power_efficiency.cpp
//
// Tokens-per-joule efficiency benchmark comparing the megakernel path against
// the legacy multi-launch path.
//
// Power is sampled via NVML (nvidia-smi library) at 100 ms intervals during
// the benchmark loop.  If NVML is unavailable the benchmark records throughput
// only (joule metrics are skipped).
//
// Build: cmake -DTRUELLM_BUILD_BENCHMARKS=ON -DTRUELLM_HAS_CUDA=ON ...
//   To enable NVML power sampling also pass -DTRUELLM_BENCH_NVML=ON.
// Run:   ./build/bin/truellm_cuda_benchmarks --benchmark_filter=BM_Power
// ---------------------------------------------------------------------------

#include <benchmark/benchmark.h>

#ifdef TRUELLM_HAS_CUDA

#include "backends/cuda/cuda_engine.h"
#include <truellm/config_schema.h>
#include <truellm/types.h>
#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <thread>

#ifdef TRUELLM_BENCH_NVML
#  include <nvml.h>
#  define NVML_AVAILABLE 1
#else
#  define NVML_AVAILABLE 0
#endif

// ---------------------------------------------------------------------------
// PowerSampler — background thread polling NVML at fixed interval.
// ---------------------------------------------------------------------------
struct PowerSampler {
    std::atomic<bool>   running{false};
    std::atomic<double> joules_accum{0.0};
    std::thread         worker;
    int                 interval_ms = 100;

#if NVML_AVAILABLE
    nvmlDevice_t device{};
    bool         nvml_ok = false;

    void start(int gpu_id)
    {
        nvmlInit();
        nvml_ok = (nvmlDeviceGetHandleByIndex(gpu_id, &device) == NVML_SUCCESS);
        if (!nvml_ok) return;
        running = true;
        joules_accum = 0.0;
        worker = std::thread([this](){
            auto last = std::chrono::steady_clock::now();
            while (running.load()) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(interval_ms));
                unsigned int mw = 0;
                if (nvmlDeviceGetPowerUsage(device, &mw) == NVML_SUCCESS) {
                    auto now = std::chrono::steady_clock::now();
                    double dt = std::chrono::duration<double>(now - last).count();
                    joules_accum = joules_accum.load()
                                 + (static_cast<double>(mw) / 1000.0) * dt;
                    last = now;
                }
            }
        });
    }

    void stop() {
        running = false;
        if (worker.joinable()) worker.join();
        nvmlShutdown();
    }

    double total_joules() const { return joules_accum.load(); }
#else
    void start(int) {}
    void stop()    {}
    double total_joules() const { return 0.0; }
#endif
};

static const char* model_path_power()
{
    const char* p = std::getenv("TRUELLM_BENCH_MODEL");
    return p ? p : "";
}

static truellm::CudaEngine* make_engine(bool use_megakernel)
{
    truellm::InferenceConfig inf;
    inf.context_size                   = 512;
    inf.backend                        = "cuda";
    inf.cuda_engine.use_megakernel     = use_megakernel;
    inf.cuda_engine.enforce_eager      = true;

    truellm::HardwareConfig hw;
    hw.gpu.device_id = 0;

    truellm::SamplingConfig sc;
    sc.temperature = 0.0f;
    sc.seed        = 1;

    auto* e = new truellm::CudaEngine(inf, hw, sc);
    e->load_model(model_path_power());
    return e;
}

static truellm::CudaEngine* g_legacy_pw  = nullptr;
static truellm::CudaEngine* g_mk_pw      = nullptr;

static void ensure_power_engines()
{
    if (g_legacy_pw) return;
    const char* mp = model_path_power();
    if (!mp || mp[0] == '\0') return;
    g_legacy_pw = make_engine(false);
    g_mk_pw     = make_engine(true);
}

static void run_power_bench(benchmark::State& state,
                             truellm::CudaEngine* engine,
                             bool use_nvml)
{
    if (!engine || !engine->is_model_loaded()) {
        state.SkipWithMessage("No model loaded (set TRUELLM_BENCH_MODEL)");
        return;
    }
    const int prompt_len   = static_cast<int>(state.range(0));
    const int decode_steps = static_cast<int>(state.range(1));

    std::vector<int32_t> toks(prompt_len, 42);
    truellm::GenerateRequest req;
    req.tokens     = toks;
    req.max_tokens = decode_steps;

    PowerSampler ps;
    int64_t total_tokens = 0;

    for (auto _ : state) {
        if (use_nvml) ps.start(0);
        auto r = engine->generate(req);
        cudaDeviceSynchronize();
        if (use_nvml) ps.stop();
        total_tokens += prompt_len + static_cast<int64_t>(r.generated_tokens);
        benchmark::DoNotOptimize(r);
    }

    state.SetItemsProcessed(total_tokens);
    if (use_nvml && ps.total_joules() > 0.0) {
        state.counters["tokens_per_joule"] =
            static_cast<double>(total_tokens) / ps.total_joules();
        state.counters["total_joules"] = ps.total_joules();
    }
}

static void BM_Power_Legacy(benchmark::State& state)
{
    ensure_power_engines();
    run_power_bench(state, g_legacy_pw, NVML_AVAILABLE);
}

static void BM_Power_Megakernel(benchmark::State& state)
{
    ensure_power_engines();
    run_power_bench(state, g_mk_pw, NVML_AVAILABLE);
}

// prompt_len × decode_steps
BENCHMARK(BM_Power_Legacy)
    ->Args({32,  16})->Args({128, 16})->Args({512, 16})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Power_Megakernel)
    ->Args({32,  16})->Args({128, 16})->Args({512, 16})
    ->Unit(benchmark::kMillisecond);

#else

static void BM_NoCuda_Power(benchmark::State& state)
{
    state.SkipWithMessage("Built without CUDA");
}
BENCHMARK(BM_NoCuda_Power);

#endif // TRUELLM_HAS_CUDA
