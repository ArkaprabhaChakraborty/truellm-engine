// ---------------------------------------------------------------------------
// bench_throughput.cpp — Google Benchmark microbenchmarks for inference path
//
// Build: cmake -DTRUELLM_BUILD_BENCHMARKS=ON ...
// Run:   ./build/bin/truellm_benchmarks [--benchmark_filter=...]
//
// These benchmarks measure the CPU-path tokenize / generate loop latency
// without a live HTTP server.  For full end-to-end throughput testing use
// tests/integration/python/test_batch_concurrent.py against a running server.
// ---------------------------------------------------------------------------

#include <benchmark/benchmark.h>

#include "backends/cpu/ggml_engine.h"
#include <truellm/config_schema.h>
#include <truellm/types.h>

#include <cstdlib>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------------------
static truellm::InferenceConfig make_bench_inf()
{
    truellm::InferenceConfig c;
    c.context_size    = 512;
    c.batch_size      = 512;
    c.ubatch_size     = 512;
    c.threads         = 0;   // auto
    c.flash_attention = false;
    return c;
}

static truellm::HardwareConfig make_bench_hw()
{
    truellm::HardwareConfig c;
    c.offload.gpu_layers = 0;
    const char* gl = std::getenv("TRUELLM_TEST_GPU_LAYERS");
    if (gl) c.offload.gpu_layers = std::atoi(gl);
    return c;
}

static truellm::SamplingConfig make_bench_smp()
{
    truellm::SamplingConfig c;
    c.temperature = 0.0f;  // greedy for reproducibility
    c.top_p       = 1.0f;
    c.top_k       = 1;
    c.seed        = 42;
    return c;
}

// ---------------------------------------------------------------------------
// BM_TokenizeShortString
// Measures the tokenizer call overhead on a short string.
// Requires TRUELLM_TEST_MODEL_PATH to be set; skips otherwise.
// ---------------------------------------------------------------------------
static void BM_TokenizeShortString(benchmark::State& state)
{
    const char* model_path = std::getenv("TRUELLM_TEST_MODEL_PATH");
    if (!model_path) {
        state.SkipWithError("TRUELLM_TEST_MODEL_PATH not set");
        return;
    }

    truellm::GgmlEngine eng(make_bench_inf(), make_bench_hw(), make_bench_smp());
    if (eng.load_model(model_path) != truellm::ErrorCode::Ok) {
        state.SkipWithError("Model failed to load");
        return;
    }

    const std::string text = "The transformer architecture uses attention mechanisms.";
    for (auto _ : state) {
        auto tokens = eng.tokenize(text);
        benchmark::DoNotOptimize(tokens);
    }
    state.SetLabel("short string (" + std::to_string(text.size()) + " chars)");
}
BENCHMARK(BM_TokenizeShortString);

// ---------------------------------------------------------------------------
// BM_TokenizeLongString
// Measures tokenizer throughput on a 1000-word passage.
// ---------------------------------------------------------------------------
static void BM_TokenizeLongString(benchmark::State& state)
{
    const char* model_path = std::getenv("TRUELLM_TEST_MODEL_PATH");
    if (!model_path) {
        state.SkipWithError("TRUELLM_TEST_MODEL_PATH not set");
        return;
    }

    truellm::GgmlEngine eng(make_bench_inf(), make_bench_hw(), make_bench_smp());
    if (eng.load_model(model_path) != truellm::ErrorCode::Ok) {
        state.SkipWithError("Model failed to load");
        return;
    }

    // ~1000 words
    std::string text;
    for (int i = 0; i < 50; ++i) {
        text += "Transformer models use self-attention to process sequences in parallel. ";
    }

    for (auto _ : state) {
        auto tokens = eng.tokenize(text);
        benchmark::DoNotOptimize(tokens);
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                             static_cast<int64_t>(text.size()));
}
BENCHMARK(BM_TokenizeLongString);

// ---------------------------------------------------------------------------
// BM_GenerateSingleToken
// Measures a single decode step latency (1 token from a 16-token prompt).
// ---------------------------------------------------------------------------
static void BM_GenerateSingleToken(benchmark::State& state)
{
    const char* model_path = std::getenv("TRUELLM_TEST_MODEL_PATH");
    if (!model_path) {
        state.SkipWithError("TRUELLM_TEST_MODEL_PATH not set");
        return;
    }

    truellm::GgmlEngine eng(make_bench_inf(), make_bench_hw(), make_bench_smp());
    if (eng.load_model(model_path) != truellm::ErrorCode::Ok) {
        state.SkipWithError("Model failed to load");
        return;
    }

    const std::string prompt = "The quick brown fox";
    auto tokens = eng.tokenize(prompt);
    if (tokens.empty()) {
        state.SkipWithError("Tokenization returned empty");
        return;
    }

    for (auto _ : state) {
        truellm::GenerateRequest req;
        req.tokens     = tokens;
        req.max_tokens = 1;

        auto result = eng.generate(req);
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations());
    state.SetLabel("1 decode step, " + std::to_string(tokens.size()) + " prompt tokens");
}
BENCHMARK(BM_GenerateSingleToken);

// ---------------------------------------------------------------------------
// BM_Generate32Tokens
// Measures throughput generating 32 tokens from a short prompt.
// Reports tokens/second via SetItemsProcessed.
// ---------------------------------------------------------------------------
static void BM_Generate32Tokens(benchmark::State& state)
{
    const char* model_path = std::getenv("TRUELLM_TEST_MODEL_PATH");
    if (!model_path) {
        state.SkipWithError("TRUELLM_TEST_MODEL_PATH not set");
        return;
    }

    truellm::GgmlEngine eng(make_bench_inf(), make_bench_hw(), make_bench_smp());
    if (eng.load_model(model_path) != truellm::ErrorCode::Ok) {
        state.SkipWithError("Model failed to load");
        return;
    }

    const int gen_tokens = 32;
    auto tokens = eng.tokenize("Explain what a neural network is:");
    if (tokens.empty()) {
        state.SkipWithError("Tokenization returned empty");
        return;
    }

    for (auto _ : state) {
        truellm::GenerateRequest req;
        req.tokens     = tokens;
        req.max_tokens = gen_tokens;

        auto result = eng.generate(req);
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(
        static_cast<int64_t>(state.iterations()) * gen_tokens);
    state.SetLabel("32 generate tokens");
}
BENCHMARK(BM_Generate32Tokens)->Unit(benchmark::kMillisecond);
