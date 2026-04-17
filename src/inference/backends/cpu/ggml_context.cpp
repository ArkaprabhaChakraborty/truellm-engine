// ---------------------------------------------------------------------------
// ggml_context.cpp
// ---------------------------------------------------------------------------

#include "ggml_context.h"
#include "ggml_model.h"
#include "gpu/gpu_device.h"

#include <llama.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <thread>

namespace truellm {

// ---------------------------------------------------------------------------
// kv_cache type helper
// ---------------------------------------------------------------------------
static ggml_type kv_type_from_string(const std::string& s)
{
    if (s == "q8_0") return GGML_TYPE_Q8_0;
    if (s == "q4_0") return GGML_TYPE_Q4_0;
    if (s == "q4_1") return GGML_TYPE_Q4_1;
    return GGML_TYPE_F16; // default
}

// ---------------------------------------------------------------------------
// GgmlContext
// ---------------------------------------------------------------------------
GgmlContext::GgmlContext(const InferenceConfig& inf_cfg, const SamplingConfig& smp_cfg)
    : inf_cfg_(inf_cfg), smp_cfg_(smp_cfg)
{}

GgmlContext::~GgmlContext()
{
    if (sampler_) { llama_sampler_free(sampler_); sampler_ = nullptr; }
    if (ctx_)     { llama_free(ctx_);             ctx_     = nullptr; }
}

bool GgmlContext::init(GgmlModel& model)
{
    if (!model.is_loaded()) {
        spdlog::error("[GgmlContext] Model not loaded");
        return false;
    }

    llama_context_params cp = llama_context_default_params();

    cp.n_ctx    = static_cast<uint32_t>(inf_cfg_.context_size);
    cp.n_batch  = static_cast<uint32_t>(inf_cfg_.batch_size);
    cp.n_ubatch = static_cast<uint32_t>(inf_cfg_.ubatch_size);

    // Multi-sequence support for continuous batching
    if (inf_cfg_.batching.enabled && inf_cfg_.batching.max_seqs > 1) {
        cp.n_seq_max   = static_cast<uint32_t>(inf_cfg_.batching.max_seqs);
        cp.kv_unified  = inf_cfg_.batching.kv_unified;
    }

    // Thread count: 0 = hardware_concurrency
    int threads = inf_cfg_.threads;
    if (threads <= 0) {
        threads = static_cast<int>(std::thread::hardware_concurrency());
        if (threads <= 0) threads = 4;
    }
    cp.n_threads         = static_cast<uint32_t>(threads);
    cp.n_threads_batch   = static_cast<uint32_t>(threads);

    cp.flash_attn_type = inf_cfg_.flash_attention
                       ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                       : LLAMA_FLASH_ATTN_TYPE_DISABLED;

    cp.type_k = kv_type_from_string(inf_cfg_.kv_cache.type_k);
    cp.type_v = kv_type_from_string(inf_cfg_.kv_cache.type_v);

    // Quantized KV cache only works reliably with GPU. Fall back to F16 on
    // CPU-only builds and emit a warning so the operator knows why.
    if (GpuDevice::device_count() == 0) {
        if (cp.type_k != GGML_TYPE_F16) {
            spdlog::warn("[GgmlContext] Quantized KV type_k='{}' not supported on CPU — "
                         "falling back to f16", inf_cfg_.kv_cache.type_k);
            cp.type_k = GGML_TYPE_F16;
        }
        if (cp.type_v != GGML_TYPE_F16) {
            spdlog::warn("[GgmlContext] Quantized KV type_v='{}' not supported on CPU — "
                         "falling back to f16", inf_cfg_.kv_cache.type_v);
            cp.type_v = GGML_TYPE_F16;
        }
    }

    spdlog::info("[GgmlContext] KV cache: K={} V={} | flash_attn={}",
                 ggml_type_name(cp.type_k), ggml_type_name(cp.type_v),
                 inf_cfg_.flash_attention ? "on" : "off");

    ctx_ = llama_init_from_model(model.raw(), cp);
    if (!ctx_) {
        spdlog::error("[GgmlContext] llama_init_from_model failed");
        return false;
    }

    build_sampler(model);
    return true;
}

void GgmlContext::reset()
{
    if (ctx_) llama_memory_clear(llama_get_memory(ctx_), false);
}

void GgmlContext::build_sampler(GgmlModel& model)
{
    if (sampler_) {
        llama_sampler_free(sampler_);
        sampler_ = nullptr;
    }

    llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    sampler_ = llama_sampler_chain_init(chain_params);

    llama_sampler_chain_add(sampler_,
        llama_sampler_init_top_k(smp_cfg_.top_k));

    llama_sampler_chain_add(sampler_,
        llama_sampler_init_top_p(smp_cfg_.top_p, /*min_keep=*/1));

    llama_sampler_chain_add(sampler_,
        llama_sampler_init_min_p(smp_cfg_.min_p, /*min_keep=*/1));

    llama_sampler_chain_add(sampler_,
        llama_sampler_init_temp(smp_cfg_.temperature));

    llama_sampler_chain_add(sampler_,
        llama_sampler_init_penalties(
            64,   // penalty_last_n: last 64 tokens
            smp_cfg_.repeat_penalty,
            smp_cfg_.frequency_penalty,
            smp_cfg_.presence_penalty));

    // Seed: -1 in config = random
    uint32_t seed = (smp_cfg_.seed < 0)
                  ? static_cast<uint32_t>(std::chrono::steady_clock::now()
                        .time_since_epoch().count() & 0xFFFFFFFF)
                  : static_cast<uint32_t>(smp_cfg_.seed);

    llama_sampler_chain_add(sampler_, llama_sampler_init_dist(seed));

    (void)model; // reserved for future vocab-dependent samplers
}

} // namespace truellm
