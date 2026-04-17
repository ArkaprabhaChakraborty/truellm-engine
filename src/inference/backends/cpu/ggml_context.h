#pragma once
// ---------------------------------------------------------------------------
// ggml_context.h — RAII owner of llama_context* + sampler chain
//
// Maps TrueLLMConfig fields -> llama_context_params and the sampler chain
// used for autoregressive token sampling.
// ---------------------------------------------------------------------------

#include <truellm/config_schema.h>

struct llama_context;
struct llama_sampler;

namespace truellm {

class GgmlModel;

class GgmlContext {
public:
    GgmlContext(const InferenceConfig& inf_cfg, const SamplingConfig& smp_cfg);
    ~GgmlContext();

    GgmlContext(const GgmlContext&)            = delete;
    GgmlContext& operator=(const GgmlContext&) = delete;

    // Build context + sampler chain from a loaded model.
    // Returns false on failure.
    bool init(GgmlModel& model);
    void reset();   // llama_kv_cache_clear — reuse context for a new request

    llama_context* raw()     { return ctx_; }
    llama_sampler* sampler() { return sampler_; }

    bool is_valid() const { return ctx_ != nullptr; }

private:
    const InferenceConfig& inf_cfg_;
    const SamplingConfig&  smp_cfg_;
    llama_context*         ctx_     = nullptr;
    llama_sampler*         sampler_ = nullptr;

    void build_sampler(GgmlModel& model);
};

} // namespace truellm
