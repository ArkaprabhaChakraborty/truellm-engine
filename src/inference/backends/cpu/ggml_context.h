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

    // Build a fresh sampler chain that additionally constrains decoding to the
    // given GBNF grammar (root rule "root").  Returns nullptr when the grammar
    // fails to parse, in which case callers fall back to the shared sampler().
    // The caller owns the returned sampler and must llama_sampler_free() it.
    llama_sampler* make_grammar_sampler(GgmlModel& model,
                                        const std::string& grammar) const;

    bool is_valid() const { return ctx_ != nullptr; }

private:
    const InferenceConfig& inf_cfg_;
    const SamplingConfig&  smp_cfg_;
    llama_context*         ctx_     = nullptr;
    llama_sampler*         sampler_ = nullptr;

    // Populate `chain` with the standard top_k/top_p/min_p/temp/penalties/dist
    // samplers from smp_cfg_.  Shared by build_sampler and make_grammar_sampler.
    void add_standard_samplers(llama_sampler* chain) const;
    void build_sampler(GgmlModel& model);
};

} // namespace truellm
