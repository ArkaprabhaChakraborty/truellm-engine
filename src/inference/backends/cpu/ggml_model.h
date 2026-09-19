#pragma once
// ---------------------------------------------------------------------------
// ggml_model.h — RAII owner of llama_model*
//
// Maps TrueLLMConfig fields -> llama_model_params, loads a GGUF file,
// and exposes the raw pointer for context/sampler construction.
// ---------------------------------------------------------------------------

#include <truellm/config_schema.h>
#include <truellm/types.h>

#include <string>

// Forward-declare llama types so we don't pull llama.h into every TU
struct llama_model;
struct llama_vocab;

namespace truellm {

class GgmlModel {
public:
    GgmlModel(const InferenceConfig& inf_cfg, const HardwareConfig& hw_cfg);
    ~GgmlModel();

    // Non-copyable, movable
    GgmlModel(const GgmlModel&)            = delete;
    GgmlModel& operator=(const GgmlModel&) = delete;
    GgmlModel(GgmlModel&&)                 = default;
    GgmlModel& operator=(GgmlModel&&)      = default;

    ErrorCode  load(const std::string& path);
    void       unload();
    bool       is_loaded() const { return model_ != nullptr; }

    llama_model*       raw()       { return model_; }
    const llama_model* raw() const { return model_; }
    const llama_vocab* vocab() const;

    // Metadata helpers (valid only when loaded)
    int64_t n_ctx_train() const;           // model's trained context length
    int32_t n_layers()    const;
    int64_t vocab_size()  const;
    int64_t layer_size_bytes_estimate() const; // avg weight bytes per layer

    // Chat-template seam for the server's chat_format layer.
    // chat_template(): the GGUF-embedded Jinja template ("" when absent).
    // bos_token()/eos_token(): the BOS/EOS token rendered to its piece string.
    std::string chat_template() const;
    std::string bos_token()     const;
    std::string eos_token()     const;

private:
    const InferenceConfig& inf_cfg_;
    const HardwareConfig&  hw_cfg_;
    llama_model*           model_ = nullptr;
};

} // namespace truellm
