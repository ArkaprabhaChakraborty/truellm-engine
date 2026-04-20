#pragma once
// ---------------------------------------------------------------------------
// ggml_engine.h — EngineInterface implementation backed by llama.cpp
//
// Handles both CPU and GPU inference: when compiled with GGML_CUDA and
// hw_cfg.offload.gpu_layers != 0, layers are fully offloaded to the GPU.
// No separate CudaEngine is needed for Phase 1.
// ---------------------------------------------------------------------------

#include "../../engine_interface.h"
#include <truellm/config_schema.h>

#include <memory>

namespace truellm {

class GgmlModel;
class GgmlContext;
class ContinuousBatchScheduler;

class GgmlEngine : public EngineInterface {
public:
    GgmlEngine(InferenceConfig inf_cfg,
               HardwareConfig  hw_cfg,
               SamplingConfig  smp_cfg);
    ~GgmlEngine() override;

    // --- EngineInterface ---
    ErrorCode   load_model(const std::string& model_path) override;
    void        unload_model() override;
    bool        is_model_loaded() const override;
    ModelInfo   get_model_info() const override;
    int64_t     get_token_budget() const override;

    GenerateResult            generate(const GenerateRequest& req) override;
    std::vector<int32_t>      tokenize(const std::string& text) const override;
    std::string               detokenize(const std::vector<int32_t>& tokens) const override;
    GpuStats                  get_gpu_stats() const override;

    // ── api_minor >= 1 — compression / RLM data access ───────────────────────
    // get_hidden_states: uses llama_get_embeddings() to read per-token output
    //   embeddings from the last forward pass.
    const void* get_hidden_states(int32_t layer_idx,
                                  int64_t* out_size_bytes) override;
    // get_attention_weights: CPU path returns nullptr (attention weights not
    //   saved by llama.cpp on CPU — Presis falls back to TF-IDF automatically).
    const void* get_attention_weights(int32_t layer_idx,
                                      int64_t* out_size_bytes) override;
    // get_kv_cache_tensor: returns nullptr — paged KV not present on CPU path.
    const void* get_kv_cache_tensor(int32_t layer_idx, int32_t type,
                                    int64_t out_shape[4]) override;

private:
    InferenceConfig inf_cfg_;
    HardwareConfig  hw_cfg_;
    SamplingConfig  smp_cfg_;

    std::unique_ptr<GgmlModel>   model_;
    std::unique_ptr<GgmlContext> ctx_;

    // Continuous batching scheduler (null when batching.enabled == false)
    std::unique_ptr<ContinuousBatchScheduler> scheduler_;

    // Stop-string matching helper used inside synchronous generate()
    bool matches_stop(const std::string& text) const;
};

} // namespace truellm
