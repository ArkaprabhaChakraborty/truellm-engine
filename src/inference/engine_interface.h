#pragma once

// ---------------------------------------------------------------------------
// engine_interface.h — Pure abstract C++ interface for all inference backends
//
// LOAD-BEARING DECISION #2: Nothing above server/ or plugin_bridge/ ever
// includes a backend header.  The CUDA backend is invisible to all
// orchestration code.  Everything goes through this interface.
// ---------------------------------------------------------------------------

#include <truellm/types.h>
#include <truellm/config_schema.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace truellm {

// ---------------------------------------------------------------------------
// GpuStats — live GPU memory and offload info (zeros on CPU-only builds)
// ---------------------------------------------------------------------------
struct GpuStats {
    int         device_id        = -1;   // -1 = no GPU configured
    std::string device_name;
    int64_t     vram_total_bytes = 0;
    int64_t     vram_free_bytes  = 0;
    int32_t     layers_on_gpu    = 0;    // actual n_gpu_layers at load time
};

// ---------------------------------------------------------------------------
// KvCacheStats — paged KV block pool utilisation (CudaEngine only)
// Default-constructed (all zeros) when GgmlEngine is active.
// ---------------------------------------------------------------------------
struct KvCacheStats {
    int   block_size_tokens = 0;
    int   blocks_total      = 0;
    int   blocks_used       = 0;
    int   blocks_free       = 0;
    float utilization_pct   = 0.f;
    bool  fp8_enabled       = false;
};

// ---------------------------------------------------------------------------
// ModelInfo — metadata about the loaded model
// ---------------------------------------------------------------------------
struct ModelInfo {
    std::string model_id;        // e.g. "meta-llama/Llama-3.1-8B-Instruct"
    std::string architecture;    // e.g. "llama", "qwen2", "gemma"
    int64_t     context_length;  // max context window in tokens
    int64_t     vocab_size;
    int32_t     n_layers;
    int32_t     n_heads;
    int32_t     head_dim;
    DataType    dtype;           // weight precision

    // Chat-template seam (filled at load time): the model's own embedded
    // Jinja chat template from GGUF metadata, plus the BOS/EOS token *strings*.
    // The server layer feeds these to common_chat_templates_init (with a null
    // llama_model) so it can render faithful prompts and parse reasoning/tool
    // markers without including any backend/llama header.  chat_template is
    // empty when the GGUF carries no template — the server falls back to chatml.
    std::string chat_template;
    std::string bos_token;
    std::string eos_token;
};

// ---------------------------------------------------------------------------
// GenerateRequest — a single inference request
// ---------------------------------------------------------------------------
struct GenerateRequest {
    std::vector<int32_t> tokens;      // input token IDs
    int32_t              max_tokens  = 512;
    float                temperature = 0.7f;
    float                top_p       = 0.95f;
    int32_t              top_k       = 40;
    float                repeat_penalty = 1.1f;

    // Streaming: called for each generated token (empty = non-streaming)
    std::function<void(int32_t token_id, const std::string& text)> on_token;

    // Stop sequences (token IDs)
    std::vector<int32_t> stop_tokens;

    // Stop sequences (strings) — per-request, merged with SamplingConfig::stop
    std::vector<std::string> stop;

    // Optional GBNF grammar to constrain decoding (empty = unconstrained).
    // Produced by the server's chat_format layer from common_chat_params:
    // either a tool-call grammar (so the model emits well-formed calls) or a
    // response_format: json_schema grammar.  Applied via a per-request
    // llama_sampler_init_grammar prepended to the standard sampler chain.
    // grammar_lazy = true means the grammar is only enforced after one of the
    // trigger patterns appears (used for tool-call grammars so plain prose is
    // still allowed); CPU backend treats lazy grammars as best-effort.
    std::string          grammar;
    bool                 grammar_lazy = false;

    // Optional request id used for active-session tracking.  Backends that
    // implement inject_tokens key their request_id → seq_id map on this
    // value.  Empty string → no session tracking (the backend still works,
    // it just cannot service inject_tokens calls naming this request).
    std::string          request_id;

    // Optional session id for cross-request KV-cache prefix reuse
    // (Code_Capability_design.md §4.2).  When non-empty and the
    // scheduler's kv-prefix cache is enabled, two requests sharing the
    // same session_id and a common token prefix reuse the prefilled KV
    // cache from the first request — the second request only prefills
    // the tail tokens.  Strict perf optimisation; semantically the
    // request is identical whether or not reuse fires.  Empty string =
    // no reuse, normal scheduler behaviour.
    std::string          session_id;
};

// ---------------------------------------------------------------------------
// GenerateResult — output from a completed generation
// ---------------------------------------------------------------------------
struct GenerateResult {
    std::vector<int32_t> tokens;          // generated token IDs
    std::string          text;            // detokenized output
    int32_t              prompt_tokens;   // input token count
    int32_t              generated_tokens;// output token count
    float                time_ms;         // total generation time
    ErrorCode            error = ErrorCode::Ok;
    std::string          error_message;
    // "stop" | "length" | "error" — set by the scheduler / engine when
    // generation terminates.  Code_Capability_design.md §9.5: surfaced
    // back to plugins through vtable_call_engine's result_json so the
    // research_synthesizer can detect length-truncation and resume.
    std::string          finish_reason = "stop";
};

// ---------------------------------------------------------------------------
// EngineInterface — the backend wall
//
// CPU backend (GgmlEngine) and CUDA backend (CudaEngine) both implement
// this.  engine_factory.cpp reads the config and returns the correct one.
// ---------------------------------------------------------------------------
class EngineInterface {
public:
    virtual ~EngineInterface() = default;

    // Lifecycle
    virtual ErrorCode load_model(const std::string& model_path) = 0;
    virtual void      unload_model() = 0;
    virtual bool      is_model_loaded() const = 0;

    // Model metadata
    virtual ModelInfo get_model_info() const = 0;

    // Token budget: how many tokens can still fit in the context
    virtual int64_t get_token_budget() const = 0;

    // Core inference
    virtual GenerateResult generate(const GenerateRequest& request) = 0;

    // Tokenizer
    virtual std::vector<int32_t> tokenize(const std::string& text) const = 0;
    virtual std::string detokenize(const std::vector<int32_t>& tokens) const = 0;

    // GPU info (zeros on CPU-only builds or when no GPU is configured)
    virtual GpuStats get_gpu_stats() const { return {}; }

    // KV cache pool stats (CudaEngine only; GgmlEngine returns default zeros)
    virtual KvCacheStats get_kv_cache_stats() const { return {}; }

    // Embeddings (optional — not all backends support this)
    virtual bool supports_embeddings() const { return false; }
    virtual std::vector<float> embed(const std::string& text) {
        (void)text;
        return {};
    }

    // ── api_minor >= 1 (compression / RLM support) ────────────────────────────

    // Returns per-token hidden state vectors for the given layer from the last
    // forward pass. Shape: [seq_len × (head_dim * num_heads)], row-major F32.
    // Buffer is engine-owned; valid until next generate(). nullptr = unsupported.
    // out_size_bytes receives byte count of the returned buffer.
    virtual const void* get_hidden_states(int32_t layer_idx,
                                          int64_t* out_size_bytes) {
        (void)layer_idx;
        if (out_size_bytes) *out_size_bytes = 0;
        return nullptr;
    }

    // Returns [num_heads × seq_len × seq_len] row-major F32 attention weights
    // for the given layer. Engine-owned; valid until next generate().
    // nullptr when: CPU backend, megakernel_save_attn=false, seq_len too large.
    virtual const void* get_attention_weights(int32_t layer_idx,
                                              int64_t* out_size_bytes) {
        (void)layer_idx;
        if (out_size_bytes) *out_size_bytes = 0;
        return nullptr;
    }

    // Returns the raw KV cache tensor for the given layer.
    // out_shape[4] = {num_blocks, num_kv_heads, block_size, head_dim}.
    // type: 0 = keys, 1 = values.
    // nullptr on CPU backend or when paged-attention is not active.
    virtual const void* get_kv_cache_tensor(int32_t layer_idx, int32_t type,
                                            int64_t out_shape[4]) {
        (void)layer_idx; (void)type;
        if (out_shape) { out_shape[0] = out_shape[1] = out_shape[2] = out_shape[3] = 0; }
        return nullptr;
    }

    // Insert token IDs into the active request's KV cache at a given logical
    // position.  Used by RLM / compression plugins to splice summary tokens
    // into the running context without re-tokenising the entire message.
    //
    // request_id : matches ctx->request_id of the in-flight generate() call;
    //              the engine maps this to the underlying seq_id of the paged
    //              KV cache.
    // pos        : logical position in the sequence.  For append-only mode
    //              (the only mode CudaEngine currently supports), pos must
    //              equal the current end of the sequence's cached tokens.
    // tokens     : the IDs to project K/V for and write into the cache.
    //
    // Returns:
    //   ErrorCode::Ok                on success
    //   ErrorCode::InvalidArgument   if request_id has no active session,
    //                                pos != current length (non-append),
    //                                or tokens is empty
    //   ErrorCode::Unavailable       if the KV pool cannot accommodate the
    //                                additional slots
    //   ErrorCode::NotImplemented    if the backend has no insertion primitive
    //                                (the GgmlEngine CPU backend reports this)
    //
    // Implementation contract
    //   Backends that support inject_tokens must call register_active_session
    //   on entry to generate() and unregister_active_session on exit so that
    //   plugin callbacks fired between forward passes can see the live
    //   seq_id.  The append-only restriction sidesteps the need to shift
    //   logical-to-physical block maps for mid-sequence inserts (a future
    //   extension).
    virtual ErrorCode inject_tokens(const std::string& request_id,
                                    int64_t pos,
                                    const std::vector<int32_t>& tokens) {
        (void)request_id; (void)pos; (void)tokens;
        return ErrorCode::NotImplemented;
    }
};

// ---------------------------------------------------------------------------
// Factory: create_engine() — reads config, returns the correct backend.
// Defined in engine_factory.cpp
// ---------------------------------------------------------------------------
std::unique_ptr<EngineInterface> create_engine(
    const InferenceConfig& inf_cfg,
    const HardwareConfig&  hw_cfg,
    const SamplingConfig&  smp_cfg);

} // namespace truellm
