#pragma once

// ---------------------------------------------------------------------------
// config_schema.h — Typed structs for every section of server.toml
//
// These are plain data structs: no toml11, no spdlog, no external headers.
// The TOML parser (src/host/config/config.cpp) fills them; everything else
// (engine_factory, server, plugins) reads them.
// ---------------------------------------------------------------------------

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace truellm {

// ─── [server] ────────────────────────────────────────────────────────────────
struct ServerConfig {
    std::string host                = "127.0.0.1";
    uint16_t    port                = 8080;
    std::string api_key             = "";
    std::vector<std::string> cors_origins = {"*"};
    int         request_timeout_ms  = 30000;
    int         max_concurrent      = 4;
    bool        tls_enabled         = false;
    std::string tls_cert_file       = "";
    std::string tls_key_file        = "";
};

// ─── [model] ─────────────────────────────────────────────────────────────────
struct ModelConfig {
    std::string path              = "";
    std::string hf_repo           = "";
    std::string hf_file           = "";
    std::string alias             = "";
    std::string chat_template     = "auto";   // "auto"|"chatml"|"llama3"|"gemma"|"phi3"
    bool        trust_remote_code = false;
};

// ─── [inference.kv_cache] ────────────────────────────────────────────────────
struct KvCacheConfig {
    std::string type_k  = "f16";   // "f16" | "q8_0" | "q4_0"
    std::string type_v  = "f16";
    int         size_mb = 0;       // 0 = let backend decide
};

// ─── [inference.cuda] ────────────────────────────────────────────────────────
struct CudaInferenceConfig {
    int                 gpu_layers              = -1;    // -1 = all
    int                 main_gpu                = 0;
    std::vector<float>  tensor_split            = {};
    float               gpu_memory_utilization  = 0.90f;
    bool                use_flash_attention_cuda = true;
    int                 compute_capability_major = 0;   // reserved: CudaEngine Phase 5 (sm_xx detection)
    int                 compute_capability_minor = 0;   // reserved: CudaEngine Phase 5
};

// ─── [inference.cuda_engine] ─────────────────────────────────────────────────
// Settings specific to the paged-attention CudaEngine (Part 2).
// These are separate from [inference.cuda] which controls GgmlEngine's
// GPU offload behaviour.
struct CudaEngineConfig {
    int   paged_kv_block_size       = 16;     // tokens per KV block (power of 2)
    int   max_num_blocks            = 0;      // 0 = auto from VRAM
    int   max_num_seqs              = 256;    // max concurrent live sequences
    int   num_cuda_streams          = 3;      // stream pool size (min 3)
    bool  use_fp8_kv                = false;  // FP8 KV cache (sm_89+ only)
    float gpu_memory_utilization    = 0.90f;  // fraction of free VRAM for KV pool
    int   tensor_parallel_size      = 1;      // GPUs for tensor parallelism (NCCL)
    bool  enforce_eager             = false;  // disable CUDA graph capture
    int   cuda_graph_max_batch_size = 8;      // max batch for graph capture
    std::string attention_layout    = "paged"; // "paged" | "flash_paged"
    // Megakernel cooperative launch (Phase M4+, sm_70+).
    // Falls back to legacy ggml+vllm path when false.
    bool  use_megakernel            = false;
    bool  megakernel_save_attn      = false;  // capture attn weights for Presis
    bool  megakernel_save_hidden    = false;  // capture hidden states for ToMe
};

// ─── [inference.compression.presis] ──────────────────────────────────────────
struct PresisConfig {
    bool        enabled             = true;
    float       threshold           = 0.85f; // trigger when seq_len > threshold × context budget
    float       keep_fraction       = 0.60f; // retain top keep_fraction tokens by importance
    int         importance_layers   = 4;     // layers to average attention over
    bool        fallback_tfidf      = true;  // use TF-IDF when attn weights unavailable
    int         max_seqlen_for_attn = 2048;  // disable attention scoring above this (OOM guard)
};

// ─── [inference.compression.tome] ────────────────────────────────────────────
struct ToMeConfig {
    bool        enabled         = false;
    int         r               = 8;     // token pairs to merge per application
    int         start_layer     = 0;     // first layer to apply bipartite matching
    bool        apply_all_layers = false; // apply at every layer (true) or once pre-inference
    std::string merge_mode      = "mean"; // "mean" | "max"
};

// ─── [inference.compression.fastv] ───────────────────────────────────────────
struct FastVConfig {
    bool        enabled          = false;
    int         start_layer      = 8;    // begin token pruning after this layer
    float       keep_ratio       = 0.50f;
    std::string importance_metric = "attn_sum"; // "attn_sum" | "attn_max"
};

// ─── [inference.compression.pyramid_drop] ────────────────────────────────────
struct PyramidDropConfig {
    bool        enabled          = false;
    float       final_keep_ratio = 0.30f;
    std::string drop_schedule    = "cosine"; // "cosine" | "linear"
    int         warmup_layers    = 4;
    int         protect_last_n_tokens = 8;  // never drop recent query tokens
};

// ─── [inference.compression.kv_kivi] ─────────────────────────────────────────
struct KiviConfig {
    bool        enabled           = false;
    int         bits              = 8;   // 4 or 8
    int         residual_length   = 128; // recent N tokens stored unquantized at FP16
    std::string key_granularity   = "per_channel"; // per-channel for keys (KVQuant finding)
    std::string value_granularity = "per_token";   // per-token for values
    bool        pre_rope_keys     = true; // quantize keys before RoPE application
};

// ─── [inference.compression.kv_vq] ───────────────────────────────────────────
// Online vector quantization — CUDA only; occupies TRUELLM_KERNEL_CACHE slot.
struct VqConfig {
    bool        enabled             = false;
    int         codebook_size       = 256;
    float       update_rate         = 0.001f; // EMA α for codebook updates
    int         residual_depth      = 1;      // 1 = standard VQ; 8 = residual VQ
    bool        outlier_tracing     = true;   // OTT: store outlier tokens at FP16
    std::string calibration_text    = "";     // path to calibration file for cold-start
    bool        disable_in_subcalls = true;   // no VQ during RLM ChunkEncoder sub-calls
};

// ─── [inference.compression] ─────────────────────────────────────────────────
struct CompressionConfig {
    bool        enabled  = false;

    // Ordered, comma-separated pipeline for token-level stages run before inference.
    // Valid names: "presis", "tome"
    // Note: "fastv" and "pyramid_drop" run inside the forward pass (megakernel slots),
    //   not in this pipeline. "kv_kivi" and "kv_vq" run during inference.
    // Architectural invariant: "tome" must appear before "presis" if both present.
    std::string pipeline = "presis";

    PresisConfig      presis;
    ToMeConfig        tome;
    FastVConfig       fastv;
    PyramidDropConfig pyramid_drop;
    KiviConfig        kv_kivi;
    VqConfig          kv_vq;
};

// ─── [inference.recursive_lm] ─────────────────────────────────────────────────
struct RlmConfig {
    bool    enabled                 = false;
    int     chunk_size_tokens       = 512;
    int     chunk_overlap_tokens    = 64;  // overlap prevents boundary information loss
    int     summary_tokens_per_chunk = 32;
    // WARNING: depth > 1 causes 95× latency blowup (arXiv 2512.24601).
    // Hard production limit = 1. Only set to 2 with summarization-finetuned models.
    int     max_hierarchy_depth     = 1;
    bool    chunk_cache_enabled     = false; // Phase C6 only — per-session chunk reuse
};

// ─── [inference.batching] ────────────────────────────────────────────────────
struct BatchingConfig {
    bool enabled       = false; // enable continuous batching (vLLM-style)
    int  max_seqs      = 8;     // max concurrent sequences in one decode step
    int  max_queue     = 256;   // max pending requests before rejecting with 503
    // kv_unified controls llama.cpp's KV cache layout when max_seqs > 1.
    // true  → n_ctx_seq = n_ctx          (each sequence can use the full context)
    // false → n_ctx_seq = n_ctx/max_seqs (context is partitioned equally per sequence)
    // Must be true whenever sequences can be longer than n_ctx/max_seqs.
    bool kv_unified    = true;
};

// ─── [inference] ─────────────────────────────────────────────────────────────
struct InferenceConfig {
    std::string       backend         = "cpu";        // "cpu" | "cuda"
    int               context_size    = 4096;
    int               batch_size      = 512;
    int               ubatch_size     = 512;
    int               threads         = 0;            // 0 = auto
    bool              flash_attention = true;
    bool              mmap            = true;
    bool              mlock           = false;
    std::string       numa            = "disabled";   // "disabled"|"distribute"|"isolate"
    KvCacheConfig       kv_cache;
    CudaInferenceConfig cuda;
    CudaEngineConfig    cuda_engine;
    BatchingConfig      batching;
    CompressionConfig   compression;
    RlmConfig           rlm;
};

// ─── [sampling] ──────────────────────────────────────────────────────────────
struct SamplingConfig {
    float                    temperature       = 0.7f;
    float                    top_p             = 0.95f;
    int                      top_k             = 40;
    float                    min_p             = 0.05f;
    float                    repeat_penalty    = 1.1f;
    float                    frequency_penalty = 0.0f;
    float                    presence_penalty  = 0.0f;
    int                      max_tokens        = 2048;
    int                      seed              = -1;
    std::vector<std::string> stop              = {};
};

// ─── [logging] ───────────────────────────────────────────────────────────────
struct LoggingConfig {
    std::string level           = "info";   // "trace"|"debug"|"info"|"warn"|"error"
    std::string format          = "text";   // "text" | "json"
    std::string file            = "";
    bool        timestamps      = true;
    bool        source_location = false;
};

// ─── [hardware.gpu] ──────────────────────────────────────────────────────────
struct GpuDeviceConfig {
    bool        enabled          = true;
    int         device_id        = 0;
    std::string name_hint        = "";      // display / disambiguation
    float       vram_capacity_gb = 0.0f;   // 0 = auto-detect
    std::string backend_api      = "cuda"; // "cuda" | "vulkan"
};

// ─── [hardware.offload] ──────────────────────────────────────────────────────
struct OffloadConfig {
    bool  use_gpu_vram          = true;
    bool  use_system_ram        = true;
    bool  use_shared_memory     = false;  // iGPU / UMA overflow
    int   gpu_layers            = -1;     // -1 = auto (fill VRAM)
    float vram_budget_fraction  = 1.0f;  // fraction of detected VRAM for weights
};

// ─── [hardware.guardrails] ───────────────────────────────────────────────────
struct GuardrailsConfig {
    // "off" | "relaxed" | "balanced" | "strict" | "custom"
    std::string preset              = "balanced";
    float       max_model_size_gb   = 0.0f;   // used when preset = "custom"
    bool        allow_overcommit    = false;
    float       ram_fraction_limit  = 0.0f;   // 0 = use preset default
    float       vram_fraction_limit = 0.0f;   // 0 = use preset default
};

// ─── [hardware] ──────────────────────────────────────────────────────────────
struct HardwareConfig {
    int              cpu_threads                   = 0;   // 0 = auto
    std::vector<int> cpu_affinity                  = {};
    float            max_ram_gb                    = 0.0f;
    int              resource_monitor_interval_ms  = 1000;
    GpuDeviceConfig  gpu;
    OffloadConfig    offload;
    GuardrailsConfig guardrails;
};

// ─── [runtime.harmony] ───────────────────────────────────────────────────────
struct HarmonyConfig {
    bool        enabled = false;
    std::string version = "latest";
};

// ─── [runtime.extensions[]] ──────────────────────────────────────────────────
struct RuntimeExtensionConfig {
    std::string name    = "";
    bool        enabled = false;
    std::string version = "latest";
};

// ─── [runtime] ───────────────────────────────────────────────────────────────
// Selects the llama.cpp runtime extension pack:
//   "cpu-llama-cpp"    — CPU-only
//   "cuda12-llama-cpp" — NVIDIA CUDA 12.x
//   "cuda-llama-cpp"   — NVIDIA CUDA (auto 11/12)
//   "vulkan-llama-cpp" — Vulkan compute (cross-vendor)
struct RuntimeConfig {
    std::string engine         = "cpu-llama-cpp";
    std::string engine_version = "latest";
    bool        auto_update    = false;
    std::string update_channel = "stable";   // "stable" | "nightly"
    HarmonyConfig harmony;
    std::vector<RuntimeExtensionConfig> extensions;
};

// ─── [plugins] ───────────────────────────────────────────────────────────────
struct PluginConfig {
    bool                     enabled              = false;
    std::string              python_path          = "python3";
    std::vector<std::string> plugin_dirs          = {};
    std::string              socket_path          = "";
    int                      sidecar_timeout_ms   = 5000;
    bool                     restart_on_crash     = true;
    int                      max_restart_attempts = 3;

    // ── Tool dispatch ─────────────────────────────────────────────────────────
    // tool_max_rounds — maximum tool-call / dispatch iterations per request
    //                   before the engine returns whatever the model last said.
    int tool_max_rounds = 5;

    // ── Context splitter ─────────────────────────────────────────────────────
    // Activated when a tool result is too large for the remaining context window.
    //
    // context_split_enabled
    //   Master switch.
    //
    // context_split_strategy
    //   "summarize" — per-chunk inference pass (slower, semantic compression).
    //   "truncate"  — hard-clip with no extra inference (fast, deterministic).
    //
    // context_split_trigger_fraction
    //   Trigger the splitter when the raw tool content alone exceeds
    //   (token_budget * trigger_fraction).  Default 0.5.
    //
    // context_split_truncate_fraction
    //   When strategy = "truncate", keep at most (token_budget * fraction) chars.
    //   Default 0.60.
    //
    // context_split_auto_chunk_fraction
    //   When strategy = "summarize" and chunk_tokens = 0 (auto), use
    //   (token_budget * fraction) as the chunk token target.  Default 0.55.
    //
    // context_split_chunk_tokens
    //   Explicit chunk size in tokens; 0 = auto (uses auto_chunk_fraction).
    //
    // context_split_max_chunks
    //   Maximum chunks to summarise per tool result.
    //
    // context_split_summary_tokens
    //   Max tokens generated per chunk summary.
    bool        context_split_enabled            = true;
    std::string context_split_strategy           = "summarize"; // "summarize" | "truncate"
    float       context_split_trigger_fraction   = 0.5f;
    float       context_split_truncate_fraction  = 0.60f;
    float       context_split_auto_chunk_fraction = 0.55f;
    int         context_split_chunk_tokens       = 0;
    int         context_split_max_chunks         = 8;
    int         context_split_summary_tokens     = 300;

    // Flat key-value settings exposed to every plugin via host_api.get_config().
    // Populated from [plugins.settings] in server.toml.
    // Keys use "plugin_id.setting_name" convention for per-plugin isolation,
    // or bare "setting_name" for global plugin settings.
    // Example: get_config("tool_web_fetch", "proxy_url") looks up
    //   "tool_web_fetch.proxy_url" then falls back to "proxy_url".
    std::unordered_map<std::string, std::string> settings;
};

// ─── Top-level config ─────────────────────────────────────────────────────────
struct TrueLLMConfig {
    std::string    profile;   // active profile preset: "local-cpu"|"workstation"|"server"|""
    ServerConfig   server;
    ModelConfig    model;
    InferenceConfig inference;
    SamplingConfig  sampling;
    LoggingConfig   logging;
    HardwareConfig  hardware;
    RuntimeConfig   runtime;
    PluginConfig    plugins;
};

} // namespace truellm
