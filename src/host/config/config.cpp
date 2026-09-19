// ---------------------------------------------------------------------------
// config.cpp — TOML parser for TrueLLM server configuration
//
// Uses toml11 v4 (find / find_or API).
// ---------------------------------------------------------------------------

#include "config.h"
#include "profile.h"
#include "secrets.h"

#include <truellm/config_schema.h>

#include <fmt/core.h>
#include <spdlog/spdlog.h>
#include <toml.hpp>

#include <functional>
#include <stdexcept>
#include <string>

namespace truellm::config {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

// Resolve secrets then return the processed string
std::string s(const std::string& v) { return resolve_secrets(v); }

// Retrieve a TOML string field with env-var substitution applied
std::string find_str(const toml::value& tbl, const std::string& key,
                     const std::string& def)
{
    return s(toml::find_or<std::string>(tbl, key, def));
}

// Parse [server]
void parse_server(const toml::value& root, ServerConfig& out)
{
    if (!root.contains("server")) return;
    const auto& t = toml::find(root, "server");

    out.host               = find_str(t, "host",    out.host);
    out.port               = static_cast<uint16_t>(
                                 toml::find_or<int>(t, "port", out.port));
    out.api_key            = find_str(t, "api_key", out.api_key);
    out.cors_origins       = toml::find_or<std::vector<std::string>>(
                                 t, "cors_origins", out.cors_origins);
    out.request_timeout_ms = toml::find_or<int>(t, "request_timeout_ms",
                                                out.request_timeout_ms);
    out.max_concurrent     = toml::find_or<int>(t, "max_concurrent",
                                                out.max_concurrent);
    out.tls_enabled        = toml::find_or<bool>(t, "tls_enabled",   out.tls_enabled);
    out.tls_cert_file      = find_str(t, "tls_cert_file", out.tls_cert_file);
    out.tls_key_file       = find_str(t, "tls_key_file",  out.tls_key_file);
    out.enable_debug_endpoints =
        toml::find_or<bool>(t, "enable_debug_endpoints", out.enable_debug_endpoints);
}

// Parse [model]
void parse_model(const toml::value& root, ModelConfig& out)
{
    if (!root.contains("model")) return;
    const auto& t = toml::find(root, "model");

    out.path              = find_str(t, "path",              out.path);
    out.hf_repo           = find_str(t, "hf_repo",           out.hf_repo);
    out.hf_file           = find_str(t, "hf_file",           out.hf_file);
    out.alias             = find_str(t, "alias",             out.alias);
    out.chat_template     = find_str(t, "chat_template",     out.chat_template);
    out.trust_remote_code = toml::find_or<bool>(t, "trust_remote_code",
                                                out.trust_remote_code);
}

// Parse [inference.kv_cache]
void parse_kv_cache(const toml::value& inf, KvCacheConfig& out)
{
    if (!inf.contains("kv_cache")) return;
    const auto& t = toml::find(inf, "kv_cache");

    out.type_k  = find_str(t, "type_k",  out.type_k);
    out.type_v  = find_str(t, "type_v",  out.type_v);
    out.size_mb = toml::find_or<int>(t, "size_mb", out.size_mb);
}

// Parse [inference.cuda]
void parse_cuda_inference(const toml::value& inf, CudaInferenceConfig& out)
{
    if (!inf.contains("cuda")) return;
    const auto& t = toml::find(inf, "cuda");

    out.gpu_layers     = toml::find_or<int>(t, "gpu_layers",  out.gpu_layers);
    out.main_gpu       = toml::find_or<int>(t, "main_gpu",    out.main_gpu);
    out.tensor_split   = toml::find_or<std::vector<float>>(t, "tensor_split",
                                                           out.tensor_split);
    out.gpu_memory_utilization  = static_cast<float>(
        toml::find_or<double>(t, "gpu_memory_utilization",
                              static_cast<double>(out.gpu_memory_utilization)));
    out.use_flash_attention_cuda = toml::find_or<bool>(t, "use_flash_attention_cuda",
                                                       out.use_flash_attention_cuda);
    out.compute_capability_major = toml::find_or<int>(t, "compute_capability_major",
                                                      out.compute_capability_major);
    out.compute_capability_minor = toml::find_or<int>(t, "compute_capability_minor",
                                                      out.compute_capability_minor);
}

// Parse [inference.cuda_engine]
void parse_cuda_engine(const toml::value& inf, CudaEngineConfig& out)
{
    if (!inf.contains("cuda_engine")) return;
    const auto& t = toml::find(inf, "cuda_engine");

    out.paged_kv_block_size    = toml::find_or<int>(t, "paged_kv_block_size",
                                                    out.paged_kv_block_size);
    out.max_num_blocks         = toml::find_or<int>(t, "max_num_blocks",
                                                    out.max_num_blocks);
    out.max_num_seqs           = toml::find_or<int>(t, "max_num_seqs",
                                                    out.max_num_seqs);
    out.num_cuda_streams       = toml::find_or<int>(t, "num_cuda_streams",
                                                    out.num_cuda_streams);
    out.use_fp8_kv             = toml::find_or<bool>(t, "use_fp8_kv",
                                                     out.use_fp8_kv);
    out.gpu_memory_utilization = static_cast<float>(
        toml::find_or<double>(t, "gpu_memory_utilization",
                              static_cast<double>(out.gpu_memory_utilization)));
    out.tensor_parallel_size   = toml::find_or<int>(t, "tensor_parallel_size",
                                                    out.tensor_parallel_size);
    out.enforce_eager          = toml::find_or<bool>(t, "enforce_eager",
                                                     out.enforce_eager);
    out.cuda_graph_max_batch_size = toml::find_or<int>(t, "cuda_graph_max_batch_size",
                                                        out.cuda_graph_max_batch_size);
    out.attention_layout       = find_str(t, "attention_layout", out.attention_layout);
}

// Parse [inference]
void parse_inference(const toml::value& root, InferenceConfig& out)
{
    if (!root.contains("inference")) return;
    const auto& t = toml::find(root, "inference");

    out.backend         = find_str(t, "backend",  out.backend);
    out.context_size    = toml::find_or<int>(t,  "context_size",  out.context_size);
    out.batch_size      = toml::find_or<int>(t,  "batch_size",    out.batch_size);
    out.ubatch_size     = toml::find_or<int>(t,  "ubatch_size",   out.ubatch_size);
    out.threads         = toml::find_or<int>(t,  "threads",       out.threads);
    out.flash_attention = toml::find_or<bool>(t, "flash_attention", out.flash_attention);
    out.mmap            = toml::find_or<bool>(t, "mmap",          out.mmap);
    out.mlock           = toml::find_or<bool>(t, "mlock",         out.mlock);
    out.numa            = find_str(t, "numa",                      out.numa);

    parse_kv_cache(t, out.kv_cache);
    parse_cuda_inference(t, out.cuda);
    parse_cuda_engine(t, out.cuda_engine);

    if (t.contains("batching")) {
        const auto& b = toml::find(t, "batching");
        out.batching.enabled    = toml::find_or<bool>(b, "enabled",    out.batching.enabled);
        out.batching.max_seqs   = toml::find_or<int> (b, "max_seqs",   out.batching.max_seqs);
        out.batching.max_queue  = toml::find_or<int> (b, "max_queue",  out.batching.max_queue);
        out.batching.kv_unified = toml::find_or<bool>(b, "kv_unified", out.batching.kv_unified);
    }

    if (t.contains("compression")) {
        const auto& c = toml::find(t, "compression");
        out.compression.enabled  = toml::find_or<bool>(c, "enabled", out.compression.enabled);
        out.compression.pipeline = find_str(c, "pipeline", out.compression.pipeline);

        if (c.contains("presis")) {
            const auto& p = toml::find(c, "presis");
            auto& pr = out.compression.presis;
            pr.enabled             = toml::find_or<bool>(p, "enabled",             pr.enabled);
            pr.threshold           = static_cast<float>(toml::find_or<double>(p, "threshold",
                                         static_cast<double>(pr.threshold)));
            pr.keep_fraction       = static_cast<float>(toml::find_or<double>(p, "keep_fraction",
                                         static_cast<double>(pr.keep_fraction)));
            pr.importance_layers   = toml::find_or<int>(p, "importance_layers",   pr.importance_layers);
            pr.fallback_tfidf      = toml::find_or<bool>(p, "fallback_tfidf",     pr.fallback_tfidf);
            pr.max_seqlen_for_attn = toml::find_or<int>(p, "max_seqlen_for_attn", pr.max_seqlen_for_attn);
        }

        if (c.contains("tome")) {
            const auto& p = toml::find(c, "tome");
            auto& tm = out.compression.tome;
            tm.enabled          = toml::find_or<bool>(p, "enabled",          tm.enabled);
            tm.r                = toml::find_or<int> (p, "r",                tm.r);
            tm.start_layer      = toml::find_or<int> (p, "start_layer",      tm.start_layer);
            tm.apply_all_layers = toml::find_or<bool>(p, "apply_all_layers", tm.apply_all_layers);
            tm.merge_mode       = find_str(p, "merge_mode",                   tm.merge_mode);
        }

        if (c.contains("fastv")) {
            const auto& p = toml::find(c, "fastv");
            auto& fv = out.compression.fastv;
            fv.enabled           = toml::find_or<bool>(p, "enabled",     fv.enabled);
            fv.start_layer       = toml::find_or<int> (p, "start_layer", fv.start_layer);
            fv.keep_ratio        = static_cast<float>(toml::find_or<double>(p, "keep_ratio",
                                       static_cast<double>(fv.keep_ratio)));
            fv.importance_metric = find_str(p, "importance_metric",      fv.importance_metric);
        }

        if (c.contains("pyramid_drop")) {
            const auto& p = toml::find(c, "pyramid_drop");
            auto& pd = out.compression.pyramid_drop;
            pd.enabled               = toml::find_or<bool>(p, "enabled",      pd.enabled);
            pd.final_keep_ratio      = static_cast<float>(toml::find_or<double>(p, "final_keep_ratio",
                                           static_cast<double>(pd.final_keep_ratio)));
            pd.drop_schedule         = find_str(p, "drop_schedule",            pd.drop_schedule);
            pd.warmup_layers         = toml::find_or<int>(p, "warmup_layers",  pd.warmup_layers);
            pd.protect_last_n_tokens = toml::find_or<int>(p, "protect_last_n_tokens",
                                           pd.protect_last_n_tokens);
        }

        if (c.contains("kv_kivi")) {
            const auto& p = toml::find(c, "kv_kivi");
            auto& ki = out.compression.kv_kivi;
            ki.enabled           = toml::find_or<bool>(p, "enabled",           ki.enabled);
            ki.bits              = toml::find_or<int> (p, "bits",              ki.bits);
            ki.residual_length   = toml::find_or<int> (p, "residual_length",   ki.residual_length);
            ki.key_granularity   = find_str(p, "key_granularity",               ki.key_granularity);
            ki.value_granularity = find_str(p, "value_granularity",             ki.value_granularity);
            ki.pre_rope_keys     = toml::find_or<bool>(p, "pre_rope_keys",     ki.pre_rope_keys);
        }

        if (c.contains("kv_vq")) {
            const auto& p = toml::find(c, "kv_vq");
            auto& vq = out.compression.kv_vq;
            vq.enabled             = toml::find_or<bool>(p, "enabled",             vq.enabled);
            vq.codebook_size       = toml::find_or<int> (p, "codebook_size",       vq.codebook_size);
            vq.update_rate         = static_cast<float>(toml::find_or<double>(p, "update_rate",
                                         static_cast<double>(vq.update_rate)));
            vq.residual_depth      = toml::find_or<int> (p, "residual_depth",      vq.residual_depth);
            vq.outlier_tracing     = toml::find_or<bool>(p, "outlier_tracing",     vq.outlier_tracing);
            vq.calibration_text    = find_str(p, "calibration_text",               vq.calibration_text);
            vq.disable_in_subcalls = toml::find_or<bool>(p, "disable_in_subcalls", vq.disable_in_subcalls);
        }
    }

    if (t.contains("recursive_lm")) {
        const auto& r = toml::find(t, "recursive_lm");
        auto& rlm = out.rlm;
        rlm.enabled                  = toml::find_or<bool>(r, "enabled",                  rlm.enabled);
        rlm.chunk_size_tokens        = toml::find_or<int> (r, "chunk_size_tokens",        rlm.chunk_size_tokens);
        rlm.chunk_overlap_tokens     = toml::find_or<int> (r, "chunk_overlap_tokens",     rlm.chunk_overlap_tokens);
        rlm.summary_tokens_per_chunk = toml::find_or<int> (r, "summary_tokens_per_chunk", rlm.summary_tokens_per_chunk);
        rlm.max_hierarchy_depth      = toml::find_or<int> (r, "max_hierarchy_depth",      rlm.max_hierarchy_depth);
        rlm.chunk_cache_enabled      = toml::find_or<bool>(r, "chunk_cache_enabled",      rlm.chunk_cache_enabled);
        if (rlm.max_hierarchy_depth > 1) {
            spdlog::warn("[Config] inference.recursive_lm.max_hierarchy_depth={} > 1 — "
                         "depth=2 causes 95× latency blowup (arXiv 2512.24601); "
                         "only use with summarization-finetuned models",
                         rlm.max_hierarchy_depth);
        }
    }

    // [inference.reasoning] — thinking/reasoning_content separation.
    if (t.contains("reasoning")) {
        const auto& r = toml::find(t, "reasoning");
        auto& rs = out.reasoning;
        rs.format          = toml::find_or<std::string>(r, "format",          rs.format);
        rs.enable_thinking = toml::find_or<bool>       (r, "enable_thinking", rs.enable_thinking);
    }
}

// Parse [sampling]
void parse_sampling(const toml::value& root, SamplingConfig& out)
{
    if (!root.contains("sampling")) return;
    const auto& t = toml::find(root, "sampling");

    out.temperature       = static_cast<float>(
        toml::find_or<double>(t, "temperature",       static_cast<double>(out.temperature)));
    out.top_p             = static_cast<float>(
        toml::find_or<double>(t, "top_p",             static_cast<double>(out.top_p)));
    out.top_k             = toml::find_or<int>(t, "top_k",         out.top_k);
    out.min_p             = static_cast<float>(
        toml::find_or<double>(t, "min_p",             static_cast<double>(out.min_p)));
    out.repeat_penalty    = static_cast<float>(
        toml::find_or<double>(t, "repeat_penalty",    static_cast<double>(out.repeat_penalty)));
    out.frequency_penalty = static_cast<float>(
        toml::find_or<double>(t, "frequency_penalty", static_cast<double>(out.frequency_penalty)));
    out.presence_penalty  = static_cast<float>(
        toml::find_or<double>(t, "presence_penalty",  static_cast<double>(out.presence_penalty)));
    out.max_tokens        = toml::find_or<int>(t, "max_tokens",    out.max_tokens);
    out.seed              = toml::find_or<int>(t, "seed",          out.seed);
    out.stop              = toml::find_or<std::vector<std::string>>(t, "stop", out.stop);
}

// Parse [logging]
void parse_logging(const toml::value& root, LoggingConfig& out)
{
    if (!root.contains("logging")) return;
    const auto& t = toml::find(root, "logging");

    out.level           = find_str(t, "level",  out.level);
    out.format          = find_str(t, "format", out.format);
    out.file            = find_str(t, "file",   out.file);
    out.timestamps      = toml::find_or<bool>(t, "timestamps",     out.timestamps);
    out.source_location = toml::find_or<bool>(t, "source_location", out.source_location);
}

// Parse [hardware.gpu]
void parse_gpu(const toml::value& hw, GpuDeviceConfig& out)
{
    if (!hw.contains("gpu")) return;
    const auto& t = toml::find(hw, "gpu");

    out.enabled          = toml::find_or<bool>(t, "enabled",     out.enabled);
    out.device_id        = toml::find_or<int>(t,  "device_id",   out.device_id);
    out.name_hint        = find_str(t, "name_hint",               out.name_hint);
    out.vram_capacity_gb = static_cast<float>(
        toml::find_or<double>(t, "vram_capacity_gb", static_cast<double>(out.vram_capacity_gb)));
    out.backend_api      = find_str(t, "backend_api",             out.backend_api);
}

// Parse [hardware.offload]
void parse_offload(const toml::value& hw, OffloadConfig& out)
{
    if (!hw.contains("offload")) return;
    const auto& t = toml::find(hw, "offload");

    out.use_gpu_vram         = toml::find_or<bool>(t, "use_gpu_vram",     out.use_gpu_vram);
    out.use_system_ram       = toml::find_or<bool>(t, "use_system_ram",   out.use_system_ram);
    out.use_shared_memory    = toml::find_or<bool>(t, "use_shared_memory",out.use_shared_memory);
    out.gpu_layers           = toml::find_or<int>(t,  "gpu_layers",       out.gpu_layers);
    out.vram_budget_fraction = static_cast<float>(
        toml::find_or<double>(t, "vram_budget_fraction",
                              static_cast<double>(out.vram_budget_fraction)));
}

// Parse [hardware.guardrails]
void parse_guardrails(const toml::value& hw, GuardrailsConfig& out)
{
    if (!hw.contains("guardrails")) return;
    const auto& t = toml::find(hw, "guardrails");

    out.preset              = find_str(t, "preset",             out.preset);
    out.max_model_size_gb   = static_cast<float>(
        toml::find_or<double>(t, "max_model_size_gb",
                              static_cast<double>(out.max_model_size_gb)));
    out.allow_overcommit    = toml::find_or<bool>(t, "allow_overcommit", out.allow_overcommit);
    out.ram_fraction_limit  = static_cast<float>(
        toml::find_or<double>(t, "ram_fraction_limit",
                              static_cast<double>(out.ram_fraction_limit)));
    out.vram_fraction_limit = static_cast<float>(
        toml::find_or<double>(t, "vram_fraction_limit",
                              static_cast<double>(out.vram_fraction_limit)));
}

// Parse [hardware]
void parse_hardware(const toml::value& root, HardwareConfig& out)
{
    if (!root.contains("hardware")) return;
    const auto& t = toml::find(root, "hardware");

    out.cpu_threads    = toml::find_or<int>(t, "cpu_threads", out.cpu_threads);
    out.cpu_affinity   = toml::find_or<std::vector<int>>(t, "cpu_affinity", out.cpu_affinity);
    out.max_ram_gb     = static_cast<float>(
        toml::find_or<double>(t, "max_ram_gb", static_cast<double>(out.max_ram_gb)));
    out.resource_monitor_interval_ms = toml::find_or<int>(
        t, "resource_monitor_interval_ms", out.resource_monitor_interval_ms);

    parse_gpu(t, out.gpu);
    parse_offload(t, out.offload);
    parse_guardrails(t, out.guardrails);
}

// Parse [runtime.harmony]
void parse_harmony(const toml::value& rt, HarmonyConfig& out)
{
    if (!rt.contains("harmony")) return;
    const auto& t = toml::find(rt, "harmony");

    out.enabled = toml::find_or<bool>(t,   "enabled", out.enabled);
    out.version = find_str(t, "version",              out.version);
}

// Parse [[runtime.extensions]]
void parse_runtime_extensions(const toml::value& rt,
                               std::vector<RuntimeExtensionConfig>& out)
{
    if (!rt.contains("extensions")) return;
    const auto& arr = toml::find(rt, "extensions");
    if (!arr.is_array()) return;

    for (const auto& item : arr.as_array()) {
        RuntimeExtensionConfig ext;
        ext.name    = find_str(item, "name",    ext.name);
        ext.enabled = toml::find_or<bool>(item, "enabled", ext.enabled);
        ext.version = find_str(item, "version", ext.version);
        out.push_back(std::move(ext));
    }
}

// Parse [runtime]
void parse_runtime(const toml::value& root, RuntimeConfig& out)
{
    if (!root.contains("runtime")) return;
    const auto& t = toml::find(root, "runtime");

    out.engine         = find_str(t, "engine",         out.engine);
    out.engine_version = find_str(t, "engine_version", out.engine_version);
    out.auto_update    = toml::find_or<bool>(t, "auto_update",   out.auto_update);
    out.update_channel = find_str(t, "update_channel",            out.update_channel);

    parse_harmony(t, out.harmony);
    parse_runtime_extensions(t, out.extensions);
}

// Parse [plugins]
void parse_plugins(const toml::value& root, PluginConfig& out)
{
    if (!root.contains("plugins")) return;
    const auto& t = toml::find(root, "plugins");

    out.enabled    = toml::find_or<bool>(t, "enabled",  out.enabled);
    out.python_path = find_str(t, "python_path",         out.python_path);
    out.plugin_dirs = toml::find_or<std::vector<std::string>>(t, "plugin_dirs",
                                                               out.plugin_dirs);
    out.socket_path          = find_str(t, "socket_path",         out.socket_path);
    out.sidecar_timeout_ms   = toml::find_or<int>(t, "sidecar_timeout_ms",
                                                  out.sidecar_timeout_ms);
    out.restart_on_crash     = toml::find_or<bool>(t, "restart_on_crash",
                                                   out.restart_on_crash);
    out.max_restart_attempts = toml::find_or<int>(t, "max_restart_attempts",
                                                  out.max_restart_attempts);
    out.data_dir = find_str(t, "data_dir", out.data_dir);

    out.tool_max_rounds = toml::find_or<int>(t, "tool_max_rounds", out.tool_max_rounds);

    out.context_split_enabled             = toml::find_or<bool>(t,        "context_split_enabled",
                                                                out.context_split_enabled);
    out.context_split_strategy            = toml::find_or<std::string>(t, "context_split_strategy",
                                                                out.context_split_strategy);
    out.context_split_trigger_fraction    = toml::find_or<double>(t,      "context_split_trigger_fraction",
                                                                static_cast<double>(out.context_split_trigger_fraction));
    out.context_split_truncate_fraction   = toml::find_or<double>(t,      "context_split_truncate_fraction",
                                                                static_cast<double>(out.context_split_truncate_fraction));
    out.context_split_auto_chunk_fraction = toml::find_or<double>(t,      "context_split_auto_chunk_fraction",
                                                                static_cast<double>(out.context_split_auto_chunk_fraction));
    out.context_split_chunk_tokens        = toml::find_or<int>(t,         "context_split_chunk_tokens",
                                                                out.context_split_chunk_tokens);
    out.context_split_max_chunks          = toml::find_or<int>(t,         "context_split_max_chunks",
                                                                out.context_split_max_chunks);
    out.context_split_summary_tokens      = toml::find_or<int>(t,         "context_split_summary_tokens",
                                                                out.context_split_summary_tokens);

    // Parse [plugins.settings] into a flat string-to-string map.
    //
    // TOML expands dotted keys like
    //     [plugins.settings]
    //     rlm_generator.chunk_size_tokens = "512"
    // into a sub-table:
    //     plugins.settings -> { rlm_generator -> { chunk_size_tokens = "512" } }
    // The previous implementation iterated only the top-level entries and
    // silently dropped any sub-table value, so plugins read their
    // compiled-in defaults instead of the per-plugin overrides the user
    // configured.  We now walk the table recursively, joining keys with '.'
    // so the trampoline's "plugin_id.key" lookup hits the correct entry.
    if (t.contains("settings")) {
        const auto& st = toml::find(t, "settings");
        if (st.is_table()) {
            // Recursive flattener: for a leaf scalar, store
            // `out.settings[prefix + key] = stringified value`.  For nested
            // tables, recurse with `prefix + key + "."`.  Arrays of tables
            // (legal in TOML but meaningless for plugin config) are skipped.
            std::function<void(const std::string&, const decltype(st)&)> walk;
            walk = [&](const std::string& prefix, const decltype(st)& node) {
                if (!node.is_table()) return;
                for (const auto& [k, v] : node.as_table()) {
                    const std::string full = prefix.empty() ? k
                                                            : (prefix + "." + k);
                    if (v.is_string()) {
                        out.settings[full] = s(v.as_string());
                    } else if (v.is_integer()) {
                        out.settings[full] = std::to_string(v.as_integer());
                    } else if (v.is_boolean()) {
                        out.settings[full] = v.as_boolean() ? "true" : "false";
                    } else if (v.is_floating()) {
                        out.settings[full] = std::to_string(v.as_floating());
                    } else if (v.is_table()) {
                        walk(full, v);
                    }
                    // Arrays / array-of-tables are intentionally not surfaced
                    // — plugin settings are a flat string map by design.
                }
            };
            walk("", st);
        }
    }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

TrueLLMConfig load(const std::string& path, const std::string& profile_override)
{
    TrueLLMConfig cfg; // all fields start at compiled-in defaults

    // 1. Apply profile preset first (TOML values will override)
    if (!profile_override.empty()) {
        apply_profile(cfg, profile_override);
    }

    // 2. Parse TOML
    spdlog::debug("[config] Loading: {}", path);
    const toml::value root = toml::parse(path);

    // 3. Apply file-level profile key (only if not overridden by CLI)
    if (profile_override.empty() && root.contains("profile")) {
        const auto file_profile = toml::find<std::string>(root, "profile");
        if (!file_profile.empty()) {
            apply_profile(cfg, file_profile);
        }
    }

    // 4. Overlay all sections from the TOML file
    parse_server(root,    cfg.server);
    parse_model(root,     cfg.model);
    parse_inference(root, cfg.inference);
    parse_sampling(root,  cfg.sampling);
    parse_logging(root,   cfg.logging);
    parse_hardware(root,  cfg.hardware);
    parse_runtime(root,   cfg.runtime);
    parse_plugins(root,   cfg.plugins);

    // 5. Validate
    const std::string err = validate(cfg);
    if (!err.empty()) {
        throw std::runtime_error(fmt::format("[config] Validation error: {}", err));
    }

    spdlog::info("[config] Loaded from {}", path);
    return cfg;
}

std::string validate(const TrueLLMConfig& cfg)
{
    if (cfg.server.port == 0)
        return "server.port must be 1–65535";

    if (cfg.server.request_timeout_ms < 0)
        return "server.request_timeout_ms must be >= 0";

    if (cfg.server.max_concurrent < 1)
        return "server.max_concurrent must be >= 1";

    if (cfg.inference.context_size < 0)
        return "inference.context_size must be >= 0";

    if (cfg.inference.batch_size < 1)
        return "inference.batch_size must be >= 1";

    const auto& backend = cfg.inference.backend;
    if (backend != "cpu" && backend != "ggml" && backend != "cuda")
        return fmt::format("inference.backend '{}' is not recognised (cpu|ggml|cuda)", backend);

    const auto& kv_k = cfg.inference.kv_cache.type_k;
    if (kv_k != "f16" && kv_k != "q8_0" && kv_k != "q4_0")
        return fmt::format("inference.kv_cache.type_k '{}' is not recognised (f16|q8_0|q4_0)", kv_k);

    const auto& kv_v = cfg.inference.kv_cache.type_v;
    if (kv_v != "f16" && kv_v != "q8_0" && kv_v != "q4_0")
        return fmt::format("inference.kv_cache.type_v '{}' is not recognised (f16|q8_0|q4_0)", kv_v);

    if (cfg.sampling.temperature < 0.0f)
        return "sampling.temperature must be >= 0.0";

    if (cfg.sampling.top_p < 0.0f || cfg.sampling.top_p > 1.0f)
        return "sampling.top_p must be in [0.0, 1.0]";

    if (cfg.sampling.max_tokens < 1)
        return "sampling.max_tokens must be >= 1";

    const auto& log_level = cfg.logging.level;
    if (log_level != "trace" && log_level != "debug" && log_level != "info" &&
        log_level != "warn"  && log_level != "error")
        return fmt::format("logging.level '{}' is not recognised (trace|debug|info|warn|error)",
                           log_level);

    const auto& log_fmt = cfg.logging.format;
    if (log_fmt != "text" && log_fmt != "json")
        return fmt::format("logging.format '{}' is not recognised (text|json)", log_fmt);

    const auto& gp = cfg.hardware.guardrails.preset;
    if (gp != "off" && gp != "relaxed" && gp != "balanced" &&
        gp != "strict" && gp != "custom")
        return fmt::format("hardware.guardrails.preset '{}' is not recognised "
                           "(off|relaxed|balanced|strict|custom)", gp);

    if (cfg.hardware.offload.vram_budget_fraction < 0.0f ||
        cfg.hardware.offload.vram_budget_fraction > 1.0f)
        return "hardware.offload.vram_budget_fraction must be in [0.0, 1.0]";

    const auto& eng = cfg.runtime.engine;
    if (eng != "cpu-llama-cpp"  && eng != "cuda12-llama-cpp" &&
        eng != "cuda-llama-cpp" && eng != "vulkan-llama-cpp")
        return fmt::format("runtime.engine '{}' is not recognised "
                           "(cpu-llama-cpp|cuda12-llama-cpp|cuda-llama-cpp|vulkan-llama-cpp)", eng);

    const auto& ch = cfg.runtime.update_channel;
    if (ch != "stable" && ch != "nightly")
        return fmt::format("runtime.update_channel '{}' is not recognised (stable|nightly)", ch);

    return ""; // OK
}

} // namespace truellm::config
