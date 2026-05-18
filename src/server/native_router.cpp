// ---------------------------------------------------------------------------
// native_router.cpp
// ---------------------------------------------------------------------------

#include "native_router.h"
#include "../inference/engine_interface.h"
#include "../plugin_bridge/plugin_bridge.h"
#include "../plugin_bridge/plugin_context.h"
#include "../plugin_bridge/hooks/tool_registry.h"

#include <truellm/version.h>
#include <truellm/config_schema.h>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <chrono>

namespace truellm {

using json = nlohmann::json;

NativeRouter::NativeRouter(const TrueLLMConfig& cfg, EngineInterface* engine)
    : cfg_(cfg), engine_(engine)
{}

void NativeRouter::register_routes(httplib::Server& svr, AuthCheck auth)
{
    svr.Get("/truellm/v1/status",
        [this, auth](const httplib::Request& req, httplib::Response& res) {
            if (!auth(req, res)) return;
            handle_status(req, res);
        });

    svr.Get("/truellm/v1/plugins",
        [this, auth](const httplib::Request& req, httplib::Response& res) {
            if (!auth(req, res)) return;
            handle_plugins(req, res);
        });

    svr.Get("/truellm/v1/rlm/status",
        [this, auth](const httplib::Request& req, httplib::Response& res) {
            if (!auth(req, res)) return;
            handle_rlm_status(req, res);
        });

    svr.Get("/truellm/v1/compression/status",
        [this, auth](const httplib::Request& req, httplib::Response& res) {
            if (!auth(req, res)) return;
            handle_compression_status(req, res);
        });

    svr.Get("/truellm/v1/chat/status",
        [this, auth](const httplib::Request& req, httplib::Response& res) {
            if (!auth(req, res)) return;
            handle_chat_status(req, res);
        });

    svr.Get("/truellm/v1/research/status",
        [this, auth](const httplib::Request& req, httplib::Response& res) {
            if (!auth(req, res)) return;
            handle_research_status(req, res);
        });

    // Debug-only: direct plugin-tool dispatch.  Lets integration tests
    // exercise plugin tool surfaces without depending on a model emitting
    // structured tool_calls.  Gated by `[server] enable_debug_endpoints`
    // in the config so production deployments can keep the route off.
    if (cfg_.server.enable_debug_endpoints) {
        svr.Post("/truellm/v1/tools/dispatch",
            [this, auth](const httplib::Request& req, httplib::Response& res) {
                if (!auth(req, res)) return;
                handle_tools_dispatch(req, res);
            });
        spdlog::info("[Native] debug endpoint /truellm/v1/tools/dispatch is ENABLED");
    }
}

void NativeRouter::handle_status(const httplib::Request&, httplib::Response& res)
{
    spdlog::debug("[Native] GET /truellm/v1/status");
    auto uptime_s = static_cast<int64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start_time_).count());

    bool loaded = engine_ && engine_->is_model_loaded();

    std::string model_id = cfg_.model.alias.empty() ? cfg_.model.path
                                                     : cfg_.model.alias;

    int64_t ctx_used = 0;
    int64_t ctx_max  = cfg_.inference.context_size;
    if (loaded) {
        int64_t budget = engine_->get_token_budget();
        ctx_used = ctx_max - budget;
    }

    json body = {
        {"status",       loaded ? "ready" : "idle"},
        {"version",      truellm::Version::string},
        {"model",        model_id},
        {"backend",      cfg_.inference.backend},
        {"context_used", ctx_used},
        {"context_max",  ctx_max},
        {"uptime_s",     uptime_s}
    };

    // GPU block — only included when a GPU device is configured
    if (engine_) {
        auto gpu = engine_->get_gpu_stats();
        if (gpu.device_id >= 0) {
            body["gpu"] = {
                {"device_id",     gpu.device_id},
                {"device_name",   gpu.device_name},
                {"vram_total_mb", gpu.vram_total_bytes / (1024 * 1024)},
                {"vram_free_mb",  gpu.vram_free_bytes  / (1024 * 1024)},
                {"layers_on_gpu", gpu.layers_on_gpu}
            };
        }

        // KV cache block — only present when CudaEngine is active
        // (block_size_tokens == 0 means GgmlEngine / no paged allocator)
        auto kv = engine_->get_kv_cache_stats();
        if (kv.block_size_tokens > 0) {
            body["kv_cache"] = {
                {"block_size_tokens", kv.block_size_tokens},
                {"blocks_total",      kv.blocks_total},
                {"blocks_used",       kv.blocks_used},
                {"blocks_free",       kv.blocks_free},
                {"utilization_pct",   kv.utilization_pct},
                {"fp8_enabled",       kv.fp8_enabled}
            };
        }
    }

    // Compression pipeline status — included when any stage is enabled.
    {
        const CompressionConfig& cc = cfg_.inference.compression;
        bool any_enabled = cc.presis.enabled || cc.tome.enabled
                        || cc.fastv.enabled  || cc.pyramid_drop.enabled
                        || cc.kv_vq.enabled  || cc.kv_kivi.enabled;
        if (any_enabled) {
            json comp = json::object();
            comp["presis"]       = {{"enabled", cc.presis.enabled},
                                    {"keep_fraction", cc.presis.keep_fraction}};
            comp["tome"]         = {{"enabled", cc.tome.enabled},
                                    {"r", cc.tome.r}};
            comp["fastv"]        = {{"enabled", cc.fastv.enabled},
                                    {"start_layer", cc.fastv.start_layer},
                                    {"keep_ratio", cc.fastv.keep_ratio}};
            comp["pyramid_drop"] = {{"enabled", cc.pyramid_drop.enabled},
                                    {"final_keep_ratio", cc.pyramid_drop.final_keep_ratio}};
            comp["kv_vq"]        = {{"enabled", cc.kv_vq.enabled},
                                    {"codebook_size", cc.kv_vq.codebook_size}};
            comp["kv_kivi"]      = {{"enabled", cc.kv_kivi.enabled},
                                    {"bits", cc.kv_kivi.bits},
                                    {"residual_length", cc.kv_kivi.residual_length}};
            body["compression"]  = comp;
        }
    }

    // Recursive LM status — included when enabled.
    {
        const RlmConfig& rlm = cfg_.inference.rlm;
        if (rlm.enabled) {
            body["rlm"] = {
                {"enabled",             true},
                {"chunk_size",          rlm.chunk_size_tokens},
                {"overlap",             rlm.chunk_overlap_tokens},
                {"summary_tokens",      rlm.summary_tokens_per_chunk},
                {"max_hierarchy_depth", rlm.max_hierarchy_depth}
            };
        }
    }

    spdlog::debug("[Native] status  model={} ctx_used={}/{} uptime={}s",
                  model_id, ctx_used, ctx_max, uptime_s);

    res.set_content(body.dump(), "application/json");
}

// ---------------------------------------------------------------------------
// GET /truellm/v1/plugins
// ---------------------------------------------------------------------------
void NativeRouter::handle_plugins(const httplib::Request&, httplib::Response& res)
{
    spdlog::debug("[Native] GET /truellm/v1/plugins");

    if (!plugin_bridge_) {
        json body = {
            {"plugin_count", 0},
            {"tool_count",   0},
            {"plugins",      json::array()}
        };
        res.set_content(body.dump(), "application/json");
        return;
    }

    res.set_content(plugin_bridge_->get_status().dump(), "application/json");
}

// ---------------------------------------------------------------------------
// GET /truellm/v1/rlm/status
// Returns runtime counters related to RLM / generator dispatch.  Useful for
// observing whether the RLM generator is actually being claimed, how many
// sub-calls the hierarchy issued, and whether depth limits are tripping.
// ---------------------------------------------------------------------------
void NativeRouter::handle_rlm_status(const httplib::Request&,
                                      httplib::Response& res)
{
    spdlog::debug("[Native] GET /truellm/v1/rlm/status");

    const RlmConfig& rlm = cfg_.inference.rlm;
    json body = {
        {"config", {
            {"enabled",             rlm.enabled},
            {"chunk_size_tokens",   rlm.chunk_size_tokens},
            {"chunk_overlap_tokens",rlm.chunk_overlap_tokens},
            {"summary_tokens_per_chunk", rlm.summary_tokens_per_chunk},
            {"max_hierarchy_depth", rlm.max_hierarchy_depth}
        }}
    };

    if (plugin_bridge_) {
        auto c = plugin_bridge_->get_runtime_counters();
        json per_plugin = json::object();
        for (const auto& kv : c.claims_per_plugin)
            per_plugin[kv.first] = kv.second;

        body["runtime"] = {
            {"generator_dispatches_attempted", c.generator_dispatches_attempted},
            {"generator_dispatches_claimed",   c.generator_dispatches_claimed},
            {"generator_dispatches_failed",    c.generator_dispatches_failed},
            {"call_engine_subcalls",           c.call_engine_subcalls},
            {"call_engine_depth_aborts",       c.call_engine_depth_aborts},
            {"max_recursion_depth_reached",    c.max_recursion_depth_reached},
            {"inject_tokens_calls",            c.inject_tokens_calls},
            {"inject_tokens_not_impl",         c.inject_tokens_not_impl},
            {"claims_per_plugin",              per_plugin}
        };
    } else {
        body["runtime"] = {
            {"generator_dispatches_attempted", 0},
            {"generator_dispatches_claimed",   0},
            {"generator_dispatches_failed",    0},
            {"call_engine_subcalls",           0},
            {"call_engine_depth_aborts",       0},
            {"max_recursion_depth_reached",    0},
            {"inject_tokens_calls",            0},
            {"inject_tokens_not_impl",         0},
            {"claims_per_plugin",              json::object()}
        };
    }

    res.set_content(body.dump(), "application/json");
}

// ---------------------------------------------------------------------------
// GET /truellm/v1/compression/status
// Returns the runtime state of each compression stage.  Per-stage counters
// (tokens pruned, chunks encoded) depend on a host->report_metric API that
// has not yet been added; this endpoint surfaces config + engine-observable
// capability flags, which is enough to diagnose "is compression wired up".
// ---------------------------------------------------------------------------
void NativeRouter::handle_compression_status(const httplib::Request&,
                                              httplib::Response& res)
{
    spdlog::debug("[Native] GET /truellm/v1/compression/status");

    const CompressionConfig& cc = cfg_.inference.compression;

    json stages;
    stages["presis"] = {
        {"enabled",          cc.presis.enabled},
        {"threshold",        cc.presis.threshold},
        {"keep_fraction",    cc.presis.keep_fraction},
        {"importance_layers",cc.presis.importance_layers},
        {"fallback_tfidf",   cc.presis.fallback_tfidf}
    };
    stages["tome"] = {
        {"enabled",          cc.tome.enabled},
        {"r",                cc.tome.r},
        {"start_layer",      cc.tome.start_layer},
        {"apply_all_layers", cc.tome.apply_all_layers},
        {"merge_mode",       cc.tome.merge_mode}
    };
    stages["fastv"] = {
        {"enabled",          cc.fastv.enabled},
        {"start_layer",      cc.fastv.start_layer},
        {"keep_ratio",       cc.fastv.keep_ratio},
        {"importance_metric",cc.fastv.importance_metric}
    };
    stages["pyramid_drop"] = {
        {"enabled",                cc.pyramid_drop.enabled},
        {"final_keep_ratio",       cc.pyramid_drop.final_keep_ratio},
        {"drop_schedule",          cc.pyramid_drop.drop_schedule},
        {"warmup_layers",          cc.pyramid_drop.warmup_layers},
        {"protect_last_n_tokens",  cc.pyramid_drop.protect_last_n_tokens}
    };
    stages["kv_kivi"] = {
        {"enabled",          cc.kv_kivi.enabled},
        {"bits",             cc.kv_kivi.bits},
        {"residual_length",  cc.kv_kivi.residual_length},
        {"key_granularity",  cc.kv_kivi.key_granularity},
        {"value_granularity",cc.kv_kivi.value_granularity},
        {"pre_rope_keys",    cc.kv_kivi.pre_rope_keys}
    };
    stages["kv_vq"] = {
        {"enabled",              cc.kv_vq.enabled},
        {"codebook_size",        cc.kv_vq.codebook_size},
        {"residual_depth",       cc.kv_vq.residual_depth},
        {"outlier_tracing",      cc.kv_vq.outlier_tracing},
        {"disable_in_subcalls",  cc.kv_vq.disable_in_subcalls}
    };

    bool any_enabled = cc.presis.enabled || cc.tome.enabled
                    || cc.fastv.enabled  || cc.pyramid_drop.enabled
                    || cc.kv_kivi.enabled|| cc.kv_vq.enabled;

    json body = {
        {"compression_enabled", cc.enabled},
        {"any_stage_enabled",   any_enabled},
        {"pipeline",            cc.pipeline},   // std::string (e.g. "tome,presis")
        {"stages",              stages}
    };

    if (plugin_bridge_) {
        auto rc = plugin_bridge_->get_runtime_counters();
        body["runtime"] = {
            {"generator_dispatches_attempted", rc.generator_dispatches_attempted},
            {"generator_dispatches_claimed",   rc.generator_dispatches_claimed},
            {"call_engine_subcalls",           rc.call_engine_subcalls}
        };
    }

    res.set_content(body.dump(), "application/json");
}

// ---------------------------------------------------------------------------
// GET /truellm/v1/chat/status   (Code_Capability_design.md §4.4)
//
// Surfaces the federated compaction tier's aggregated decisions so operators
// can answer "are my prompts hitting full-compactor?" / "is auto_compactor
// claiming?" without having to grep logs.  Populated by openai_router's
// request-completion hook calling PluginBridge::record_chat_outcome.
// ---------------------------------------------------------------------------
void NativeRouter::handle_chat_status(const httplib::Request&,
                                       httplib::Response& res)
{
    spdlog::debug("[Native] GET /truellm/v1/chat/status");

    json body = json::object();
    body["compaction_loaded"] = false;
    body["totals"] = {
        {"requests",                       0},
        {"with_session_id",                0},
        {"auto_compactor_skipped",         0},
        {"consecutive_compact_failures_max", 0}
    };
    body["phase_counts"]          = json::object();
    body["strategy_counts"]       = json::object();
    body["classification_counts"] = json::object();

    if (plugin_bridge_) {
        auto outcomes = plugin_bridge_->get_chat_outcomes();
        body["compaction_loaded"] = true;
        body["totals"] = {
            {"requests",                       outcomes.total_requests},
            {"with_session_id",                outcomes.with_session_id},
            {"auto_compactor_skipped",         outcomes.auto_compactor_skipped},
            {"consecutive_compact_failures_max",
                                               outcomes.consecutive_compact_failures_max}
        };
        for (const auto& kv : outcomes.phase_counts)
            body["phase_counts"][kv.first] = kv.second;
        for (const auto& kv : outcomes.strategy_counts)
            body["strategy_counts"][kv.first] = kv.second;
        for (const auto& kv : outcomes.classification_counts)
            body["classification_counts"][kv.first] = kv.second;
    }

    res.set_content(body.dump(), "application/json");
}

// ---------------------------------------------------------------------------
// GET /truellm/v1/research/status   (Code_Capability_design.md §9.8)
//
// Surfaces aggregated research-mode counts: total research turns, sub-call
// token usage, and clusters by `effort` and `research.terminal_reason`.
// Empty / zeroed when the research tier is not loaded or no research turn
// has completed.
// ---------------------------------------------------------------------------
void NativeRouter::handle_research_status(const httplib::Request&,
                                           httplib::Response& res)
{
    spdlog::debug("[Native] GET /truellm/v1/research/status");

    json body = json::object();
    body["research_loaded"] = false;
    body["totals"] = {
        {"requests",            0},
        {"subcalls",            0},
        {"prompt_tokens",       0},
        {"completion_tokens",   0}
    };
    body["effort_counts"]          = json::object();
    body["terminal_reason_counts"] = json::object();

    if (plugin_bridge_) {
        auto outcomes = plugin_bridge_->get_research_outcomes();
        body["research_loaded"] = (outcomes.total_requests > 0);
        body["totals"] = {
            {"requests",          outcomes.total_requests},
            {"subcalls",          outcomes.total_subcalls},
            {"prompt_tokens",     outcomes.total_prompt_tokens},
            {"completion_tokens", outcomes.total_completion_tokens}
        };
        for (const auto& kv : outcomes.effort_counts)
            body["effort_counts"][kv.first] = kv.second;
        for (const auto& kv : outcomes.terminal_reason_counts)
            body["terminal_reason_counts"][kv.first] = kv.second;
    }

    res.set_content(body.dump(), "application/json");
}

// ---------------------------------------------------------------------------
// POST /truellm/v1/tools/dispatch   (debug only, gated on
// [server] enable_debug_endpoints = true)
//
// Body: {
//   "qualified_name":   "evidence_store.add",            (required)
//   "args":             { ... },                          (optional, defaults to {})
//   "context_metadata": { "session.id": "...", ... }     (optional)
// }
//
// Response: {
//   "ok":         bool,
//   "error_code": int,        // TRUELLM_ERR_* value
//   "error_msg":  "...",      // empty on success
//   "payload":    "..."       // tool's payload string (typically JSON)
// }
//
// Intended for integration tests that need deterministic invocation of
// plugin tools without depending on a model to emit tool_calls.  Do NOT
// expose to production traffic — it bypasses the openai_router tool
// loop, including the agent.allowed_tools allow-list (though the test
// can opt in by passing the key in context_metadata).
// ---------------------------------------------------------------------------
void NativeRouter::handle_tools_dispatch(const httplib::Request& req,
                                          httplib::Response& res)
{
    spdlog::debug("[Native] POST /truellm/v1/tools/dispatch");

    if (!plugin_bridge_) {
        res.status = 503;
        res.set_content(
            R"({"ok":false,"error_code":3,"error_msg":"plugin_bridge unavailable","payload":""})",
            "application/json");
        return;
    }

    json body = json::parse(req.body, nullptr, false);
    if (body.is_discarded() || !body.is_object()) {
        res.status = 400;
        res.set_content(
            R"({"ok":false,"error_code":1,"error_msg":"invalid JSON body","payload":""})",
            "application/json");
        return;
    }

    const std::string qname = body.value("qualified_name", std::string{});
    if (qname.empty()) {
        res.status = 400;
        res.set_content(
            R"({"ok":false,"error_code":1,"error_msg":"qualified_name required","payload":""})",
            "application/json");
        return;
    }

    std::string args_str = "{}";
    if (body.contains("args")) {
        if (body["args"].is_string()) {
            args_str = body["args"].get<std::string>();
        } else {
            args_str = body["args"].dump();
        }
    }

    // Build a synthetic per-request context.  No engine state is needed
    // here — tools that demand a model fall through to host->call_engine
    // which uses the live engine bound to plugin_bridge.
    auto ctx = std::make_unique<truellm_context_t>();
    ctx->request_id   = "debug-tool-" + std::to_string(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    ctx->model_id     = cfg_.model.alias.empty() ? "default" : cfg_.model.alias;
    ctx->model_arch   = engine_ ? engine_->get_model_info().architecture : "unknown";
    ctx->token_budget = engine_ ? engine_->get_token_budget() : 0;
    ctx->max_tokens   = 512;
    ctx->temperature  = 0.2f;
    ctx->status_sink  = nullptr;

    if (body.contains("context_metadata") && body["context_metadata"].is_object()) {
        std::lock_guard<std::mutex> lk(ctx->metadata_mu);
        for (auto it = body["context_metadata"].begin();
             it != body["context_metadata"].end(); ++it)
        {
            if (it.value().is_string()) {
                ctx->metadata[it.key()] = it.value().get<std::string>();
            } else {
                ctx->metadata[it.key()] = it.value().dump();
            }
        }
    }

    ToolCallResult tc = plugin_bridge_->dispatch_tool(qname, args_str, ctx.get());
    json out = {
        {"ok",         tc.error == TRUELLM_OK},
        {"error_code", static_cast<int>(tc.error)},
        {"error_msg",  tc.error_msg},
        {"payload",    tc.error == TRUELLM_OK ? tc.payload : std::string{}}
    };
    res.set_content(out.dump(), "application/json");
}

} // namespace truellm
