// ---------------------------------------------------------------------------
// native_router.cpp
// ---------------------------------------------------------------------------

#include "native_router.h"
#include "../inference/engine_interface.h"
#include "../plugin_bridge/plugin_bridge.h"

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

} // namespace truellm
