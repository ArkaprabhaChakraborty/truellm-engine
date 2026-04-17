// ---------------------------------------------------------------------------
// application.cpp
// ---------------------------------------------------------------------------

#include "application.h"

#include "../inference/engine_interface.h"
#include "../server/http_server.h"
#include "../plugin_bridge/plugin_bridge.h"

#include <spdlog/spdlog.h>
#include <fmt/core.h>

#include <atomic>
#include <chrono>

namespace truellm {

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------
Application::Application(TrueLLMConfig cfg)
    : cfg_(std::move(cfg))
{}

Application::~Application() = default;

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------
ErrorCode Application::init()
{
    // ---- Startup config banner (debug level) ----
    spdlog::info("[Application] Initialising — backend={} engine={}",
                 cfg_.inference.backend, cfg_.runtime.engine);
    spdlog::debug("[Application] ─── Configuration ────────────────────────────");
    spdlog::debug("[Application]  server:     {}:{}", cfg_.server.host, cfg_.server.port);
    spdlog::debug("[Application]  model:      {}",
                  cfg_.model.alias.empty() ? cfg_.model.path : cfg_.model.alias);
    spdlog::debug("[Application]  model path: {}", cfg_.model.path);
    spdlog::debug("[Application]  template:   {}", cfg_.model.chat_template);
    spdlog::debug("[Application]  backend:    {}", cfg_.inference.backend);
    spdlog::debug("[Application]  ctx_size:   {}", cfg_.inference.context_size);
    spdlog::debug("[Application]  batch:      {} / ubatch: {}",
                  cfg_.inference.batch_size, cfg_.inference.ubatch_size);
    spdlog::debug("[Application]  threads:    {}",
                  cfg_.inference.threads == 0 ? std::string("auto")
                                              : std::to_string(cfg_.inference.threads));
    spdlog::debug("[Application]  flash_attn: {}", cfg_.inference.flash_attention);
    spdlog::debug("[Application]  kv_cache:   K={} V={}",
                  cfg_.inference.kv_cache.type_k, cfg_.inference.kv_cache.type_v);
    spdlog::debug("[Application]  gpu:        device={} layers={} vram_frac={:.0f}%",
                  cfg_.hardware.gpu.device_id, cfg_.hardware.offload.gpu_layers,
                  cfg_.hardware.offload.vram_budget_fraction * 100.f);
    spdlog::debug("[Application]  sampling:   temp={:.2f} top_p={:.2f} top_k={} "
                  "max_tokens={} seed={}",
                  cfg_.sampling.temperature, cfg_.sampling.top_p, cfg_.sampling.top_k,
                  cfg_.sampling.max_tokens, cfg_.sampling.seed);
    spdlog::debug("[Application] ────────────────────────────────────────────────");

    // ---- Create engine ----
    try {
        engine_ = create_engine(cfg_.inference, cfg_.hardware, cfg_.sampling);
    } catch (const std::exception& ex) {
        spdlog::error("[Application] Failed to create engine: {}", ex.what());
        return ErrorCode::InternalError;
    }

    // ---- Load model if path is specified ----
    if (!cfg_.model.path.empty()) {
        spdlog::info("[Application] Loading model: {}", cfg_.model.path);
        auto t0 = std::chrono::steady_clock::now();
        ErrorCode ec = engine_->load_model(cfg_.model.path);
        if (ec != ErrorCode::Ok) {
            spdlog::error("[Application] Failed to load model '{}' (ec={})",
                          cfg_.model.path, static_cast<int>(ec));
            return ec;
        }
        float elapsed_s = static_cast<float>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count()) / 1000.f;

        auto info = engine_->get_model_info();
        spdlog::info("[Application] Model loaded: {} ({:.1f}s)  ctx={} vocab={} layers={}",
                     info.model_id, elapsed_s, info.context_length,
                     info.vocab_size, info.n_layers);

        auto gpu = engine_->get_gpu_stats();
        if (gpu.device_id >= 0 && gpu.vram_total_bytes > 0) {
            spdlog::info("[Application] GPU: {}  VRAM {:.0f}/{:.0f} MB  layers_on_gpu={}",
                         gpu.device_name,
                         (gpu.vram_total_bytes - gpu.vram_free_bytes) / (1024.0 * 1024.0),
                         gpu.vram_total_bytes / (1024.0 * 1024.0),
                         gpu.layers_on_gpu);
        }
    } else {
        spdlog::warn("[Application] No model path configured — engine is idle");
    }

    // ---- Create and initialise plugin bridge ----
    plugin_bridge_ = std::make_unique<PluginBridge>(cfg_);
    {
        ErrorCode pec = plugin_bridge_->init(engine_.get());
        if (pec != ErrorCode::Ok) {
            // Plugin errors are non-fatal — server starts without plugins.
            spdlog::warn("[Application] Plugin bridge init had errors (ec={}); "
                         "continuing without full plugin support",
                         static_cast<int>(pec));
        }
    }

    // ---- Create HTTP server ----
    http_ = std::make_unique<HttpServer>(cfg_, engine_.get());
    http_->set_plugin_bridge(plugin_bridge_.get());
    http_->setup_routes();

    spdlog::info("[Application] Ready — listening on {}:{}",
                 cfg_.server.host, cfg_.server.port);
    return ErrorCode::Ok;
}

// ---------------------------------------------------------------------------
// run / request_shutdown
// ---------------------------------------------------------------------------
void Application::run()
{
    if (!http_) {
        spdlog::error("[Application] run() called before init()");
        return;
    }
    http_->listen(); // blocks until stop() is called
}

void Application::request_shutdown()
{
    if (http_) http_->stop();
}

} // namespace truellm
