// ---------------------------------------------------------------------------
// main.cpp — TrueLLM server entry point
// ---------------------------------------------------------------------------

#include <truellm/version.h>
#include <truellm/types.h>

#include "config/config.h"
#include "config/profile.h"
#include "application.h"

#include <CLI/CLI.hpp>
#include <fmt/core.h>
#include <spdlog/spdlog.h>

#include <csignal>
#include <cstdlib>
#include <string>

// ---------------------------------------------------------------------------
// Signal handling
// ---------------------------------------------------------------------------
static truellm::Application* g_app = nullptr;

static void signal_handler(int /*signum*/)
{
    if (g_app) g_app->request_shutdown();
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    // ---- CLI ----
    CLI::App app{fmt::format("TrueLLM v{}", truellm::Version::string)};

    std::string config_path     = "configs/server.toml";
    std::string profile_override;
    std::string backend_override;
    std::string model_path;
    std::string host_override;
    int         port_override   = 0;
    bool        verbose         = false;

    app.add_option("-c,--config",  config_path,      "Path to server.toml");
    app.add_option("--profile",    profile_override,  "Config profile preset: local-cpu | workstation | server")
       ->check([](const std::string& v) -> std::string {
           if (truellm::config::is_known_profile(v) || v.empty()) return "";
           return fmt::format("Unknown profile '{}'. Known: local-cpu, workstation, server", v);
       });
    app.add_option("-m,--model",   model_path,        "Path to GGUF model file (overrides config)");
    app.add_option("-b,--backend", backend_override,  "Inference backend: cpu | cuda (overrides config)")
       ->check(CLI::IsMember({"cpu", "ggml", "cuda"}));
    app.add_option("-H,--host",    host_override,     "Listen address (overrides config)");
    app.add_option("-p,--port",    port_override,     "Listen port (overrides config)")
       ->check(CLI::Range(1, 65535));
    app.add_flag("-v,--verbose",   verbose,           "Enable debug logging");

    CLI11_PARSE(app, argc, argv);

    // ---- Early logging ----
    if (verbose) spdlog::set_level(spdlog::level::debug);

    spdlog::info("TrueLLM v{} (ABI {})", truellm::Version::string, truellm::Version::abi);

    // ---- Load config ----
    truellm::TrueLLMConfig cfg;
    try {
        cfg = truellm::config::load(config_path, profile_override);
    } catch (const std::exception& ex) {
        spdlog::error("Failed to load config '{}': {}", config_path, ex.what());
        return EXIT_FAILURE;
    }

    // ---- Apply CLI overrides ----
    if (!backend_override.empty()) cfg.inference.backend = backend_override;
    if (!model_path.empty())       cfg.model.path        = model_path;
    if (!host_override.empty())    cfg.server.host       = host_override;
    if (port_override > 0)         cfg.server.port       = static_cast<uint16_t>(port_override);

    // ---- Configure logging from config ----
    if (!verbose) {
        const auto& lvl = cfg.logging.level;
        if      (lvl == "trace") spdlog::set_level(spdlog::level::trace);
        else if (lvl == "debug") spdlog::set_level(spdlog::level::debug);
        else if (lvl == "warn")  spdlog::set_level(spdlog::level::warn);
        else if (lvl == "error") spdlog::set_level(spdlog::level::err);
        else                     spdlog::set_level(spdlog::level::info);
    }

    spdlog::info("Backend:       {}", cfg.inference.backend);
    spdlog::info("Runtime:       {} ({})", cfg.runtime.engine, cfg.runtime.engine_version);
    spdlog::info("Context:       {} tokens", cfg.inference.context_size);
    spdlog::info("GPU offload:   {} (layers: {})",
                 cfg.hardware.offload.use_gpu_vram ? "enabled" : "disabled",
                 cfg.hardware.offload.gpu_layers);
    spdlog::info("Guardrails:    {}", cfg.hardware.guardrails.preset);
    if (!cfg.model.path.empty())
        spdlog::info("Model:         {}", cfg.model.path);

    // ---- Signals ----
    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);

    // ---- Boot ----
    truellm::Application application(std::move(cfg));
    g_app = &application;

    if (application.init() != truellm::ErrorCode::Ok) {
        spdlog::error("Initialisation failed - exiting");
        return EXIT_FAILURE;
    }

    application.run(); // blocks until shutdown

    spdlog::info("Shutdown complete");
    return EXIT_SUCCESS;
}
