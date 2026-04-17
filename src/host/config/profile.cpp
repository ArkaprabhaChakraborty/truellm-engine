// ---------------------------------------------------------------------------
// profile.cpp — Named configuration profile presets
// ---------------------------------------------------------------------------

#include "profile.h"

#include <spdlog/spdlog.h>

namespace truellm::config {

namespace {

// ---------------------------------------------------------------------------
// local-cpu — Laptop / single-user machine, CPU-only inference
// ---------------------------------------------------------------------------
void apply_local_cpu(TrueLLMConfig& cfg)
{
    // Server: single-user, no auth needed
    cfg.server.max_concurrent = 2;

    // Inference: conservative context, auto threads
    cfg.inference.backend      = "cpu";
    cfg.inference.context_size = 4096;
    cfg.inference.threads      = 0;    // auto
    cfg.inference.mmap         = true;
    cfg.inference.mlock        = false;

    // Sampling: sensible interactive defaults
    cfg.sampling.max_tokens = 1024;

    // Hardware: guardrails protect the user from OOM
    cfg.hardware.guardrails.preset = "balanced";

    // Runtime: CPU-only pack
    cfg.runtime.engine = "cpu-llama-cpp";

    // Logging: human-readable
    cfg.logging.level  = "info";
    cfg.logging.format = "text";
}

// ---------------------------------------------------------------------------
// workstation — Desktop with a discrete GPU (e.g. RTX 5090)
// ---------------------------------------------------------------------------
void apply_workstation(TrueLLMConfig& cfg)
{
    cfg.server.max_concurrent = 4;

    cfg.inference.backend      = "cuda";
    cfg.inference.context_size = 8192;
    cfg.inference.threads      = 0;
    cfg.inference.flash_attention = true;

    cfg.hardware.gpu.enabled        = true;
    cfg.hardware.offload.use_gpu_vram   = true;
    cfg.hardware.offload.use_system_ram = true;
    cfg.hardware.offload.gpu_layers     = -1;  // fill VRAM
    cfg.hardware.guardrails.preset  = "balanced";

    cfg.runtime.engine = "cuda12-llama-cpp";

    cfg.logging.level  = "info";
    cfg.logging.format = "text";
}

// ---------------------------------------------------------------------------
// server — Production headless daemon
// ---------------------------------------------------------------------------
void apply_server(TrueLLMConfig& cfg)
{
    // Bind on all interfaces, require an API key
    cfg.server.host           = "0.0.0.0";
    cfg.server.max_concurrent = 8;
    // api_key intentionally left for user to set via ENV

    cfg.inference.backend         = "cuda";
    cfg.inference.context_size    = 32768;
    cfg.inference.threads         = 0;
    cfg.inference.flash_attention = true;
    cfg.inference.mlock           = true;  // prevent swap on production

    cfg.hardware.guardrails.preset = "strict";

    cfg.runtime.engine = "cuda12-llama-cpp";

    // JSON logging for log aggregators
    cfg.logging.level      = "warn";
    cfg.logging.format     = "json";
    cfg.logging.timestamps = true;
}

} // anonymous namespace

// ---------------------------------------------------------------------------

void apply_profile(TrueLLMConfig& cfg, const std::string& preset_name)
{
    if (preset_name == "local-cpu") {
        spdlog::info("[config::profile] Applying preset: local-cpu");
        apply_local_cpu(cfg);
    } else if (preset_name == "workstation") {
        spdlog::info("[config::profile] Applying preset: workstation");
        apply_workstation(cfg);
    } else if (preset_name == "server") {
        spdlog::info("[config::profile] Applying preset: server");
        apply_server(cfg);
    } else {
        spdlog::warn("[config::profile] Unknown profile preset '{}' — ignored", preset_name);
    }
    cfg.profile = preset_name;
}

bool is_known_profile(const std::string& name)
{
    return name == "local-cpu" || name == "workstation" || name == "server";
}

} // namespace truellm::config
