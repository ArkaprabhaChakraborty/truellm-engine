// ---------------------------------------------------------------------------
// engine_factory.cpp — Config → correct EngineInterface implementation
//
// "cpu" | "ggml"  →  GgmlEngine   (always available; GPU offload via n_gpu_layers)
// "cuda"          →  CudaEngine   (paged attention, vllm kernels)
//                    when TRUELLM_HAS_CUDA && TRUELLM_BUILD_CUDA_ENGINE
//                    otherwise falls back to GgmlEngine with a warning.
// ---------------------------------------------------------------------------

#include "engine_interface.h"
#include "backends/cpu/ggml_engine.h"

#if defined(TRUELLM_HAS_CUDA) && defined(TRUELLM_BUILD_CUDA_ENGINE)
#  include "backends/cuda/cuda_engine.h"
#endif

#include <fmt/core.h>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace truellm {

std::unique_ptr<EngineInterface> create_engine(
    const InferenceConfig& inf_cfg,
    const HardwareConfig&  hw_cfg,
    const SamplingConfig&  smp_cfg)
{
    const std::string& backend = inf_cfg.backend;
    spdlog::info("[engine_factory] Requested backend: {}", backend);

    if (backend == "cpu" || backend == "ggml") {
        spdlog::info("[engine_factory] Creating GgmlEngine (backend={})", backend);
        return std::make_unique<GgmlEngine>(inf_cfg, hw_cfg, smp_cfg);
    }

    if (backend == "cuda") {
#if defined(TRUELLM_HAS_CUDA) && defined(TRUELLM_BUILD_CUDA_ENGINE)
        spdlog::info("[engine_factory] Creating CudaEngine (paged attention)");
        return std::make_unique<CudaEngine>(inf_cfg, hw_cfg, smp_cfg);
#else
        spdlog::warn("[engine_factory] backend=cuda requested but CudaEngine is not "
                     "compiled in. Falling back to GgmlEngine with GPU offload. "
                     "Rebuild with -DTRUELLM_BUILD_CUDA_ENGINE=ON to enable.");
        return std::make_unique<GgmlEngine>(inf_cfg, hw_cfg, smp_cfg);
#endif
    }

    throw std::runtime_error(
        fmt::format("Unknown backend '{}'. Supported: cpu, ggml, cuda", backend));
}

} // namespace truellm
