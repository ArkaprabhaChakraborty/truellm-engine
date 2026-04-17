// ---------------------------------------------------------------------------
// ggml_model.cpp
// ---------------------------------------------------------------------------

#include "ggml_model.h"
#include "gpu/gpu_device.h"

#include <llama.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <numeric>

namespace truellm {

GgmlModel::GgmlModel(const InferenceConfig& inf_cfg, const HardwareConfig& hw_cfg)
    : inf_cfg_(inf_cfg), hw_cfg_(hw_cfg)
{}

GgmlModel::~GgmlModel()
{
    unload();
}

ErrorCode GgmlModel::load(const std::string& path)
{
    if (path.empty()) {
        spdlog::error("[GgmlModel] Empty model path");
        return ErrorCode::InvalidArgument;
    }

    unload(); // safety — free any previously loaded model

    llama_model_params mp = llama_model_default_params();

    // GPU offload: -1 in config means "all layers" → use 999 (llama.cpp convention)
    const int cfg_layers = hw_cfg_.offload.gpu_layers;
    mp.n_gpu_layers = (cfg_layers < 0) ? 999 : cfg_layers;

    mp.main_gpu  = hw_cfg_.gpu.device_id;
    mp.use_mmap  = inf_cfg_.mmap;
    mp.use_mlock = inf_cfg_.mlock;

    // tensor_split for multi-GPU (pass nullptr if not set)
    if (!inf_cfg_.cuda.tensor_split.empty()) {
        // tensor_split is a float[128] in llama; values are relative weights,
        // so we normalise them to sum to 1.0 before passing on.
        static float ts_buf[128] = {};
        const std::size_t n = std::min(inf_cfg_.cuda.tensor_split.size(),
                                       static_cast<std::size_t>(128));

        float ts_sum = 0.f;
        for (std::size_t i = 0; i < n; ++i) ts_sum += inf_cfg_.cuda.tensor_split[i];

        if (ts_sum > 0.f) {
            for (std::size_t i = 0; i < n; ++i)
                ts_buf[i] = inf_cfg_.cuda.tensor_split[i] / ts_sum;
            spdlog::info("[GgmlModel] Multi-GPU tensor split ({} devices): {:.2f}:{:.2f}...",
                         n, ts_buf[0], n > 1 ? ts_buf[1] : 0.f);
        } else {
            for (std::size_t i = 0; i < n; ++i) ts_buf[i] = inf_cfg_.cuda.tensor_split[i];
        }
        mp.tensor_split = ts_buf;
    }

    spdlog::info("[GgmlModel] Loading '{}' (gpu_layers={})", path, mp.n_gpu_layers);

    auto t0 = std::chrono::steady_clock::now();
    model_ = llama_model_load_from_file(path.c_str(), mp);
    auto t1 = std::chrono::steady_clock::now();

    if (!model_) {
        spdlog::error("[GgmlModel] Failed to load model '{}'", path);
        return ErrorCode::NotFound;
    }

    float load_ms = static_cast<float>(
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
    spdlog::info("[GgmlModel] Loaded in {:.1f}s  vocab={} ctx_train={}",
                 load_ms / 1000.f, vocab_size(), n_ctx_train());

    // Post-load VRAM budget check — warn if we're over the configured fraction.
    // Uses gpu_memory_utilization (inference-scoped) as the primary budget knob;
    // falls back to vram_budget_fraction (hardware-scoped) when the inference
    // value is at its default of 0 (i.e. not explicitly set).
    // (GpuDevice::query returns zeros in CPU-only builds, so the check is a no-op.)
    if (mp.n_gpu_layers > 0) {
        auto dev_info = GpuDevice::query(hw_cfg_.gpu.device_id);
        if (dev_info.vram_total_bytes > 0) {
            float used_frac = 1.0f - static_cast<float>(dev_info.vram_free_bytes)
                                   / static_cast<float>(dev_info.vram_total_bytes);

            // Prefer the inference-scoped utilization knob; fall back to the
            // hardware offload fraction when unset (default 0).
            float budget = (inf_cfg_.cuda.gpu_memory_utilization > 0.f)
                         ? inf_cfg_.cuda.gpu_memory_utilization
                         : hw_cfg_.offload.vram_budget_fraction;

            spdlog::info("[GgmlModel] VRAM after load: {:.0f} MB free / {:.0f} MB total ({:.1f}% used)",
                         dev_info.vram_free_bytes  / (1024.0 * 1024.0),
                         dev_info.vram_total_bytes / (1024.0 * 1024.0),
                         used_frac * 100.f);
            if (used_frac > budget) {
                // Compute how many layers would safely fit within budget and
                // surface a concrete recommendation to the operator.
                int64_t layer_bytes = layer_size_bytes_estimate();
                int recommended = GpuDevice::estimate_gpu_layers(
                    hw_cfg_.gpu.device_id,
                    layer_bytes,
                    /*kv_budget_bytes=*/0,
                    budget);
                spdlog::warn("[GgmlModel] VRAM usage {:.1f}% exceeds budget {:.0f}%. "
                             "Consider setting gpu_layers={} (or reducing context_size).",
                             used_frac * 100.f, budget * 100.f,
                             (recommended >= 0) ? recommended : 0);
            }
        }
    }

    return ErrorCode::Ok;
}

void GgmlModel::unload()
{
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
}

const llama_vocab* GgmlModel::vocab() const
{
    return model_ ? llama_model_get_vocab(model_) : nullptr;
}

int64_t GgmlModel::n_ctx_train() const
{
    return model_ ? static_cast<int64_t>(llama_model_n_ctx_train(model_)) : 0;
}

int32_t GgmlModel::n_layers() const
{
    return model_ ? llama_model_n_layer(model_) : 0;
}

int64_t GgmlModel::vocab_size() const
{
    return model_ ? static_cast<int64_t>(llama_vocab_n_tokens(llama_model_get_vocab(model_))) : 0;
}

int64_t GgmlModel::layer_size_bytes_estimate() const
{
    if (!model_) return 0;
    int32_t n = llama_model_n_layer(model_);
    if (n <= 0) return 0;
    // llama_model_size returns total weight bytes for all tensors in the model.
    return static_cast<int64_t>(llama_model_size(model_)) / n;
}

} // namespace truellm
