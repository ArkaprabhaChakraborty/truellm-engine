// ---------------------------------------------------------------------------
// gpu_device.cpp
// ---------------------------------------------------------------------------

#include "gpu_device.h"

#include <spdlog/spdlog.h>

// Include ggml-cuda only when the CUDA backend is compiled in.
// TRUELLM_HAS_CUDA is defined via target_compile_definitions in CMakeLists.
#if defined(TRUELLM_HAS_CUDA)
#  include <ggml-cuda.h>
#endif

namespace truellm {

int GpuDevice::device_count()
{
#if defined(TRUELLM_HAS_CUDA)
    return ggml_backend_cuda_get_device_count();
#else
    return 0;
#endif
}

GpuDeviceInfo GpuDevice::query(int device_id)
{
    GpuDeviceInfo info{};
    info.device_id = device_id;

#if defined(TRUELLM_HAS_CUDA)
    if (device_id < 0 || device_id >= ggml_backend_cuda_get_device_count()) {
        spdlog::warn("[GpuDevice] device_id {} out of range (count={})",
                     device_id, ggml_backend_cuda_get_device_count());
        info.device_id = -1;
        return info;
    }

    char name_buf[256] = {};
    ggml_backend_cuda_get_device_description(device_id, name_buf, sizeof(name_buf));
    info.name = name_buf;

    size_t free_bytes  = 0;
    size_t total_bytes = 0;
    ggml_backend_cuda_get_device_memory(device_id, &free_bytes, &total_bytes);
    info.vram_free_bytes  = static_cast<int64_t>(free_bytes);
    info.vram_total_bytes = static_cast<int64_t>(total_bytes);
#else
    (void)device_id;
#endif

    return info;
}

int GpuDevice::estimate_gpu_layers(int     device_id,
                                   int64_t layer_size_bytes,
                                   int64_t kv_budget_bytes,
                                   float   vram_budget_fraction)
{
#if defined(TRUELLM_HAS_CUDA)
    if (layer_size_bytes <= 0) return -1;

    auto info = query(device_id);
    if (info.vram_total_bytes == 0) return -1;

    int64_t budget = static_cast<int64_t>(
        static_cast<float>(info.vram_free_bytes) * vram_budget_fraction);
    budget -= kv_budget_bytes;
    if (budget <= 0) return 0;

    return static_cast<int>(budget / layer_size_bytes);
#else
    (void)device_id; (void)layer_size_bytes;
    (void)kv_budget_bytes; (void)vram_budget_fraction;
    return -1;
#endif
}

} // namespace truellm
