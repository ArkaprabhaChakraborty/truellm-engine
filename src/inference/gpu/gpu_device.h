#pragma once
// ---------------------------------------------------------------------------
// gpu_device.h — Thin wrapper around ggml-cuda.h device queries
//
// All CUDA-specific includes are isolated here.  In CPU-only builds every
// method returns a safe zero/empty value so callers need no #ifdef guards.
// ---------------------------------------------------------------------------

#include <cstdint>
#include <string>

namespace truellm {

struct GpuDeviceInfo {
    int         device_id        = -1;
    std::string name;
    int64_t     vram_total_bytes = 0;
    int64_t     vram_free_bytes  = 0;
};

class GpuDevice {
public:
    // Number of CUDA devices visible to this process.
    // Returns 0 in CPU-only builds.
    static int device_count();

    // Query free/total VRAM for one device.
    // Returns {-1, "", 0, 0} when device_id is invalid or CUDA unavailable.
    static GpuDeviceInfo query(int device_id);

    // Estimate how many transformer layers fit in available VRAM.
    //   layer_size_bytes : average weight bytes per layer (from GGUF metadata)
    //   kv_budget_bytes  : bytes reserved for the KV cache
    //   vram_budget_fraction : fraction of free VRAM we're allowed to use
    // Returns -1 if CUDA is unavailable (caller should default to 0 = CPU).
    static int estimate_gpu_layers(int     device_id,
                                   int64_t layer_size_bytes,
                                   int64_t kv_budget_bytes,
                                   float   vram_budget_fraction = 0.90f);
};

} // namespace truellm
