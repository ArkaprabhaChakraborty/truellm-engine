#pragma once

// ---------------------------------------------------------------------------
// kernel.h — Kernel registry ABI stub (Part 4 reserved)
//
// In Part 4, third-party CUDA kernels can register themselves with TrueLLM
// via this ABI.  The struct is defined here so the include path is stable;
// no implementation exists yet.
//
// A native plugin that provides a custom kernel implements:
//   truellm_kernel_manifest_t* truellm_kernel_init();
// and links against the CUDA backend.
// ---------------------------------------------------------------------------

#include <truellm/types.h>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Bump when the binary layout of truellm_kernel_manifest_t changes.
#define TRUELLM_KERNEL_ABI_VERSION 1

// ---------------------------------------------------------------------------
// Kernel category tags
// ---------------------------------------------------------------------------
typedef enum truellm_kernel_category {
    TRUELLM_KERNEL_ATTENTION    = 0,  // Replaces paged_attention_v1/v2
    TRUELLM_KERNEL_ACTIVATION   = 1,  // Replaces silu_and_mul / gelu_and_mul
    TRUELLM_KERNEL_LAYERNORM    = 2,  // Replaces rms_norm
    TRUELLM_KERNEL_GEMM         = 3,  // Replaces cuBLAS projection calls
    TRUELLM_KERNEL_CACHE        = 4,  // Replaces reshape_and_cache / copy_blocks
    TRUELLM_KERNEL_CUSTOM       = 99,
} truellm_kernel_category_t;

// ---------------------------------------------------------------------------
// truellm_kernel_manifest_t — the ABI contract each kernel plugin exports
// ---------------------------------------------------------------------------
typedef struct truellm_kernel_manifest {
    uint32_t                abi_version;    // must equal TRUELLM_KERNEL_ABI_VERSION
    const char*             name;           // e.g. "flash_attention_3"
    const char*             version;        // semver string
    truellm_kernel_category_t category;
    uint32_t                min_sm_major;   // minimum compute capability
    uint32_t                min_sm_minor;

    // Function pointer called once when the kernel is selected.
    // Returns 0 on success.
    int (*initialize)(void* engine_ctx);

    // Function pointer called on engine shutdown.
    void (*shutdown)(void* engine_ctx);

    // Opaque pointer to kernel-private state (set by the kernel, read-only to engine).
    void* kernel_state;
} truellm_kernel_manifest_t;

#ifdef __cplusplus
} // extern "C"
#endif
