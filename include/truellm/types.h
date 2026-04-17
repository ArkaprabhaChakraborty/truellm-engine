#pragma once

// ---------------------------------------------------------------------------
// types.h — Core types shared across the entire TrueLLM codebase
//
// TrueBuffer:   Universal tensor handle (replaces torch::Tensor in vendored
//               CUDA kernels).  Defined here, implemented in backends.
// Error codes:  Uniform error reporting across engine, server, plugins.
// Hook enums:   The four plugin hook types.
// ---------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>

namespace truellm {

// ---------------------------------------------------------------------------
// Error codes
// ---------------------------------------------------------------------------
enum class ErrorCode : int32_t {
    Ok                  = 0,
    InvalidArgument     = 1,
    NotFound            = 2,
    NotImplemented      = 3,
    OutOfMemory         = 4,
    BackendError        = 5,
    ConfigError         = 6,
    PluginError         = 7,
    IoError             = 8,
    Unavailable         = 9,    // temporarily unavailable (queue full, shutting down)
    InternalError       = 99,
};

// ---------------------------------------------------------------------------
// Hook types — the four plugin categories
// ---------------------------------------------------------------------------
enum class HookType : uint8_t {
    ToolProvider  = 0,  // Registers callable tools (function calling)
    Preprocessor  = 1,  // Transforms prompt before inference
    Generator     = 2,  // Overrides the inference engine entirely
    ConfigSchema  = 3,  // Declares typed configuration fields
};

// ---------------------------------------------------------------------------
// Data types for tensor elements
// ---------------------------------------------------------------------------
enum class DataType : uint8_t {
    F32     = 0,
    F16     = 1,
    BF16    = 2,
    I32     = 3,
    I16     = 4,
    I8      = 5,
    Q8_0    = 10,   // llama.cpp quantized types
    Q4_0    = 11,
    Q4_1    = 12,
    Q5_0    = 13,
    Q5_1    = 14,
    Q4_K_M  = 20,
};

// ---------------------------------------------------------------------------
// TrueBuffer — universal tensor handle
//
// This is the "libtorch firewall".  All vendored CUDA kernels operate on
// TrueBuffer instead of torch::Tensor / at::Tensor.  The buffer does NOT
// own the memory — it is a view.
// ---------------------------------------------------------------------------
struct TrueBuffer {
    void*       data;       // Pointer to raw data (host or device)
    std::size_t size_bytes; // Total allocation in bytes
    DataType    dtype;      // Element type
    int32_t     device_id;  // -1 = host, 0+ = CUDA device ordinal
    int32_t     ndim;       // Number of dimensions (max 4)
    int64_t     shape[4];   // Shape (unused dims = 1)
    int64_t     stride[4];  // Strides in elements (unused = 0)
};

// ---------------------------------------------------------------------------
// KV cache type selection (for TurboQuant and future quant types)
// ---------------------------------------------------------------------------
enum class KVCacheType : uint8_t {
    F16     = 0,
    Q8_0    = 1,
    Q4_0    = 2,
    // Reserved: TQ3 = 10, TQ4 = 11 (TurboQuant — future)
};

} // namespace truellm
