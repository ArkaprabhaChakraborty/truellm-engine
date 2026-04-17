#pragma once
// ---------------------------------------------------------------------------
// cuda_model.h — Weight loader and device tensor map for CudaEngine
//
// CudaModel wraps GgmlModel for GGUF loading but does NOT create a
// llama_context (which would allocate llama.cpp's own KV cache).
// Instead, it builds a flat map of weight tensors on the device:
//
//   weight_map_[layer_idx][tensor_name] → TrueBuffer
//
// The TrueBuffer wraps the device pointer returned by ggml's backend
// buffer API plus the tensor's shape/stride metadata.  No data is copied:
// the pointers refer directly into llama.cpp's device allocation.
//
// Also precomputes the RoPE frequency table (cos/sin) on the device.
// ---------------------------------------------------------------------------

#include <truellm/types.h>
#include <truellm/config_schema.h>
#include "../engine_interface.h"   // for ModelInfo

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// Forward-declare llama.cpp / ggml types so callers don't need those headers
struct llama_model;
struct llama_vocab;
struct ggml_tensor;
struct ggml_context;
struct ggml_backend;
typedef struct ggml_backend* ggml_backend_t;

#ifdef TRUELLM_HAS_CUDA
#  include <cuda_runtime.h>
#endif

namespace truellm {

// Per-layer weight map: tensor_name → TrueBuffer (device ptr)
using LayerWeights = std::unordered_map<std::string, TrueBuffer>;

class CudaModel {
public:
    explicit CudaModel(const InferenceConfig& inf_cfg,
                       const HardwareConfig&  hw_cfg);
    ~CudaModel();

    CudaModel(const CudaModel&)            = delete;
    CudaModel& operator=(const CudaModel&) = delete;

    // -----------------------------------------------------------------------
    // Lifecycle
    // -----------------------------------------------------------------------
    ErrorCode load(const std::string& path);
    void      unload();
    bool      is_loaded() const { return model_ != nullptr; }

    // -----------------------------------------------------------------------
    // Model metadata
    // -----------------------------------------------------------------------
    ModelInfo   get_model_info() const;
    int32_t     n_layers()       const { return n_layers_; }
    int32_t     n_heads()        const { return n_heads_; }
    int32_t     n_kv_heads()     const { return n_kv_heads_; }
    int32_t     head_dim()       const { return head_dim_; }
    int32_t     hidden_size()    const { return hidden_size_; }
    int32_t     ffn_hidden_size()const { return ffn_hidden_size_; }
    int64_t     n_ctx_train()    const { return n_ctx_train_; }
    int64_t     vocab_size()     const { return vocab_size_; }
    int64_t     layer_size_bytes_estimate() const;

    // Hyperparameters read from GGUF metadata at load time
    float       rope_freq_base() const { return rope_freq_base_; }
    float       rms_norm_eps()   const { return rms_norm_eps_; }

    // -----------------------------------------------------------------------
    // Weight access
    // -----------------------------------------------------------------------

    // Per-layer weights as TrueBuffer (metadata + device ptr)
    const LayerWeights& layer_weights(int layer_idx) const;

    // Global weights (embedding, final norm, lm_head)
    const TrueBuffer& global_weight(const std::string& name) const;

    // Raw ggml_tensor* for use in ggml compute graphs.
    // Returns nullptr if the tensor was not found at load time.
    using GgmlLayerWeights = std::unordered_map<std::string, ggml_tensor*>;
    const GgmlLayerWeights& layer_tensors(int layer_idx) const;
    ggml_tensor* global_tensor(const std::string& name) const;

    // ggml CUDA backend (device owns all weight tensors)
    ggml_backend_t cuda_backend() const { return cuda_backend_; }

    // -----------------------------------------------------------------------
    // RoPE frequency table (device)
    // freqs_cos / freqs_sin: [max_seq_len, head_dim/2]
    // -----------------------------------------------------------------------
    const float* freqs_cos_dev() const { return freqs_cos_dev_; }
    const float* freqs_sin_dev() const { return freqs_sin_dev_; }

    // -----------------------------------------------------------------------
    // Tokenizer (delegated to llama.cpp)
    // -----------------------------------------------------------------------
    const llama_vocab*        vocab()      const;
    std::vector<int32_t>      tokenize(const std::string& text, bool add_bos) const;
    std::string               detokenize(const std::vector<int32_t>& tokens)  const;

    // -----------------------------------------------------------------------
    // Raw llama_model pointer (needed for llama_sampler_chain construction)
    // -----------------------------------------------------------------------
    llama_model* raw() const { return model_; }

private:
    InferenceConfig inf_cfg_;
    HardwareConfig  hw_cfg_;

    llama_model* model_ = nullptr;

    // Cached dimensions
    int32_t n_layers_         = 0;
    int32_t n_heads_          = 0;
    int32_t n_kv_heads_       = 0;
    int32_t head_dim_         = 0;
    int32_t hidden_size_      = 0;
    int32_t ffn_hidden_size_  = 0;
    int64_t n_ctx_train_      = 0;
    int64_t vocab_size_       = 0;

    // Hyperparameters from GGUF metadata
    float   rope_freq_base_   = 10000.f;
    float   rms_norm_eps_     = 1e-5f;

    // TrueBuffer weight maps (shape / dtype metadata + device ptr)
    std::vector<LayerWeights>                   layer_weights_;
    std::unordered_map<std::string, TrueBuffer> global_weights_;

    // Raw ggml_tensor* maps — used by CudaEngine to build ggml compute graphs
    std::vector<GgmlLayerWeights>               layer_tensors_;
    std::unordered_map<std::string, ggml_tensor*> global_tensors_;

    // ggml CUDA backend handle (owned — created at load, destroyed at unload)
    ggml_backend_t cuda_backend_ = nullptr;

    // RoPE precomputed tables on device
    float* freqs_cos_dev_ = nullptr;
    float* freqs_sin_dev_ = nullptr;

    // Internal helpers
    void build_weight_map();
    void precompute_rope();
    void extract_dimensions();
};

} // namespace truellm
