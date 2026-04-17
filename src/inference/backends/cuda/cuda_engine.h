#pragma once
// ---------------------------------------------------------------------------
// cuda_engine.h — Custom paged-attention CUDA inference backend
//
// Implements EngineInterface using:
//   - CudaModel    : GGUF weight loader, device tensor map, RoPE tables
//   - CudaAllocator: paged KV block pool (non-contiguous, per-sequence)
//   - CudaStream   : prefill / decode / copy CUDA stream pool
//   - MultiGpu     : tensor parallelism coordinator (stub, tp_size=1)
//   - vllm_kernels : paged_attention_v1/v2, cache_kernels, layernorm, etc.
//
// Route through engine_factory.cpp: backend = "cuda" + TRUELLM_HAS_CUDA.
// ---------------------------------------------------------------------------

#include "../engine_interface.h"
#include <truellm/config_schema.h>

#include "cuda_model.h"
#include "cuda_allocator.h"
#include "cuda_stream.h"
#include "multi_gpu.h"

#include <llama.h>      // llama_sampler_chain_*
#include <ggml.h>       // ggml_tensor, ggml_context
#include <ggml-alloc.h> // ggml_gallocr_t
#include <ggml-backend.h>

#include <atomic>
#include <memory>
#include <string>
#include <vector>

namespace truellm {

class CudaEngine : public EngineInterface {
public:
    CudaEngine(const InferenceConfig&   inf_cfg,
               const HardwareConfig&    hw_cfg,
               const SamplingConfig&    smp_cfg);
    ~CudaEngine() override;

    // -----------------------------------------------------------------------
    // EngineInterface
    // -----------------------------------------------------------------------
    ErrorCode load_model(const std::string& model_path) override;
    void      unload_model() override;
    bool      is_model_loaded() const override;

    ModelInfo get_model_info() const override;
    int64_t   get_token_budget() const override;

    GenerateResult generate(const GenerateRequest& request) override;

    std::vector<int32_t> tokenize(const std::string& text) const override;
    std::string          detokenize(const std::vector<int32_t>& tokens) const override;

    GpuStats     get_gpu_stats()     const override;
    KvCacheStats get_kv_cache_stats()const;

    bool supports_embeddings() const override { return false; }

private:
    // Config (owned)
    InferenceConfig  inf_cfg_;
    HardwareConfig   hw_cfg_;
    SamplingConfig   smp_cfg_;

    // Sub-components (created in load_model)
    std::unique_ptr<CudaModel>     model_;
    std::unique_ptr<CudaAllocator> allocator_;
    std::unique_ptr<CudaStream>    streams_;
    std::unique_ptr<MultiGpu>      multi_gpu_;

    // llama.cpp sampler (CPU-side)
    llama_sampler* sampler_ = nullptr;

    // ggml CUDA backend + graph allocator (shared with CudaModel but owned here
    // for compute graphs built in run_forward)
    ggml_gallocr_t gallocr_ = nullptr;

    // Device scratch buffers — allocated once at load_model, 1-token sized.
    // d_attn_out_ : output of paged_attention, input to O-projection
    // d_residual_ : cross-graph residual state (ggml gallocr invalidates
    //               intermediate tensor memory between graph calls, so we
    //               own this buffer explicitly)
    // d_seq_lens_ : single int32 with current sequence length
    void*    d_attn_out_ = nullptr;  // [n_heads * head_dim] F16
    void*    d_residual_ = nullptr;  // [hidden_size]        F16
    int32_t* d_seq_lens_ = nullptr;  // [1]                  I32

    // Pre-allocated paged_attention_v2 temp buffers.
    // Sized at load time for worst-case context (n_ctx_train).
    // Only used when max_context_len / block_size > 32 (v2 path).
    void* d_v2_exp_sums_   = nullptr;  // [n_heads, max_partitions] F32
    void* d_v2_max_logits_ = nullptr;  // [n_heads, max_partitions] F32
    void* d_v2_tmp_out_    = nullptr;  // [n_heads, max_partitions, head_dim] F16
    int   v2_max_partitions_ = 0;

    // Host-side logits buffer — filled by run_forward, read by generate()
    std::vector<float> d_logits_buf_;

    // CUDA graph cache (decode step, one graph per batch size)
    struct GraphEntry { int batch_size; void* graph_exec; };
    std::vector<GraphEntry> cuda_graphs_;

    // Request counter for seq_id generation
    std::atomic<uint64_t> next_seq_id_{1};

    // -----------------------------------------------------------------------
    // Internal — forward pass phases
    // -----------------------------------------------------------------------

    // Run the transformer forward pass for a single token at seq_pos.
    // Used for both prefill (called N times) and decode (called once per step).
    ErrorCode run_forward(
        uint64_t     seq_id,
        int32_t      token_id,    // single input token (host value)
        int          seq_pos,     // absolute position in sequence (0-based)
        cudaStream_t stream);

    // Build sampler chain from smp_cfg_
    void build_sampler();
    void free_sampler();

    // Allocate/free small device scratch buffers (attn_out, seq_lens)
    void alloc_scratch_buffers();
    void free_scratch_buffers();

    // CUDA graph capture (called after model load when !enforce_eager)
    void capture_cuda_graphs();
};

} // namespace truellm
