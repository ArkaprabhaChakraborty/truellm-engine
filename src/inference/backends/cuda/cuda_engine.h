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
// Compression passes (engine-internal, formerly plugin-side):
//   - KvCompressionPass       : post-prefill VQ / KiVi KV cache compression
//   - CompressionScoringPass  : pre-prefill Presis / ToMe token reduction
//
// Route through engine_factory.cpp: backend = "cuda" + TRUELLM_HAS_CUDA.
// ---------------------------------------------------------------------------

#include "../engine_interface.h"
#include <truellm/config_schema.h>

#include "cuda_model.h"
#include "cuda_allocator.h"
#include "cuda_stream.h"
#include "multi_gpu.h"
#include "compression/kv_compression_pass.h"
#include "compression/compression_scoring_pass.h"

#include <cuda_fp16.h>  
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

    // ── api_minor >= 1 — compression / RLM data access ───────────────────────
    // get_hidden_states: returns residual stream at given layer from last forward.
    // Populated in run_forward() when the megakernel saves hidden states.
    // Returns nullptr until Phase M4 wires the megakernel output buffers.
    const void* get_hidden_states(int32_t layer_idx,
                                  int64_t* out_size_bytes) override;
    // get_attention_weights: returns saved attention matrices when
    // megakernel_save_attn=true. Returns nullptr until Phase M4.
    const void* get_attention_weights(int32_t layer_idx,
                                      int64_t* out_size_bytes) override;
    // get_kv_cache_tensor: exposes the CudaAllocator KV block tensor for a
    // given layer. Returns the device pointer (caller must not dereference on
    // CPU). Returns nullptr until Phase M4.
    const void* get_kv_cache_tensor(int32_t layer_idx, int32_t type,
                                    int64_t out_shape[4]) override;

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

    // ── Compression passes (created in load_model when enabled) ─────────────
    std::unique_ptr<KvCompressionPass>      kv_compress_;
    std::unique_ptr<CompressionScoringPass> scoring_;

    // ── Megakernel device buffers (allocated in alloc_megakernel_buffers) ─────
    // Contiguous activation double-buffer: [2 * max_seq_tokens * d_model] F16
    void*   d_mk_act_buf_        = nullptr;
    // Softmax scratch: [n_heads * max_seq_tokens * kv_cache_capacity] F32
    void*   d_mk_softmax_buf_    = nullptr;
    // Input token ids: [max_seq_tokens] I32
    int32_t* d_mk_input_ids_     = nullptr;
    // Argmax output: [1] I32
    int32_t* d_mk_argmax_out_    = nullptr;
    // Attention weights (optional): [n_layers, n_heads, seq, kv_seq] F32
    void*   d_mk_attn_weights_   = nullptr;
    // Hidden states (optional): [n_layers+1, seq, d_model] F16
    void*   d_mk_hidden_states_  = nullptr;
    // FastV/PyramidDrop token mask: [kv_cache_capacity] bool
    bool*   d_mk_attn_mask_      = nullptr;
    // Per-layer contiguous weight pointer arrays (device pointers, host-alloc)
    // These are host arrays of device pointers passed to MegakernelArgs.
    // __half is a CUDA type; guard with TRUELLM_HAS_CUDA so MSVC never sees
    // it in a non-NVCC compilation unit (cuda_fp16.h + /Zc:preprocessor clash).
#ifdef TRUELLM_HAS_CUDA
    std::vector<const __half*> h_wq_ptrs_, h_wk_ptrs_, h_wv_ptrs_, h_wo_ptrs_;
    std::vector<const __half*> h_wgate_ptrs_, h_wup_ptrs_, h_wdown_ptrs_;
    std::vector<const float*>  h_rms_att_ptrs_, h_rms_ffn_ptrs_;
#endif
    // Device arrays of pointers (MegakernelArgs takes const __half* const*)
    void*   d_mk_ptr_arrays_     = nullptr; // packed allocation, see alloc fn
    // KV cache pointer arrays from CudaAllocator (device ptrs per-layer)
#ifdef TRUELLM_HAS_CUDA
    std::vector<__half*>       h_k_cache_ptrs_, h_v_cache_ptrs_;
#endif
    void*   d_mk_kv_ptr_arrays_  = nullptr;
    // Dimensions captured at alloc time for get_hidden_states/get_attn_weights
    int32_t mk_n_layers_   = 0;
    int32_t mk_n_heads_    = 0;
    int32_t mk_seq_cap_    = 0; // max seq tokens for which buffers were allocated
    int32_t mk_kv_cap_     = 0; // kv_cache_capacity

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

    // Allocate/free megakernel device buffers
    void alloc_megakernel_buffers();
    void free_megakernel_buffers();

    // Cooperative megakernel forward pass (Phase M4).
    // Called instead of run_forward() when use_megakernel=true.
    ErrorCode run_forward_megakernel(
        const std::vector<int32_t>& tokens,
        int                         kv_seq_len,
        int                         kv_write_offset,
        cudaStream_t                stream);

    // Post-prefill: run KV compression on all layers after megakernel prefill.
    void run_kv_compression_pass(int seq_len, cudaStream_t stream);

    // Pre-prefill: score and reduce token sequence via Presis / ToMe.
    std::vector<int32_t> run_scoring_pass(const std::vector<int32_t>& tokens,
                                          int kv_seq);

    // CUDA graph capture (called after model load when !enforce_eager)
    void capture_cuda_graphs();
};

} // namespace truellm
