// ---------------------------------------------------------------------------
// cuda_engine.cpp — CudaEngine implementation
//
// Forward pass uses ggml compute graphs for all linear algebra
// (embedding, QKV/FFN projections, RMSNorm, RoPE, SiLU) so that
// quantised weight formats (Q4_K_M, Q8_0, F16, …) work without any
// custom GEMM code.
//
// Paged KV management (block pool, slot mapping, block tables) and the
// paged-attention kernel itself remain the CudaEngine's unique value-add.
//
// Token processing is always 1-token-at-a-time:
//   Prefill  → loop over each prompt token calling run_forward()
//   Decode   → single run_forward() call per generated token
// This is O(n²) but correct for every KV entry and avoids a separate
// batched-prefill flash-attention kernel for now.
// ---------------------------------------------------------------------------

#include "cuda_engine.h"
#include "gpu/gpu_device.h"

#include <spdlog/spdlog.h>
#include <chrono>
#include <cmath>
#include <stdexcept>

#ifdef TRUELLM_HAS_CUDA
#  include <cuda_runtime.h>
#  include <ggml.h>
#  include <ggml-alloc.h>
#  include <ggml-backend.h>
#  include <ggml-cuda.h>
#  include "vllm_types.h"  // brings TrueBuffer + DataType into global scope
#  include <cuda_fp16.h>
// Megakernel cooperative launch (Phase M4).
// Only included when CUDA is available; megakernel.cu is compiled into
// truellm_custom_kernels and linked by the CUDA backend CMakeLists.
#  include "../../../../third_party/custom_kernels/include/megakernel.cuh"
#  include "compression/kv_compression_pass.h"
#  include "compression/compression_scoring_pass.h"

// ---------------------------------------------------------------------------
// Forward-declare the vllm paged-attention kernel launchers
// (implemented in third_party/vllm_kernels/)
// ---------------------------------------------------------------------------
void rms_norm(TrueBuffer& out, const TrueBuffer& x, const TrueBuffer& weight,
              float eps, cudaStream_t stream);
void reshape_and_cache(const TrueBuffer& key, const TrueBuffer& value,
                       TrueBuffer& kv_cache, const TrueBuffer& slot_mapping,
                       cudaStream_t stream);
void paged_attention_v1(TrueBuffer& out, const TrueBuffer& q,
                        const TrueBuffer& k_cache, const TrueBuffer& v_cache,
                        const TrueBuffer& block_tables, const TrueBuffer& seq_lens,
                        int num_kv_heads, float scale,
                        int head_size, int block_size, cudaStream_t stream);
void paged_attention_v2(TrueBuffer& out, TrueBuffer& exp_sums,
                        TrueBuffer& max_logits, TrueBuffer& tmp_out,
                        const TrueBuffer& q, const TrueBuffer& k_cache,
                        const TrueBuffer& v_cache, const TrueBuffer& block_tables,
                        const TrueBuffer& seq_lens, int num_kv_heads, float scale,
                        int head_size, int block_size, int max_context_len,
                        cudaStream_t stream);

#  define CUDA_CHECK(expr)                                                      \
    do {                                                                        \
        cudaError_t _e = (expr);                                                \
        if (_e != cudaSuccess)                                                  \
            throw std::runtime_error(                                           \
                std::string("[CudaEngine] CUDA error: ")                        \
                + cudaGetErrorString(_e));                                      \
    } while (0)

// ---------------------------------------------------------------------------
// ggml graph context — sized for up to 128 nodes per graph build.
// GGML_CTX_BYTES uses runtime functions so must be static const, not constexpr.
// ---------------------------------------------------------------------------
static constexpr size_t GGML_CTX_NODES = 128;
static const     size_t GGML_CTX_BYTES =
    GGML_CTX_NODES * ggml_tensor_overhead()
    + ggml_graph_overhead_custom(GGML_CTX_NODES, false);

// ---------------------------------------------------------------------------
// Tiny TrueBuffer builder from a raw device pointer + shape (F16 / F32)
// ---------------------------------------------------------------------------
static TrueBuffer make_tb(void* ptr, DataType dt, int device_id,
                          int d0, int d1 = 1, int d2 = 1, int d3 = 1)
{
    TrueBuffer b{};
    b.data      = ptr;
    b.dtype     = dt;
    b.device_id = device_id;
    b.ndim      = (d3>1) ? 4 : (d2>1) ? 3 : (d1>1) ? 2 : 1;
    b.shape[0]  = d0; b.shape[1] = d1; b.shape[2] = d2; b.shape[3] = d3;
    return b;
}

#endif // TRUELLM_HAS_CUDA

namespace truellm {

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------
CudaEngine::CudaEngine(const InferenceConfig& inf_cfg,
                       const HardwareConfig&  hw_cfg,
                       const SamplingConfig&  smp_cfg)
    : inf_cfg_(inf_cfg), hw_cfg_(hw_cfg), smp_cfg_(smp_cfg)
{}

CudaEngine::~CudaEngine()
{
    unload_model();
}

// ---------------------------------------------------------------------------
// load_model
// ---------------------------------------------------------------------------
ErrorCode CudaEngine::load_model(const std::string& model_path)
{
#ifndef TRUELLM_HAS_CUDA
    spdlog::error("[CudaEngine] Built without CUDA support — use GgmlEngine");
    return ErrorCode::NotImplemented;
#else
    unload_model();

    // 1. Load weights via CudaModel (llama.cpp GGUF → all layers on GPU)
    model_ = std::make_unique<CudaModel>(inf_cfg_, hw_cfg_);
    ErrorCode ec = model_->load(model_path);
    if (ec != ErrorCode::Ok) { model_.reset(); return ec; }

    const CudaEngineConfig& cec = inf_cfg_.cuda_engine;

    // 2. Validate FP8 requirements
    if (cec.use_fp8_kv) {
        int maj = inf_cfg_.cuda.compute_capability_major;
        int min_cc = inf_cfg_.cuda.compute_capability_minor;
        if (maj < 8 || (maj == 8 && min_cc < 9)) {
            spdlog::warn("[CudaEngine] use_fp8_kv needs sm_89+. Disabling.");
            const_cast<CudaEngineConfig&>(cec).use_fp8_kv = false;
        }
    }

    // 3. Paged KV allocator
    allocator_ = std::make_unique<CudaAllocator>(
        cec, model_->n_layers(), model_->n_kv_heads(),
        model_->head_dim(), DataType::F16, hw_cfg_.gpu.device_id);

    // 4. CUDA stream pool
    streams_ = std::make_unique<CudaStream>(
        cec.num_cuda_streams, hw_cfg_.gpu.device_id);

    // 5. Multi-GPU coordinator (tp_size=1 no-op)
    multi_gpu_ = std::make_unique<MultiGpu>(
        cec.tensor_parallel_size, hw_cfg_.gpu.device_id);

    // 6. ggml graph allocator — uses the same CUDA backend as the model
    gallocr_ = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(model_->cuda_backend()));

    // 7. Small device scratch buffers
    alloc_scratch_buffers();

    // 8. CPU sampler chain
    build_sampler();

    // 9. Megakernel buffers (if enabled — requires model + allocator ready).
    //
    // The megakernel uses grid.sync() inside truellm_transformer_megakernel
    // and is launched via cudaLaunchCooperativeKernel.  That requires
    //   (a) compute capability >= 6.0  (cooperativeLaunch attribute)
    //   (b) grid.sync() support, which is guaranteed on sm_70+.
    // On older GPUs, mutate the config to fall back to the legacy per-token
    // ggml+vllm path rather than crashing at launch time.
    if (cec.use_megakernel) {
        int device_id = hw_cfg_.gpu.device_id;
        int coop = 0;
        cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, device_id);
        int sm_major = 0;
        cudaDeviceGetAttribute(&sm_major,
                               cudaDevAttrComputeCapabilityMajor, device_id);
        if (coop == 0 || sm_major < 7) {
            spdlog::warn("[CudaEngine] Megakernel requested but device {} "
                         "does not support cooperative grid.sync() "
                         "(compute capability {}.x, cooperativeLaunch={}); "
                         "falling back to legacy per-token forward.",
                         device_id, sm_major, coop);
            inf_cfg_.cuda_engine.use_megakernel = false;
        } else {
            alloc_megakernel_buffers();
        }
    }

    // 10. KV compression pass (VQ / KiVi) — engine-internal, no plugin needed
    {
        const CompressionConfig& cc = inf_cfg_.compression;
        if (cc.kv_vq.enabled || cc.kv_kivi.enabled) {
            kv_compress_ = std::make_unique<KvCompressionPass>(
                cc,
                model_->n_layers(), model_->n_kv_heads(),
                model_->head_dim(), allocator_->block_size());
            kv_compress_->init(allocator_.get(),
                               streams_->get(StreamRole::Copy));
            spdlog::info("[CudaEngine] KvCompressionPass ready "
                         "(vq={} kivi={})",
                         cc.kv_vq.enabled, cc.kv_kivi.enabled);
        }
    }

    // 11. Compression scoring pass (Presis / ToMe) — runs before prefill
    {
        const CompressionConfig& cc = inf_cfg_.compression;
        if (cc.presis.enabled || cc.tome.enabled) {
            scoring_ = std::make_unique<CompressionScoringPass>(
                cc,
                model_->n_heads(), model_->head_dim(),
                model_->hidden_size());
            spdlog::info("[CudaEngine] CompressionScoringPass ready "
                         "(presis={} tome={})",
                         cc.presis.enabled, cc.tome.enabled);
        }
    }

    // 12. Optionally capture CUDA graphs (stub — no-op when enforce_eager=true)
    if (!cec.enforce_eager) capture_cuda_graphs();

    spdlog::info("[CudaEngine] Ready: model={} layers={} heads={}(kv={}) "
                 "hidden={} vocab={}",
                 model_path, model_->n_layers(), model_->n_heads(),
                 model_->n_kv_heads(), model_->hidden_size(),
                 model_->vocab_size());
    return ErrorCode::Ok;
#endif
}

void CudaEngine::unload_model()
{
#ifdef TRUELLM_HAS_CUDA
    free_sampler();
    free_megakernel_buffers();
    free_scratch_buffers();
    if (gallocr_) { ggml_gallocr_free(gallocr_); gallocr_ = nullptr; }
#endif
    kv_compress_.reset();
    scoring_.reset();
    multi_gpu_.reset();
    streams_.reset();
    allocator_.reset();
    model_.reset();
    cuda_graphs_.clear();
}

bool CudaEngine::is_model_loaded() const
{
    return model_ && model_->is_loaded();
}

// ---------------------------------------------------------------------------
// Scratch buffers (1-token sized)
// ---------------------------------------------------------------------------
void CudaEngine::alloc_scratch_buffers()
{
#ifdef TRUELLM_HAS_CUDA
    const int n_heads = model_->n_heads();
    const int h_dim   = model_->head_dim();
    const int hidden  = model_->hidden_size();

    CUDA_CHECK(cudaMalloc(&d_attn_out_,
               static_cast<size_t>(n_heads) * h_dim * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_residual_,
               static_cast<size_t>(hidden) * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_seq_lens_, sizeof(int32_t)));

    // Pre-allocate paged_attention_v2 temp buffers for worst-case context.
    // max_partitions = ceil(n_ctx_train / (32 * block_size))
    const int max_ctx = static_cast<int>(model_->n_ctx_train());
    const int blk_sz  = allocator_->block_size();
    v2_max_partitions_ = (max_ctx + 32 * blk_sz - 1) / (32 * blk_sz);
    if (v2_max_partitions_ > 0) {
        CUDA_CHECK(cudaMalloc(&d_v2_exp_sums_,
            static_cast<size_t>(n_heads) * v2_max_partitions_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_v2_max_logits_,
            static_cast<size_t>(n_heads) * v2_max_partitions_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_v2_tmp_out_,
            static_cast<size_t>(n_heads) * v2_max_partitions_ * h_dim * sizeof(__half)));
    }
#endif
}

void CudaEngine::free_scratch_buffers()
{
#ifdef TRUELLM_HAS_CUDA
    if (d_attn_out_)    { cudaFree(d_attn_out_);    d_attn_out_    = nullptr; }
    if (d_residual_)    { cudaFree(d_residual_);    d_residual_    = nullptr; }
    if (d_seq_lens_)    { cudaFree(d_seq_lens_);    d_seq_lens_    = nullptr; }
    if (d_v2_exp_sums_) { cudaFree(d_v2_exp_sums_); d_v2_exp_sums_ = nullptr; }
    if (d_v2_max_logits_){ cudaFree(d_v2_max_logits_); d_v2_max_logits_ = nullptr; }
    if (d_v2_tmp_out_)  { cudaFree(d_v2_tmp_out_);  d_v2_tmp_out_  = nullptr; }
    v2_max_partitions_ = 0;
#endif
}

// ---------------------------------------------------------------------------
// Megakernel buffers
// ---------------------------------------------------------------------------
void CudaEngine::alloc_megakernel_buffers()
{
#ifdef TRUELLM_HAS_CUDA
    if (!model_) return;
    const CudaEngineConfig& cec = inf_cfg_.cuda_engine;

    mk_n_layers_  = model_->n_layers();
    mk_n_heads_   = model_->n_heads();
    mk_seq_cap_   = static_cast<int32_t>(inf_cfg_.context_size);
    mk_kv_cap_    = allocator_ ? allocator_->num_total_blocks()
                               * allocator_->block_size() : mk_seq_cap_;

    const int32_t n_kv   = model_->n_kv_heads();
    const int32_t h_dim  = model_->head_dim();
    const int32_t hidden = model_->hidden_size();
    const int32_t d_ffn  = model_->ffn_hidden_size();
    const int32_t L      = mk_n_layers_;
    const int32_t S      = mk_seq_cap_;
    const int32_t KVC    = mk_kv_cap_;
    const int32_t H      = mk_n_heads_;

    // Activation double-buffer
    CUDA_CHECK(cudaMalloc(&d_mk_act_buf_,
        static_cast<size_t>(2) * S * hidden * sizeof(__half)));
    // Softmax scratch: generous upper bound (H * S * KVC)
    CUDA_CHECK(cudaMalloc(&d_mk_softmax_buf_,
        static_cast<size_t>(H) * S * KVC * sizeof(float)));
    // Token ids + argmax
    CUDA_CHECK(cudaMalloc(&d_mk_input_ids_, static_cast<size_t>(S) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_mk_argmax_out_, sizeof(int32_t)));
    // FastV mask
    CUDA_CHECK(cudaMalloc(&d_mk_attn_mask_,
        static_cast<size_t>(KVC) * sizeof(bool)));

    // Optional attention weights: [L, H, S, KVC] F32
    if (cec.megakernel_save_attn) {
        CUDA_CHECK(cudaMalloc(&d_mk_attn_weights_,
            static_cast<size_t>(L) * H * S * KVC * sizeof(float)));
    }
    // Optional hidden states: [L+1, S, hidden] F16
    if (cec.megakernel_save_hidden) {
        CUDA_CHECK(cudaMalloc(&d_mk_hidden_states_,
            static_cast<size_t>(L + 1) * S * hidden * sizeof(__half)));
    }

    // Build per-layer weight pointer arrays from TrueBuffer weight map.
    // Each ptr points directly into ggml's device buffer (no copy).
    h_wq_ptrs_.resize(L); h_wk_ptrs_.resize(L); h_wv_ptrs_.resize(L);
    h_wo_ptrs_.resize(L); h_wgate_ptrs_.resize(L); h_wup_ptrs_.resize(L);
    h_wdown_ptrs_.resize(L);
    h_rms_att_ptrs_.resize(L); h_rms_ffn_ptrs_.resize(L);

    // Helper: get F16 device ptr from layer TrueBuffer map.
    auto get_f16 = [&](int l, const char* name) -> const __half* {
        const auto& lw = model_->layer_weights(l);
        auto it = lw.find(name);
        return (it != lw.end())
            ? reinterpret_cast<const __half*>(it->second.data)
            : nullptr;
    };
    auto get_f32 = [&](int l, const char* name) -> const float* {
        const auto& lw = model_->layer_weights(l);
        auto it = lw.find(name);
        return (it != lw.end())
            ? reinterpret_cast<const float*>(it->second.data)
            : nullptr;
    };

    for (int l = 0; l < L; ++l) {
        h_wq_ptrs_[l]      = get_f16(l, "attn_q.weight");
        h_wk_ptrs_[l]      = get_f16(l, "attn_k.weight");
        h_wv_ptrs_[l]      = get_f16(l, "attn_v.weight");
        h_wo_ptrs_[l]      = get_f16(l, "attn_output.weight");
        h_wgate_ptrs_[l]   = get_f16(l, "ffn_gate.weight");
        h_wup_ptrs_[l]     = get_f16(l, "ffn_up.weight");
        h_wdown_ptrs_[l]   = get_f16(l, "ffn_down.weight");
        h_rms_att_ptrs_[l] = get_f32(l, "attn_norm.weight");
        h_rms_ffn_ptrs_[l] = get_f32(l, "ffn_norm.weight");
    }

    // Copy pointer arrays to device (packed into one allocation).
    // Layout: wq[L], wk[L], wv[L], wo[L], wgate[L], wup[L], wdown[L] × __half*
    //         rms_att[L], rms_ffn[L] × float*
    size_t f16_arr_bytes = static_cast<size_t>(L) * sizeof(const __half*);
    size_t f32_arr_bytes = static_cast<size_t>(L) * sizeof(const float*);
    size_t total_ptr_bytes = 7 * f16_arr_bytes + 2 * f32_arr_bytes;
    CUDA_CHECK(cudaMalloc(&d_mk_ptr_arrays_, total_ptr_bytes));
    char* dst = reinterpret_cast<char*>(d_mk_ptr_arrays_);
    auto upload_f16_arr = [&](const std::vector<const __half*>& v) {
        CUDA_CHECK(cudaMemcpy(dst, v.data(), f16_arr_bytes, cudaMemcpyHostToDevice));
        dst += f16_arr_bytes;
    };
    auto upload_f32_arr = [&](const std::vector<const float*>& v) {
        CUDA_CHECK(cudaMemcpy(dst, v.data(), f32_arr_bytes, cudaMemcpyHostToDevice));
        dst += f32_arr_bytes;
    };
    upload_f16_arr(h_wq_ptrs_); upload_f16_arr(h_wk_ptrs_); upload_f16_arr(h_wv_ptrs_);
    upload_f16_arr(h_wo_ptrs_); upload_f16_arr(h_wgate_ptrs_); upload_f16_arr(h_wup_ptrs_);
    upload_f16_arr(h_wdown_ptrs_);
    upload_f32_arr(h_rms_att_ptrs_); upload_f32_arr(h_rms_ffn_ptrs_);

    // KV cache pointer arrays — per-layer K and V base pointers from CudaAllocator.
    h_k_cache_ptrs_.resize(L);
    h_v_cache_ptrs_.resize(L);
    for (int l = 0; l < L; ++l) {
        h_k_cache_ptrs_[l] = reinterpret_cast<__half*>(allocator_->kv_layer_k_ptr(l));
        h_v_cache_ptrs_[l] = reinterpret_cast<__half*>(allocator_->kv_layer_v_ptr(l));
    }
    size_t kv_arr_bytes = static_cast<size_t>(L) * sizeof(__half*);
    CUDA_CHECK(cudaMalloc(&d_mk_kv_ptr_arrays_, 2 * kv_arr_bytes));
    CUDA_CHECK(cudaMemcpy(d_mk_kv_ptr_arrays_,
               h_k_cache_ptrs_.data(), kv_arr_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<char*>(d_mk_kv_ptr_arrays_) + kv_arr_bytes,
               h_v_cache_ptrs_.data(), kv_arr_bytes, cudaMemcpyHostToDevice));

    spdlog::debug("[CudaEngine] Megakernel buffers allocated: "
                  "seq_cap={} kv_cap={} save_attn={} save_hidden={}",
                  S, KVC, cec.megakernel_save_attn, cec.megakernel_save_hidden);
#endif
}

void CudaEngine::free_megakernel_buffers()
{
#ifdef TRUELLM_HAS_CUDA
    auto cf = [](void*& p) { if (p) { cudaFree(p); p = nullptr; } };
    cf(d_mk_act_buf_); cf(d_mk_softmax_buf_); cf(d_mk_attn_weights_);
    cf(d_mk_hidden_states_); cf(d_mk_ptr_arrays_); cf(d_mk_kv_ptr_arrays_);
    if (d_mk_input_ids_)  { cudaFree(d_mk_input_ids_);  d_mk_input_ids_  = nullptr; }
    if (d_mk_argmax_out_) { cudaFree(d_mk_argmax_out_); d_mk_argmax_out_ = nullptr; }
    if (d_mk_attn_mask_)  { cudaFree(d_mk_attn_mask_);  d_mk_attn_mask_  = nullptr; }
    h_wq_ptrs_.clear(); h_wk_ptrs_.clear(); h_wv_ptrs_.clear();
    h_wo_ptrs_.clear(); h_wgate_ptrs_.clear(); h_wup_ptrs_.clear();
    h_wdown_ptrs_.clear(); h_rms_att_ptrs_.clear(); h_rms_ffn_ptrs_.clear();
    h_k_cache_ptrs_.clear(); h_v_cache_ptrs_.clear();
    mk_n_layers_ = mk_n_heads_ = mk_seq_cap_ = mk_kv_cap_ = 0;
#endif
}

// ---------------------------------------------------------------------------
// run_forward_megakernel — cooperative single-launch forward pass
//
// tokens        : all input token ids for this forward pass
// kv_seq_len    : total KV slots used after this forward (past + current)
// kv_write_offset: index of first slot to write (= kv_seq_len - seq_len)
// ---------------------------------------------------------------------------
ErrorCode CudaEngine::run_forward_megakernel(
    const std::vector<int32_t>& tokens,
    int                         kv_seq_len,
    int                         kv_write_offset,
    cudaStream_t                stream)
{
#ifndef TRUELLM_HAS_CUDA
    return ErrorCode::NotImplemented;
#else
    const CudaEngineConfig& cec = inf_cfg_.cuda_engine;
    const int32_t seq_len = static_cast<int32_t>(tokens.size());

    // Upload input token ids.
    CUDA_CHECK(cudaMemcpyAsync(d_mk_input_ids_, tokens.data(),
               static_cast<size_t>(seq_len) * sizeof(int32_t),
               cudaMemcpyHostToDevice, stream));

    // Reset FastV/PyramidDrop mask to all-true.
    if (d_mk_attn_mask_) {
        CUDA_CHECK(cudaMemsetAsync(d_mk_attn_mask_, 1,
                   static_cast<size_t>(kv_seq_len) * sizeof(bool), stream));
    }

    // Device-side packed pointer arrays (laid out by alloc_megakernel_buffers).
    char* pp = reinterpret_cast<char*>(d_mk_ptr_arrays_);
    size_t f16s = static_cast<size_t>(mk_n_layers_) * sizeof(const __half*);
    size_t f32s = static_cast<size_t>(mk_n_layers_) * sizeof(const float*);
    const __half* const* d_wq    = reinterpret_cast<const __half* const*>(pp); pp += f16s;
    const __half* const* d_wk    = reinterpret_cast<const __half* const*>(pp); pp += f16s;
    const __half* const* d_wv    = reinterpret_cast<const __half* const*>(pp); pp += f16s;
    const __half* const* d_wo    = reinterpret_cast<const __half* const*>(pp); pp += f16s;
    const __half* const* d_wgate = reinterpret_cast<const __half* const*>(pp); pp += f16s;
    const __half* const* d_wup   = reinterpret_cast<const __half* const*>(pp); pp += f16s;
    const __half* const* d_wdown = reinterpret_cast<const __half* const*>(pp); pp += f16s;
    const float*  const* d_rms_a = reinterpret_cast<const float* const*>(pp);  pp += f32s;
    const float*  const* d_rms_f = reinterpret_cast<const float* const*>(pp);

    char* kp = reinterpret_cast<char*>(d_mk_kv_ptr_arrays_);
    size_t ks = static_cast<size_t>(mk_n_layers_) * sizeof(__half*);
    __half* const* d_k_cache = reinterpret_cast<__half* const*>(kp);
    __half* const* d_v_cache = reinterpret_cast<__half* const*>(kp + ks);

    // Global weight device pointers.
    const __half* w_embed = nullptr, *w_unembed = nullptr;
    const float*  rms_final = nullptr;
    {
        const auto& gw = model_->layer_weights(0); // unused — use global_weight
        auto gw_f16 = [&](const char* n) -> const __half* {
            try {
                return reinterpret_cast<const __half*>(
                    model_->global_weight(n).data);
            } catch (...) { return nullptr; }
        };
        auto gw_f32 = [&](const char* n) -> const float* {
            try {
                return reinterpret_cast<const float*>(
                    model_->global_weight(n).data);
            } catch (...) { return nullptr; }
        };
        w_embed    = gw_f16("token_embd.weight");
        w_unembed  = gw_f16("output.weight");
        rms_final  = gw_f32("output_norm.weight");
        (void)gw;
    }

    // Compression config from plugin context metadata is not accessible here;
    // FastV/PyramidDrop parameters are set by the preprocessor via context
    // metadata and read during generate(). For now, read from inf_cfg_.
    const CompressionConfig& cc = inf_cfg_.compression;
    int fastv_start = -1;
    float fastv_keep = 1.0f;
    if (cc.fastv.enabled) {
        fastv_start = cc.fastv.start_layer;
        fastv_keep  = cc.fastv.keep_ratio;
    }
    int pyramid_layers = 0;
    float pyramid_lambda = 1.0f;
    if (cc.pyramid_drop.enabled) {
        pyramid_layers = cc.pyramid_drop.warmup_layers;
        pyramid_lambda = cc.pyramid_drop.final_keep_ratio;
    }

    MegakernelArgs args{};
    args.n_layers      = model_->n_layers();
    args.n_heads       = model_->n_heads();
    args.n_kv_heads    = model_->n_kv_heads();
    args.head_dim      = model_->head_dim();
    args.d_model       = model_->hidden_size();
    args.d_ffn         = model_->ffn_hidden_size();
    args.vocab_size    = static_cast<int32_t>(model_->vocab_size());
    args.seq_len       = seq_len;
    args.kv_seq_len    = kv_seq_len;
    args.batch_size    = 1;
    args.wq            = d_wq;
    args.wk            = d_wk;
    args.wv            = d_wv;
    args.wo            = d_wo;
    args.w_gate        = d_wgate;
    args.w_up          = d_wup;
    args.w_down        = d_wdown;
    args.rms_att_w     = d_rms_a;
    args.rms_ffn_w     = d_rms_f;
    args.w_embed       = w_embed;
    args.w_unembed     = w_unembed;
    args.rms_final_w   = rms_final;
    args.rope_cos      = model_->freqs_cos_dev();
    args.rope_sin      = model_->freqs_sin_dev();
    args.rope_offset   = kv_write_offset;
    args.k_cache       = d_k_cache;
    args.v_cache       = d_v_cache;
    args.kv_cache_capacity = mk_kv_cap_;
    args.act_buf       = reinterpret_cast<__half*>(d_mk_act_buf_);
    args.softmax_buf   = reinterpret_cast<float*>(d_mk_softmax_buf_);
    args.input_ids     = d_mk_input_ids_;
    args.output_logit_argmax = d_mk_argmax_out_;
    args.logits_out    = nullptr; // not needed; argmax is sufficient
    args.fastv_start_layer   = fastv_start;
    args.fastv_keep_ratio    = fastv_keep;
    args.attn_logit_mask     = d_mk_attn_mask_;
    args.pyramid_drop_layers = pyramid_layers;
    args.pyramid_drop_lambda = pyramid_lambda;
    args.save_attn_weights   = cec.megakernel_save_attn;
    args.attn_weight_out     = reinterpret_cast<float*>(d_mk_attn_weights_);
    args.save_hidden_states  = cec.megakernel_save_hidden;
    args.hidden_state_out    = reinterpret_cast<__half*>(d_mk_hidden_states_);

    // Determine cooperative launch geometry.
    // Blocks = number of SM * occupancy estimate; threads = 128 (4 warps).
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor gives the SM-occupancy bound.
    int device_id   = hw_cfg_.gpu.device_id;
    int num_sms     = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&num_sms,
               cudaDevAttrMultiProcessorCount, device_id));

    int max_blocks_per_sm = 0;
    const int threads_per_block = 128;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm,
        (const void*)truellm_transformer_megakernel,
        threads_per_block, 0);
    if (max_blocks_per_sm < 1) max_blocks_per_sm = 1;

    int num_blocks = num_sms * max_blocks_per_sm;

    // Shared memory: warp_gemm needs max(d_model, d_ffn)*sizeof(__half)
    // plus blockDim.x * sizeof(float) for RMS reduce.
    size_t smem = static_cast<size_t>(model_->ffn_hidden_size()) * 2 * sizeof(__half)
                + static_cast<size_t>(threads_per_block) * sizeof(float);

    void* kernel_args[] = { &args };
    CUDA_CHECK(cudaLaunchCooperativeKernel(
        (const void*)truellm_transformer_megakernel,
        dim3(num_blocks), dim3(threads_per_block),
        kernel_args, smem, stream));

    // Copy argmax result to logits buffer (host vector expected by generate()).
    // We synthesise a one-hot logit vector: 0.0 everywhere, 1.0 at argmax.
    // This avoids a full vocab D2H copy; generate() reads d_logits_buf_.
    int32_t argmax_host = 0;
    CUDA_CHECK(cudaMemcpyAsync(&argmax_host, d_mk_argmax_out_,
               sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const int vsz = static_cast<int>(model_->vocab_size());
    d_logits_buf_.assign(vsz, 0.0f);
    if (argmax_host >= 0 && argmax_host < vsz)
        d_logits_buf_[argmax_host] = 1.0f;

    return ErrorCode::Ok;
#endif
}

// ---------------------------------------------------------------------------
// Sampler
// ---------------------------------------------------------------------------
void CudaEngine::build_sampler()
{
    free_sampler();
    llama_sampler_chain_params cp = llama_sampler_chain_default_params();
    sampler_ = llama_sampler_chain_init(cp);
    llama_sampler_chain_add(sampler_, llama_sampler_init_top_k(smp_cfg_.top_k));
    llama_sampler_chain_add(sampler_, llama_sampler_init_top_p(smp_cfg_.top_p, 1));
    llama_sampler_chain_add(sampler_, llama_sampler_init_min_p(smp_cfg_.min_p, 1));
    llama_sampler_chain_add(sampler_, llama_sampler_init_temp(smp_cfg_.temperature));
    llama_sampler_chain_add(sampler_,
        llama_sampler_init_penalties(64, smp_cfg_.repeat_penalty,
                                     smp_cfg_.frequency_penalty,
                                     smp_cfg_.presence_penalty));
    uint32_t seed = (smp_cfg_.seed < 0)
        ? static_cast<uint32_t>(
              std::chrono::steady_clock::now().time_since_epoch().count()
              & 0xFFFFFFFFu)
        : static_cast<uint32_t>(smp_cfg_.seed);
    llama_sampler_chain_add(sampler_, llama_sampler_init_dist(seed));
}

void CudaEngine::free_sampler()
{
    if (sampler_) { llama_sampler_free(sampler_); sampler_ = nullptr; }
}

// ---------------------------------------------------------------------------
// CUDA graph capture stub
// ---------------------------------------------------------------------------
void CudaEngine::capture_cuda_graphs()
{
    spdlog::debug("[CudaEngine] CUDA graph capture deferred — running eager");
}

// ---------------------------------------------------------------------------
// run_scoring_pass — pre-prefill Presis / ToMe token reduction
// ---------------------------------------------------------------------------
std::vector<int32_t> CudaEngine::run_scoring_pass(
    const std::vector<int32_t>& tokens, int kv_seq)
{
    if (!scoring_ || !scoring_->any_enabled()) return tokens;
    return scoring_->compress_tokens(tokens, static_cast<int>(tokens.size()), kv_seq);
}

// ---------------------------------------------------------------------------
// run_kv_compression_pass — post-prefill per-layer VQ / KiVi
// ---------------------------------------------------------------------------
void CudaEngine::run_kv_compression_pass(int seq_len, cudaStream_t stream)
{
#ifdef TRUELLM_HAS_CUDA
    if (!kv_compress_ || !kv_compress_->enabled()) return;
    const int L = model_->n_layers();
    for (int l = 0; l < L; ++l) {
        kv_compress_->run(l, seq_len,
                          allocator_->kv_layer_k_ptr(l),
                          allocator_->kv_layer_v_ptr(l),
                          stream);
    }
#else
    (void)seq_len; (void)stream;
#endif
}

// ---------------------------------------------------------------------------
// generate — prefill (sequential) + decode loop
// ---------------------------------------------------------------------------
GenerateResult CudaEngine::generate(const GenerateRequest& request)
{
    GenerateResult result{};
    if (!is_model_loaded()) {
        result.error = ErrorCode::Unavailable;
        result.error_message = "Model not loaded";
        return result;
    }

    auto t_start = std::chrono::steady_clock::now();
    uint64_t seq_id = next_seq_id_.fetch_add(1);

    if (!allocator_->allocate_sequence(seq_id,
                                       static_cast<int>(request.tokens.size()))) {
        result.error = ErrorCode::Unavailable;
        result.error_message = "KV pool exhausted";
        return result;
    }

    result.prompt_tokens = static_cast<int>(request.tokens.size());

#ifdef TRUELLM_HAS_CUDA
    const int block_size      = allocator_->block_size();

    // ------------------------------------------------------------------
    // Pre-prefill: Presis / ToMe token reduction (engine-internal).
    // Replaces the compression_preprocessor plugin's on_preprocess() hook.
    // ------------------------------------------------------------------
    const auto prompt_tokens =
        run_scoring_pass(request.tokens, static_cast<int>(request.tokens.size()));
    const int num_prompt = static_cast<int>(prompt_tokens.size());
    result.prompt_tokens = num_prompt;  // update with compressed count

    // ------------------------------------------------------------------
    // Prefill — megakernel path (whole sequence in one cooperative launch)
    //           or legacy path (one token at a time via ggml+vllm).
    // ------------------------------------------------------------------
    const bool use_mk = inf_cfg_.cuda_engine.use_megakernel;

    if (use_mk) {
        // Allocate all KV blocks for the full prompt at once.
        for (int i = block_size; i < num_prompt; i += block_size) {
            if (allocator_->extend_sequence(seq_id) < 0) {
                allocator_->free_sequence(seq_id);
                result.error = ErrorCode::Unavailable;
                result.error_message = "KV pool exhausted during prefill";
                return result;
            }
        }
        ErrorCode ec = run_forward_megakernel(
            prompt_tokens, num_prompt, /*kv_write_offset=*/0,
            streams_->get(StreamRole::Prefill));
        if (ec != ErrorCode::Ok) {
            allocator_->free_sequence(seq_id);
            result.error = ec;
            result.error_message = "Megakernel prefill failed";
            return result;
        }
        streams_->sync(StreamRole::Prefill);

        // Post-prefill: KV compression (VQ / KiVi) on all layers.
        run_kv_compression_pass(num_prompt, streams_->get(StreamRole::Copy));

        // Update scoring cache with newly captured attention / hidden states.
        if (scoring_) {
            scoring_->update_cache(d_mk_attn_weights_, d_mk_hidden_states_,
                                   num_prompt, num_prompt);
        }
    } else {
        for (int i = 0; i < num_prompt; ++i) {
            if (i > 0 && (i % block_size) == 0) {
                if (allocator_->extend_sequence(seq_id) < 0) {
                    allocator_->free_sequence(seq_id);
                    result.error = ErrorCode::Unavailable;
                    result.error_message = "KV pool exhausted during prefill";
                    return result;
                }
            }
            ErrorCode ec = run_forward(seq_id, prompt_tokens[i], i,
                                       streams_->get(StreamRole::Prefill));
            if (ec != ErrorCode::Ok) {
                allocator_->free_sequence(seq_id);
                result.error = ec;
                result.error_message = "Prefill step failed";
                return result;
            }
        }
        streams_->sync(StreamRole::Prefill);
    }

    // Sample first token — d_logits_buf_ is already a host std::vector
    const std::vector<float>& h_logits = d_logits_buf_;

    std::vector<llama_token_data> td(model_->vocab_size());
    for (int i = 0; i < static_cast<int>(model_->vocab_size()); ++i)
        td[i] = {i, h_logits[i], 0.f};
    llama_token_data_array tda{td.data(), td.size(), -1, false};
    llama_sampler_apply(sampler_, &tda);
    int32_t next_token = tda.data[tda.selected].id;

    // ------------------------------------------------------------------
    // Decode loop
    // ------------------------------------------------------------------
    int seq_pos = num_prompt;

    for (int step = 0; step < request.max_tokens; ++step) {
        // Stop conditions
        bool is_eos = (llama_vocab_type(model_->vocab()) != LLAMA_VOCAB_TYPE_NONE
                       && next_token == llama_vocab_eos(model_->vocab()));
        bool in_stop = false;
        for (int32_t st : request.stop_tokens)
            if (next_token == st) { in_stop = true; break; }
        if (is_eos || in_stop) break;

        result.tokens.push_back(next_token);
        std::string piece = model_->detokenize({next_token});
        result.text += piece;
        if (request.on_token) request.on_token(next_token, piece);

        // Extend KV pool at block boundary
        if ((seq_pos % block_size) == 0) {
            if (allocator_->extend_sequence(seq_id) < 0) {
                spdlog::warn("[CudaEngine] Out of KV blocks at step {}", step);
                break;
            }
        }

        ErrorCode ec;
        if (use_mk) {
            ec = run_forward_megakernel(
                {next_token}, seq_pos + 1, seq_pos,
                streams_->get(StreamRole::Decode));
        } else {
            ec = run_forward(seq_id, next_token, seq_pos,
                             streams_->get(StreamRole::Decode));
        }
        if (ec != ErrorCode::Ok) break;
        streams_->sync(StreamRole::Decode);

        // d_logits_buf_ is a host vector — no device copy needed
        for (int i = 0; i < static_cast<int>(model_->vocab_size()); ++i)
            td[i] = {i, h_logits[i], 0.f};
        tda = {td.data(), td.size(), -1, false};
        llama_sampler_apply(sampler_, &tda);
        next_token = tda.data[tda.selected].id;
        ++seq_pos;
    }
#endif // TRUELLM_HAS_CUDA

    allocator_->free_sequence(seq_id);

    auto t_end = std::chrono::steady_clock::now();
    result.generated_tokens = static_cast<int>(result.tokens.size());
    result.time_ms = static_cast<float>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            t_end - t_start).count());
    result.error = ErrorCode::Ok;
    spdlog::debug("[CudaEngine] Generated {} tokens in {:.1f}ms",
                  result.generated_tokens, result.time_ms);
    return result;
}

// ---------------------------------------------------------------------------
// run_forward — single-token transformer forward pass using ggml + vllm
//
// seq_pos  : absolute position of this token in the sequence (0-based)
// stream   : CUDA stream for compute (prefill or decode)
//
// Layout contract with vllm kernels:
//   K/V per block: [num_kv_heads, head_size, block_size]  (paged v1/v2 layout)
//   Q for attention: [1, n_heads, head_dim]   (flat: [n_heads*head_dim] F16)
//   attn output:     [1, n_heads, head_dim]   (same flat layout)
//
// ggml convention used here:
//   ggml_mul_mat(W, x)  computes  W^T x  where
//     W.ne[0] = in_features, W.ne[1] = out_features
//     x.ne[0] = in_features, x.ne[1] = 1
//   Result.ne[0] = out_features
// ---------------------------------------------------------------------------
ErrorCode CudaEngine::run_forward(
    uint64_t     seq_id,
    int32_t      token_id,
    int          seq_pos,
    cudaStream_t stream)
{
#ifndef TRUELLM_HAS_CUDA
    return ErrorCode::NotImplemented;
#else
    ggml_backend_t backend = model_->cuda_backend();
    if (!backend) return ErrorCode::NotImplemented;

    const int  hidden   = model_->hidden_size();
    const int  n_heads  = model_->n_heads();
    const int  n_kv     = model_->n_kv_heads();
    const int  h_dim    = model_->head_dim();
    const int  n_layers = model_->n_layers();
    const int  blk_sz   = allocator_->block_size();
    const int  dev      = hw_cfg_.gpu.device_id;

    // ------------------------------------------------------------------
    // Upload slot mapping + block table on copy stream, then
    // make the compute stream wait for that transfer.
    // ------------------------------------------------------------------
    const int32_t* slot_map = allocator_->get_device_slot_mapping(
        seq_id, 1, seq_pos, streams_->get(StreamRole::Copy));
    const int32_t* block_table = allocator_->get_device_block_table(
        seq_id, streams_->get(StreamRole::Copy));

    StreamRole compute_role = (stream == streams_->get(StreamRole::Prefill))
                            ? StreamRole::Prefill : StreamRole::Decode;
    int ev = streams_->record_event(StreamRole::Copy);
    streams_->wait_event(compute_role, ev);

    // Seq-len for paged attention = tokens seen so far (inclusive)
    int32_t seq_len_host = seq_pos + 1;
    CUDA_CHECK(cudaMemcpyAsync(d_seq_lens_, &seq_len_host, sizeof(int32_t),
                               cudaMemcpyHostToDevice, stream));
    // Synchronise so d_seq_lens_ is visible to vllm kernels below
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // ------------------------------------------------------------------
    // Shared per-step TrueBuffer views (block table, slot mapping, attn out)
    // KV cache views are built per-layer inside the loop (see below).
    // ------------------------------------------------------------------
    const int n_blk = allocator_->num_total_blocks();
    TrueBuffer bt_buf   = make_tb(const_cast<int32_t*>(block_table),
                                  DataType::I32, dev,
                                  allocator_->max_blocks_per_seq());
    TrueBuffer sl_buf   = make_tb(d_seq_lens_, DataType::I32, dev, 1);
    TrueBuffer slot_buf = make_tb(const_cast<int32_t*>(slot_map),
                                  DataType::I32, dev, 1);
    TrueBuffer attn_out_buf = make_tb(d_attn_out_, DataType::F16, dev,
                                      n_heads, h_dim);

    // ------------------------------------------------------------------
    // Step 1 — Embedding lookup → d_residual_
    //
    // ggml_backend_graph_compute is synchronous: GPU work completes
    // before the call returns, so we can safely copy ggml-managed device
    // data to our persistent d_residual_ scratch buffer immediately after.
    //
    // We must do this copy because ggml_gallocr_alloc_graph() can reuse
    // the same device memory region for subsequent graphs.  d_residual_
    // is owned by CudaEngine and is never touched by gallocr.
    // ------------------------------------------------------------------
    {
        struct ggml_init_params p{ GGML_CTX_BYTES, nullptr, true };
        ggml_context* ctx = ggml_init(p);

        ggml_tensor* emb = model_->global_tensor("token_embd.weight");
        ggml_tensor* ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_tensor* out = ggml_get_rows(ctx, emb, ids);

        ggml_cgraph* gf = ggml_new_graph_custom(ctx, GGML_CTX_NODES, false);
        ggml_build_forward_expand(gf, out);
        ggml_gallocr_alloc_graph(gallocr_, gf);

        ggml_backend_tensor_set(ids, &token_id, 0, sizeof(int32_t));
        ggml_backend_graph_compute(backend, gf);   // sync

        CUDA_CHECK(cudaMemcpy(d_residual_, ggml_get_data(out),
                              static_cast<size_t>(hidden) * sizeof(__half),
                              cudaMemcpyDeviceToDevice));
        ggml_free(ctx);
    }

    // ------------------------------------------------------------------
    // Steps 2–N — Layer loop
    //
    // Per layer:
    //   Graph A  (ggml, sync): RMSNorm + QKV + RoPE
    //               reads d_residual_  (pre-set, gallocr leaves it alone)
    //               writes q/k/v into gallocr buffer
    //   vllm launch (async on `stream`): reshape_and_cache, paged_attention
    //               reads q/k/v device ptrs captured just above
    //   cudaStreamSynchronize: ensures vllm done before next gallocr call
    //               (gallocr may overwrite q/k/v memory)
    //   Graph B  (ggml, sync): O-proj + add + RMSNorm + FFN
    //               reads d_residual_  (pre-set)
    //               reads d_attn_out_  (pre-set, written by paged_attention)
    //               writes res_out into gallocr buffer
    //   D2D copy: res_out → d_residual_  (so next graph can read it)
    // ------------------------------------------------------------------
    for (int l = 0; l < n_layers; ++l) {
        const auto& lt = model_->layer_tensors(l);
        auto get = [&](const char* name) -> ggml_tensor* {
            auto it = lt.find(name);
            return (it != lt.end()) ? it->second : nullptr;
        };

        // Per-layer KV cache TrueBuffer views — each layer has its own pool slice
        TrueBuffer kv_cache = make_tb(allocator_->kv_layer_kv_ptr(l), DataType::F16,
                                      dev, n_blk * 2, n_kv, h_dim, blk_sz);
        TrueBuffer k_cache  = make_tb(allocator_->kv_layer_k_ptr(l),  DataType::F16,
                                      dev, n_blk, n_kv, h_dim, blk_sz);
        TrueBuffer v_cache  = make_tb(allocator_->kv_layer_v_ptr(l),  DataType::F16,
                                      dev, n_blk, n_kv, h_dim, blk_sz);

        // ---- Graph A: RMSNorm + QKV + RoPE ----------------------------
        void* q_dev = nullptr;
        void* k_dev = nullptr;
        void* v_dev = nullptr;
        {
            struct ggml_init_params p{ GGML_CTX_BYTES, nullptr, true };
            ggml_context* ctx = ggml_init(p);

            // Residual input: pre-allocated, gallocr won't overwrite it
            ggml_tensor* res_in = ggml_new_tensor_2d(
                ctx, GGML_TYPE_F16, hidden, 1);
            res_in->data = d_residual_;

            ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

            ggml_tensor* normed = ggml_rms_norm(ctx, res_in,
                                                model_->rms_norm_eps());
            if (ggml_tensor* w = get("attn_norm.weight"))
                normed = ggml_mul(ctx, normed, w);

            ggml_tensor* q = ggml_mul_mat(ctx, get("attn_q.weight"), normed);
            ggml_tensor* k = ggml_mul_mat(ctx, get("attn_k.weight"), normed);
            ggml_tensor* v = ggml_mul_mat(ctx, get("attn_v.weight"), normed);

            ggml_tensor* q3 = ggml_reshape_3d(ctx, q, h_dim, n_heads, 1);
            ggml_tensor* k3 = ggml_reshape_3d(ctx, k, h_dim, n_kv,    1);

            const float rope_base = model_->rope_freq_base();
            ggml_tensor* qr = ggml_rope_ext(ctx, q3, pos, nullptr,
                                            h_dim, LLAMA_ROPE_TYPE_NORM,
                                            (int)model_->n_ctx_train(),
                                            rope_base, 1.f, 0.f, 1.f, 32.f, 1.f);
            ggml_tensor* kr = ggml_rope_ext(ctx, k3, pos, nullptr,
                                            h_dim, LLAMA_ROPE_TYPE_NORM,
                                            (int)model_->n_ctx_train(),
                                            rope_base, 1.f, 0.f, 1.f, 32.f, 1.f);

            ggml_cgraph* gf = ggml_new_graph_custom(ctx, GGML_CTX_NODES, false);
            ggml_build_forward_expand(gf, qr);
            ggml_build_forward_expand(gf, kr);
            ggml_build_forward_expand(gf, v);
            ggml_gallocr_alloc_graph(gallocr_, gf);

            ggml_backend_tensor_set(pos, &seq_pos, 0, sizeof(int32_t));
            ggml_backend_graph_compute(backend, gf);   // sync

            // Capture raw device ptrs before freeing ctx.
            // These point into the gallocr buffer — valid until the NEXT
            // ggml_gallocr_alloc_graph call, which happens after
            // cudaStreamSynchronize guarantees vllm is done reading them.
            q_dev = ggml_get_data(qr);
            k_dev = ggml_get_data(kr);
            v_dev = ggml_get_data(v);
            ggml_free(ctx);
        }

        // ---- vllm: write K,V to paged cache (async on stream) ----------
        {
            TrueBuffer k_buf = make_tb(k_dev, DataType::F16, dev, 1, n_kv, h_dim);
            TrueBuffer v_buf = make_tb(v_dev, DataType::F16, dev, 1, n_kv, h_dim);
            reshape_and_cache(k_buf, v_buf, kv_cache, slot_buf, stream);
        }

        // ---- vllm: paged attention (async on stream) -------------------
        {
            TrueBuffer q_buf  = make_tb(q_dev, DataType::F16, dev, 1, n_heads, h_dim);
            float scale   = 1.f / std::sqrt(static_cast<float>(h_dim));
            int   max_ctx = seq_pos + 1;

            if ((max_ctx / blk_sz) > 32) {
                // v2 path — use pre-allocated temp buffers (no malloc in hot path)
                int n_part = (max_ctx + 32 * blk_sz - 1) / (32 * blk_sz);
                TrueBuffer exp_b = make_tb(d_v2_exp_sums_,   DataType::F32,
                                           dev, n_heads, n_part);
                TrueBuffer max_b = make_tb(d_v2_max_logits_,  DataType::F32,
                                           dev, n_heads, n_part);
                TrueBuffer tmp_b = make_tb(d_v2_tmp_out_,     DataType::F16,
                                           dev, n_heads, n_part, h_dim);
                paged_attention_v2(attn_out_buf, exp_b, max_b, tmp_b,
                                   q_buf, k_cache, v_cache, bt_buf, sl_buf,
                                   n_kv, scale, h_dim, blk_sz, max_ctx, stream);
            } else {
                paged_attention_v1(attn_out_buf, q_buf, k_cache, v_cache,
                                   bt_buf, sl_buf, n_kv, scale, h_dim,
                                   blk_sz, stream);
            }
        }

        // Sync stream before graph B:
        //   1. d_attn_out_ must be fully written before ggml reads it.
        //   2. vllm is done reading q/k/v from the gallocr buffer,
        //      so the next ggml_gallocr_alloc_graph call may safely
        //      reuse that region.
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // ---- Graph B: O-proj + add + RMSNorm + FFN → d_residual_ -------
        {
            struct ggml_init_params p{ GGML_CTX_BYTES, nullptr, true };
            ggml_context* ctx = ggml_init(p);

            // Residual and attn_out inputs are pre-allocated (gallocr skips them)
            ggml_tensor* res_in = ggml_new_tensor_2d(
                ctx, GGML_TYPE_F16, hidden, 1);
            res_in->data = d_residual_;

            ggml_tensor* attn_out_t = ggml_new_tensor_2d(
                ctx, GGML_TYPE_F16, n_heads * h_dim, 1);
            attn_out_t->data = d_attn_out_;

            ggml_tensor* o_proj  = ggml_mul_mat(ctx, get("attn_output.weight"), attn_out_t);
            ggml_tensor* res2    = ggml_add(ctx, res_in, o_proj);

            ggml_tensor* normed2 = ggml_rms_norm(ctx, res2,
                                                  model_->rms_norm_eps());
            if (ggml_tensor* w = get("ffn_norm.weight"))
                normed2 = ggml_mul(ctx, normed2, w);

            ggml_tensor* gate    = ggml_mul_mat(ctx, get("ffn_gate.weight"), normed2);
            ggml_tensor* up      = ggml_mul_mat(ctx, get("ffn_up.weight"),   normed2);
            ggml_tensor* ffn_act = ggml_mul(ctx, ggml_silu(ctx, gate), up);
            ggml_tensor* down    = ggml_mul_mat(ctx, get("ffn_down.weight"), ffn_act);
            ggml_tensor* res_out = ggml_add(ctx, res2, down);

            ggml_cgraph* gf = ggml_new_graph_custom(ctx, GGML_CTX_NODES, false);
            ggml_build_forward_expand(gf, res_out);
            ggml_gallocr_alloc_graph(gallocr_, gf);
            ggml_backend_graph_compute(backend, gf);   // sync

            // Copy layer output to d_residual_ before gallocr reuses its buffer
            CUDA_CHECK(cudaMemcpy(d_residual_, ggml_get_data(res_out),
                                  static_cast<size_t>(hidden) * sizeof(__half),
                                  cudaMemcpyDeviceToDevice));
            ggml_free(ctx);
        }
    } // end layer loop

    // ------------------------------------------------------------------
    // Final norm + lm_head → d_logits_buf_ (host std::vector<float>)
    // ------------------------------------------------------------------
    {
        struct ggml_init_params p{ GGML_CTX_BYTES, nullptr, true };
        ggml_context* ctx = ggml_init(p);

        ggml_tensor* res_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, hidden, 1);
        res_in->data = d_residual_;

        ggml_tensor* out_norm_w = model_->global_tensor("output_norm.weight");
        ggml_tensor* lm_head_w  = model_->global_tensor("output.weight");

        ggml_tensor* normed = ggml_rms_norm(ctx, res_in, model_->rms_norm_eps());
        if (out_norm_w) normed = ggml_mul(ctx, normed, out_norm_w);

        ggml_tensor* logits = lm_head_w
            ? ggml_mul_mat(ctx, lm_head_w, normed)
            : normed;

        ggml_cgraph* gf = ggml_new_graph_custom(ctx, GGML_CTX_NODES, false);
        ggml_build_forward_expand(gf, logits);
        ggml_gallocr_alloc_graph(gallocr_, gf);
        ggml_backend_graph_compute(backend, gf);   // sync; logits->data valid

        const int vsz = static_cast<int>(model_->vocab_size());
        d_logits_buf_.resize(vsz);

        if (logits->type == GGML_TYPE_F32) {
            // ggml_backend_graph_compute synchronised the GPU — plain D2H copy
            CUDA_CHECK(cudaMemcpy(d_logits_buf_.data(), ggml_get_data(logits),
                                  static_cast<size_t>(vsz) * sizeof(float),
                                  cudaMemcpyDeviceToHost));
        } else {
            // F16 → copy to temp, then widen to float on CPU
            std::vector<__half> tmp(vsz);
            CUDA_CHECK(cudaMemcpy(tmp.data(), ggml_get_data(logits),
                                  static_cast<size_t>(vsz) * sizeof(__half),
                                  cudaMemcpyDeviceToHost));
            for (int i = 0; i < vsz; ++i)
                d_logits_buf_[i] = __half2float(tmp[i]);
        }
        ggml_free(ctx);
    }

    return ErrorCode::Ok;
#endif
}

// ---------------------------------------------------------------------------
// ModelInfo / get_token_budget / GpuStats / KvCacheStats
// ---------------------------------------------------------------------------
ModelInfo CudaEngine::get_model_info() const
{
    if (!is_model_loaded()) return {};
    return model_->get_model_info();
}

int64_t CudaEngine::get_token_budget() const
{
    if (!allocator_) return 0;
    return static_cast<int64_t>(allocator_->num_free_blocks())
         * allocator_->block_size();
}

GpuStats CudaEngine::get_gpu_stats() const
{
    if (GpuDevice::device_count() == 0) return {};
    GpuStats s;
    s.device_id     = hw_cfg_.gpu.device_id;
    s.layers_on_gpu = is_model_loaded() ? model_->n_layers() : 0;
    auto info = GpuDevice::query(s.device_id);
    s.device_name      = std::move(info.name);
    s.vram_total_bytes = info.vram_total_bytes;
    s.vram_free_bytes  = info.vram_free_bytes;
    return s;
}

KvCacheStats CudaEngine::get_kv_cache_stats() const
{
    if (!allocator_) return {};
    return allocator_->stats();
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------
std::vector<int32_t> CudaEngine::tokenize(const std::string& text) const
{
    if (!is_model_loaded()) return {};
    return model_->tokenize(text, /*add_bos=*/true);
}

std::string CudaEngine::detokenize(const std::vector<int32_t>& tokens) const
{
    if (!is_model_loaded()) return {};
    return model_->detokenize(tokens);
}

// ---------------------------------------------------------------------------
// api_minor >= 1 — compression / RLM data access (Phase C1 stubs)
// Full implementations added in Phase M4 when megakernel saves intermediate
// tensors. Until then, plugins see nullptr and fall back gracefully.
// ---------------------------------------------------------------------------

const void* CudaEngine::get_hidden_states(int32_t layer_idx,
                                          int64_t* out_size_bytes)
{
    if (!d_mk_hidden_states_ || layer_idx < 0 || layer_idx > mk_n_layers_) {
        if (out_size_bytes) *out_size_bytes = 0;
        return nullptr;
    }
    // hidden_state_out: [n_layers+1, seq_len, d_model] F16
    // layer_idx == n_layers → final post-norm hidden state
    int64_t layer_stride = static_cast<int64_t>(mk_seq_cap_)
                         * model_->hidden_size() * sizeof(__half);
    if (out_size_bytes) *out_size_bytes = layer_stride;
    return reinterpret_cast<const char*>(d_mk_hidden_states_)
         + layer_idx * layer_stride;
}

const void* CudaEngine::get_attention_weights(int32_t layer_idx,
                                              int64_t* out_size_bytes)
{
    if (!d_mk_attn_weights_ || layer_idx < 0 || layer_idx >= mk_n_layers_) {
        if (out_size_bytes) *out_size_bytes = 0;
        return nullptr;
    }
    // attn_weight_out: [n_layers, n_heads, seq_len, kv_seq_len] F32
    int64_t layer_stride = static_cast<int64_t>(mk_n_heads_)
                         * mk_seq_cap_ * mk_kv_cap_ * sizeof(float);
    if (out_size_bytes) *out_size_bytes = layer_stride;
    return reinterpret_cast<const char*>(d_mk_attn_weights_)
         + layer_idx * layer_stride;
}

const void* CudaEngine::get_kv_cache_tensor(int32_t layer_idx, int32_t type,
                                            int64_t out_shape[4])
{
    if (!allocator_ || layer_idx < 0 || layer_idx >= mk_n_layers_) {
        if (out_shape) { out_shape[0]=out_shape[1]=out_shape[2]=out_shape[3]=0; }
        return nullptr;
    }
    void* ptr = (type == 0) ? allocator_->kv_layer_k_ptr(layer_idx)
                             : allocator_->kv_layer_v_ptr(layer_idx);
    if (out_shape) {
        const int blk_sz = allocator_->block_size();
        const int n_blk  = allocator_->num_total_blocks();
        out_shape[0] = n_blk;
        out_shape[1] = model_->n_kv_heads();
        out_shape[2] = blk_sz;
        out_shape[3] = model_->head_dim();
    }
    return ptr;
}

} // namespace truellm
