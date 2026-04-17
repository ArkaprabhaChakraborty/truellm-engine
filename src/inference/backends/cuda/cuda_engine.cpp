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

    // 9. Optionally capture CUDA graphs (stub — no-op when enforce_eager=true)
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
    free_scratch_buffers();
    if (gallocr_) { ggml_gallocr_free(gallocr_); gallocr_ = nullptr; }
#endif
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
    const int max_context_len = static_cast<int>(model_->n_ctx_train());
    const int num_prompt      = static_cast<int>(request.tokens.size());

    // ------------------------------------------------------------------
    // Prefill — run one token at a time so each token attends to all
    // preceding cached K,V (causal, fully correct).
    // ------------------------------------------------------------------
    for (int i = 0; i < num_prompt; ++i) {
        // Extend block table when we cross a block boundary
        if (i > 0 && (i % block_size) == 0) {
            if (allocator_->extend_sequence(seq_id) < 0) {
                allocator_->free_sequence(seq_id);
                result.error = ErrorCode::Unavailable;
                result.error_message = "KV pool exhausted during prefill";
                return result;
            }
        }

        ErrorCode ec = run_forward(seq_id, request.tokens[i], i,
                                   streams_->get(StreamRole::Prefill));
        if (ec != ErrorCode::Ok) {
            allocator_->free_sequence(seq_id);
            result.error = ec;
            result.error_message = "Prefill step failed";
            return result;
        }
    }
    streams_->sync(StreamRole::Prefill);

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

        ErrorCode ec = run_forward(seq_id, next_token, seq_pos,
                                   streams_->get(StreamRole::Decode));
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

} // namespace truellm
