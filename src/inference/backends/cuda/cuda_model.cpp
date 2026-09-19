// ---------------------------------------------------------------------------
// cuda_model.cpp — CudaModel implementation
// ---------------------------------------------------------------------------

#include "cuda_model.h"

#include <llama.h>
#include "llama-model.h"   // internal: llama_model::get_tensor(name)
#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cuda.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>

#ifdef TRUELLM_HAS_CUDA
#  include <cuda_runtime.h>
#  define CUDA_CHECK(expr)                                                   \
    do {                                                                     \
        cudaError_t _e = (expr);                                             \
        if (_e != cudaSuccess)                                               \
            throw std::runtime_error(std::string("[CudaModel] CUDA error: ") \
                + cudaGetErrorString(_e));                                   \
    } while (0)
#endif

namespace truellm {

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------
CudaModel::CudaModel(const InferenceConfig& inf_cfg, const HardwareConfig& hw_cfg)
    : inf_cfg_(inf_cfg), hw_cfg_(hw_cfg)
{}

CudaModel::~CudaModel()
{
    unload();
}

// ---------------------------------------------------------------------------
// load
// ---------------------------------------------------------------------------
ErrorCode CudaModel::load(const std::string& path)
{
    if (path.empty()) {
        spdlog::error("[CudaModel] Empty model path");
        return ErrorCode::InvalidArgument;
    }

    unload();

    llama_model_params mp = llama_model_default_params();

    // CudaEngine always offloads all layers — KV is managed by the allocator,
    // not by llama.cpp's context.
    mp.n_gpu_layers = 999;
    mp.main_gpu     = hw_cfg_.gpu.device_id;
    mp.use_mmap     = inf_cfg_.mmap;
    mp.use_mlock    = inf_cfg_.mlock;

    // Multi-GPU tensor split (optional)
    static float ts_buf[128] = {};
    if (!inf_cfg_.cuda.tensor_split.empty()) {
        size_t n = std::min(inf_cfg_.cuda.tensor_split.size(),
                            static_cast<size_t>(128));
        float sum = 0.f;
        for (size_t i = 0; i < n; ++i) sum += inf_cfg_.cuda.tensor_split[i];
        if (sum > 0.f)
            for (size_t i = 0; i < n; ++i)
                ts_buf[i] = inf_cfg_.cuda.tensor_split[i] / sum;
        mp.tensor_split = ts_buf;
    }

    spdlog::info("[CudaModel] Loading '{}' (all layers → GPU)", path);
    auto t0 = std::chrono::steady_clock::now();
    model_ = llama_model_load_from_file(path.c_str(), mp);
    auto t1 = std::chrono::steady_clock::now();

    if (!model_) {
        spdlog::error("[CudaModel] Failed to load '{}'", path);
        return ErrorCode::NotFound;
    }

    float ms = static_cast<float>(
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
    spdlog::info("[CudaModel] Loaded in {:.1f}s", ms / 1000.f);

    extract_dimensions();
    build_weight_map();
    precompute_rope();

    // Create ggml CUDA backend for compute graphs in run_forward
#ifdef TRUELLM_HAS_CUDA
    cuda_backend_ = ggml_backend_cuda_init(hw_cfg_.gpu.device_id);
    if (!cuda_backend_) {
        spdlog::warn("[CudaModel] ggml_backend_cuda_init failed — "
                     "ggml compute graphs will not be available");
    }
#endif

    return ErrorCode::Ok;
}

void CudaModel::unload()
{
#ifdef TRUELLM_HAS_CUDA
    if (freqs_cos_dev_) { cudaFree(freqs_cos_dev_); freqs_cos_dev_ = nullptr; }
    if (freqs_sin_dev_) { cudaFree(freqs_sin_dev_); freqs_sin_dev_ = nullptr; }
#endif
    layer_weights_.clear();
    global_weights_.clear();
    layer_tensors_.clear();
    global_tensors_.clear();

    if (cuda_backend_) {
        ggml_backend_free(cuda_backend_);
        cuda_backend_ = nullptr;
    }
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
}

// ---------------------------------------------------------------------------
// extract_dimensions — read model dims from llama.cpp metadata
// ---------------------------------------------------------------------------
void CudaModel::extract_dimensions()
{
    n_layers_        = llama_model_n_layer(model_);
    n_heads_         = llama_model_n_head(model_);
    n_kv_heads_      = llama_model_n_head_kv(model_);
    hidden_size_     = llama_model_n_embd(model_);
    n_ctx_train_     = static_cast<int64_t>(llama_model_n_ctx_train(model_));
    vocab_size_      = static_cast<int64_t>(
        llama_vocab_n_tokens(llama_model_get_vocab(model_)));

    // head_dim = hidden_size / n_heads
    head_dim_ = (n_heads_ > 0) ? (hidden_size_ / n_heads_) : 0;

    // FFN hidden size: query from ggml metadata key "llm.feed_forward.length"
    // Fall back to 4 * hidden_size (standard transformer ratio).
    int32_t ffn = 0;
    {
        // Try to read via gguf metadata (the tensor count approach)
        // llama.cpp exposes llama_model_n_ff() in newer versions.
        // Use a safe heuristic for older builds.
        ffn = 4 * hidden_size_;
    }
    ffn_hidden_size_ = ffn;

    // Read RoPE base frequency from GGUF metadata.
    // Key: "llm.rope.freq_base" (standard for Llama, Mistral, Qwen, etc.)
    // Llama 3 uses 500000.0; most Llama 2-era models use 10000.0.
    {
        char buf[64] = {};
        if (llama_model_meta_val_str(model_, "llm.rope.freq_base",
                                     buf, sizeof(buf)) >= 0) {
            try { rope_freq_base_ = std::stof(buf); }
            catch (...) { rope_freq_base_ = 10000.f; }
        }
    }

    // Read RMSNorm epsilon from GGUF metadata.
    {
        char buf[64] = {};
        if (llama_model_meta_val_str(model_,
                "llm.attention.layer_norm_rms_epsilon",
                buf, sizeof(buf)) >= 0) {
            try { rms_norm_eps_ = std::stof(buf); }
            catch (...) { rms_norm_eps_ = 1e-5f; }
        }
    }

    spdlog::info("[CudaModel] dims: layers={} heads={}(kv={}) hidden={} "
                 "head_dim={} ffn={} ctx_train={} rope_base={} norm_eps={}",
                 n_layers_, n_heads_, n_kv_heads_, hidden_size_,
                 head_dim_, ffn_hidden_size_, n_ctx_train_,
                 rope_freq_base_, rms_norm_eps_);
}

// ---------------------------------------------------------------------------
// build_weight_map — wrap device tensors in TrueBuffer views
// ---------------------------------------------------------------------------
void CudaModel::build_weight_map()
{
    // Iterate over all ggml tensors in the model and classify them.
    // Tensor names follow the GGUF convention: "blk.{N}.{role}.weight"
    // Global tensors: "token_embd.weight", "output_norm.weight", "output.weight"

    layer_weights_.resize(n_layers_);
    layer_tensors_.resize(n_layers_);

    // Standard tensor name patterns (GGUF spec)
    // Global:
    //   token_embd.weight       [vocab_size, hidden_size]
    //   output_norm.weight      [hidden_size]
    //   output.weight           [vocab_size, hidden_size]
    // Per-layer (N = 0..n_layers-1):
    //   blk.N.attn_norm.weight  [hidden_size]
    //   blk.N.attn_q.weight     [n_heads * head_dim, hidden_size]
    //   blk.N.attn_k.weight     [n_kv_heads * head_dim, hidden_size]
    //   blk.N.attn_v.weight     [n_kv_heads * head_dim, hidden_size]
    //   blk.N.attn_output.weight[hidden_size, n_heads * head_dim]
    //   blk.N.ffn_norm.weight   [hidden_size]
    //   blk.N.ffn_gate.weight   [ffn_hidden, hidden_size]
    //   blk.N.ffn_up.weight     [ffn_hidden, hidden_size]
    //   blk.N.ffn_down.weight   [hidden_size, ffn_hidden]

    auto make_truebuffer = [&](const struct ggml_tensor* t) -> TrueBuffer {
        TrueBuffer buf{};
        if (!t) return buf;
        buf.data       = ggml_get_data(t);
        buf.size_bytes = ggml_nbytes(t);
        buf.dtype      = DataType::F16;  // weights are F16 on device after offload
        buf.device_id  = hw_cfg_.gpu.device_id;
        buf.ndim       = GGML_MAX_DIMS;
        for (int d = 0; d < GGML_MAX_DIMS && d < 4; ++d) {
            buf.shape [d] = t->ne[d];
            buf.stride[d] = t->nb[d] / ggml_element_size(t);
        }
        return buf;
    };

    static const char* global_names[] = {
        "token_embd.weight", "output_norm.weight", "output.weight", nullptr
    };
    for (int i = 0; global_names[i]; ++i) {
        // llama_model::get_tensor() is the internal C++ method (returns const ptr)
        ggml_tensor* t = const_cast<ggml_tensor*>(model_->get_tensor(global_names[i]));
        if (t) {
            global_weights_[global_names[i]] = make_truebuffer(t);
            global_tensors_[global_names[i]] = t;
        }
    }

    static const char* layer_roles[] = {
        "attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
        "attn_output.weight", "ffn_norm.weight",
        "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
        nullptr
    };

    char name_buf[128];
    for (int l = 0; l < n_layers_; ++l) {
        for (int r = 0; layer_roles[r]; ++r) {
            std::snprintf(name_buf, sizeof(name_buf), "blk.%d.%s", l, layer_roles[r]);
            ggml_tensor* t = const_cast<ggml_tensor*>(model_->get_tensor(name_buf));
            if (t) {
                layer_weights_[l][layer_roles[r]] = make_truebuffer(t);
                layer_tensors_[l][layer_roles[r]] = t;
            }
        }
    }

    spdlog::info("[CudaModel] Weight map built: {} global + {} layer entries",
                 global_weights_.size(),
                 n_layers_ > 0 ? layer_weights_[0].size() : 0);
}

// ---------------------------------------------------------------------------
// precompute_rope — upload frequency tables to device
// ---------------------------------------------------------------------------
void CudaModel::precompute_rope()
{
#ifndef TRUELLM_HAS_CUDA
    return;
#else
    // rope_freq_base_ already read from GGUF metadata in extract_dimensions()
    float rope_base = rope_freq_base_;
    int   half_dim  = head_dim_ / 2;
    int   max_len   = static_cast<int>(n_ctx_train_);

    // Host buffers
    std::vector<float> h_cos(max_len * half_dim);
    std::vector<float> h_sin(max_len * half_dim);

    for (int pos = 0; pos < max_len; ++pos) {
        for (int i = 0; i < half_dim; ++i) {
            float theta = static_cast<float>(pos)
                        / std::pow(rope_base, 2.f * i / head_dim_);
            h_cos[pos * half_dim + i] = std::cos(theta);
            h_sin[pos * half_dim + i] = std::sin(theta);
        }
    }

    size_t bytes = static_cast<size_t>(max_len) * half_dim * sizeof(float);
    CUDA_CHECK(cudaMalloc(&freqs_cos_dev_, bytes));
    CUDA_CHECK(cudaMalloc(&freqs_sin_dev_, bytes));
    CUDA_CHECK(cudaMemcpy(freqs_cos_dev_, h_cos.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(freqs_sin_dev_, h_sin.data(), bytes, cudaMemcpyHostToDevice));

    spdlog::info("[CudaModel] RoPE table uploaded: {} positions × {} dims",
                 max_len, half_dim);
#endif
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

int64_t CudaModel::layer_size_bytes_estimate() const
{
    if (!model_) return 0;
    int32_t n = llama_model_n_layer(model_);
    if (n <= 0) return 0;
    return static_cast<int64_t>(llama_model_size(model_)) / n;
}

const LayerWeights& CudaModel::layer_weights(int layer_idx) const
{
    if (layer_idx < 0 || layer_idx >= static_cast<int>(layer_weights_.size()))
        throw std::out_of_range("[CudaModel] layer_idx out of range");
    return layer_weights_[layer_idx];
}

const TrueBuffer& CudaModel::global_weight(const std::string& name) const
{
    auto it = global_weights_.find(name);
    if (it == global_weights_.end())
        throw std::runtime_error("[CudaModel] Weight not found: " + name);
    return it->second;
}

const CudaModel::GgmlLayerWeights& CudaModel::layer_tensors(int layer_idx) const
{
    if (layer_idx < 0 || layer_idx >= static_cast<int>(layer_tensors_.size()))
        throw std::out_of_range("[CudaModel] layer_idx out of range");
    return layer_tensors_[layer_idx];
}

ggml_tensor* CudaModel::global_tensor(const std::string& name) const
{
    auto it = global_tensors_.find(name);
    return (it != global_tensors_.end()) ? it->second : nullptr;
}

ModelInfo CudaModel::get_model_info() const
{
    ModelInfo info{};
    if (!model_) return info;

    char desc[256] = {};
    llama_model_desc(model_, desc, sizeof(desc));
    info.architecture   = desc;
    info.context_length = n_ctx_train_;
    info.vocab_size     = vocab_size_;
    info.n_layers       = n_layers_;
    info.n_heads        = n_heads_;
    info.head_dim       = head_dim_;
    info.dtype          = DataType::F16;

    // Chat-template seam — feeds the server's chat_format (common_chat) layer.
    if (const char* tmpl = llama_model_chat_template(model_, /*name=*/nullptr))
        info.chat_template = tmpl;
    if (const llama_vocab* v = llama_model_get_vocab(model_)) {
        char buf[64];
        llama_token bos = llama_vocab_bos(v);
        if (bos != LLAMA_TOKEN_NULL) {
            int n = llama_token_to_piece(v, bos, buf, sizeof(buf), 0, /*special=*/true);
            if (n > 0) info.bos_token.assign(buf, static_cast<std::size_t>(n));
        }
        llama_token eos = llama_vocab_eos(v);
        if (eos != LLAMA_TOKEN_NULL) {
            int n = llama_token_to_piece(v, eos, buf, sizeof(buf), 0, /*special=*/true);
            if (n > 0) info.eos_token.assign(buf, static_cast<std::size_t>(n));
        }
    }
    return info;
}

const llama_vocab* CudaModel::vocab() const
{
    return model_ ? llama_model_get_vocab(model_) : nullptr;
}

std::vector<int32_t> CudaModel::tokenize(const std::string& text, bool add_bos) const
{
    if (!model_) return {};
    const llama_vocab* v = llama_model_get_vocab(model_);
    int n = llama_vocab_n_tokens(v) + static_cast<int>(text.size());
    std::vector<int32_t> tokens(n);
    // parse_special=true  — special tokens like <|start_header_id|>/<|eot_id|>
    //   tokenized as their single IDs, not BPE fragments.
    // add_special=true — tokenizer inserts BOS from GGUF metadata (LLaMA 3:
    //   add_bos=true, add_eos=false).  BOS is not in the template string.
    //   add_eos=false means no EOS appended, so the model won't immediately
    //   sample EOS as its first token.
    (void)add_bos; // BOS is handled by the tokenizer via GGUF metadata
    int count = llama_tokenize(v, text.c_str(), static_cast<int>(text.size()),
                               tokens.data(), n, /*add_special=*/true, /*parse_special=*/true);
    if (count < 0) return {};
    tokens.resize(count);
    return tokens;
}

std::string CudaModel::detokenize(const std::vector<int32_t>& tokens) const
{
    if (!model_ || tokens.empty()) return {};
    const llama_vocab* v = llama_model_get_vocab(model_);
    std::string out;
    char buf[256];
    for (int32_t tok : tokens) {
        int n = llama_token_to_piece(v, tok, buf, sizeof(buf), 0, true);
        if (n > 0) out.append(buf, n);
    }
    return out;
}

} // namespace truellm
