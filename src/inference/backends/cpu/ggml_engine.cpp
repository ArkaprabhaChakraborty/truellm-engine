// ---------------------------------------------------------------------------
// ggml_engine.cpp
// ---------------------------------------------------------------------------

#include "ggml_engine.h"
#include "ggml_model.h"
#include "ggml_context.h"
#include "gpu/gpu_device.h"
#include "../../scheduler/batch_scheduler.h"

#include <llama.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace truellm {

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------
GgmlEngine::GgmlEngine(InferenceConfig inf_cfg,
                       HardwareConfig  hw_cfg,
                       SamplingConfig  smp_cfg)
    : inf_cfg_(std::move(inf_cfg))
    , hw_cfg_ (std::move(hw_cfg))
    , smp_cfg_(std::move(smp_cfg))
{
    llama_backend_init();
}

GgmlEngine::~GgmlEngine()
{
    unload_model();
    llama_backend_free();
}

// ---------------------------------------------------------------------------
// load_model / unload_model
// ---------------------------------------------------------------------------
ErrorCode GgmlEngine::load_model(const std::string& model_path)
{
    model_ = std::make_unique<GgmlModel>(inf_cfg_, hw_cfg_);
    ErrorCode ec = model_->load(model_path);
    if (ec != ErrorCode::Ok) {
        model_.reset();
        return ec;
    }

    ctx_ = std::make_unique<GgmlContext>(inf_cfg_, smp_cfg_);
    if (!ctx_->init(*model_)) {
        model_.reset();
        ctx_.reset();
        return ErrorCode::InternalError;
    }

    // Start continuous batching scheduler if enabled
    if (inf_cfg_.batching.enabled) {
        const int max_seqs  = std::max(1, inf_cfg_.batching.max_seqs);
        const int max_queue = std::max(1, inf_cfg_.batching.max_queue);
        scheduler_ = std::make_unique<ContinuousBatchScheduler>(
            ctx_->raw(), model_->vocab(), smp_cfg_, max_seqs, max_queue);
        spdlog::info("[Engine] Continuous batching ENABLED  max_seqs={} max_queue={}",
                     max_seqs, max_queue);
    }

    return ErrorCode::Ok;
}

void GgmlEngine::unload_model()
{
    scheduler_.reset();  // stop scheduler before destroying context
    ctx_.reset();
    model_.reset();
}

bool GgmlEngine::is_model_loaded() const
{
    return model_ && model_->is_loaded();
}

// ---------------------------------------------------------------------------
// get_model_info
// ---------------------------------------------------------------------------
ModelInfo GgmlEngine::get_model_info() const
{
    ModelInfo info{};
    if (!is_model_loaded()) return info;

    // Architecture string from model metadata
    char arch_buf[128] = {};
    llama_model_meta_val_str(model_->raw(), "general.architecture",
                             arch_buf, sizeof(arch_buf));
    info.architecture   = arch_buf[0] ? arch_buf : "unknown";

    char name_buf[256] = {};
    llama_model_meta_val_str(model_->raw(), "general.name",
                             name_buf, sizeof(name_buf));
    info.model_id       = name_buf[0] ? name_buf : info.architecture;

    info.context_length = model_->n_ctx_train();
    info.vocab_size     = model_->vocab_size();
    info.n_layers       = model_->n_layers();
    info.n_heads        = 0; // not directly exposed via stable API
    info.head_dim       = 0;
    info.dtype          = DataType::F16; // GGUF quantized; report as f16 for now
    return info;
}

// ---------------------------------------------------------------------------
// get_token_budget
// ---------------------------------------------------------------------------
int64_t GgmlEngine::get_token_budget() const
{
    if (!ctx_ || !ctx_->is_valid()) return 0;
    // Return full context capacity; precise used-cell tracking is Part 2
    return static_cast<int64_t>(llama_n_ctx(ctx_->raw()));
}

// ---------------------------------------------------------------------------
// tokenize / detokenize
// ---------------------------------------------------------------------------
std::vector<int32_t> GgmlEngine::tokenize(const std::string& text) const
{
    if (!is_model_loaded()) return {};

    const llama_vocab* vocab = model_->vocab();
    // Allocate a buffer large enough for the worst case
    std::vector<llama_token> buf(text.size() + 16);

    // parse_special=true  — special tokens like <|start_header_id|> and <|eot_id|>
    //   are tokenized as their single special-token IDs, not BPE fragments.
    //
    // add_special=true  — lets the tokenizer insert the BOS token from GGUF metadata
    //   (e.g. LLaMA 3 specifies add_bos=true, add_eos=false).  BOS is NOT embedded
    //   in the prompt string; it is handled here.  The GGUF-specified add_eos=false
    //   means no EOS is appended, so the model will not immediately sample EOS.
    int n = llama_tokenize(vocab,
                           text.c_str(), static_cast<int32_t>(text.size()),
                           buf.data(), static_cast<int32_t>(buf.size()),
                           /*add_special=*/true, /*parse_special=*/true);
    if (n < 0) {
        // Buffer too small — retry with exact size
        buf.resize(static_cast<std::size_t>(-n));
        n = llama_tokenize(vocab,
                           text.c_str(), static_cast<int32_t>(text.size()),
                           buf.data(), static_cast<int32_t>(buf.size()),
                           true, true);
    }
    if (n < 0) return {};

    buf.resize(static_cast<std::size_t>(n));
    return std::vector<int32_t>(buf.begin(), buf.end());
}

std::string GgmlEngine::detokenize(const std::vector<int32_t>& tokens) const
{
    if (!is_model_loaded()) return {};

    const llama_vocab* vocab = model_->vocab();
    std::string result;
    result.reserve(tokens.size() * 4);

    char piece_buf[256];
    for (int32_t tok : tokens) {
        int len = llama_token_to_piece(vocab, static_cast<llama_token>(tok),
                                       piece_buf, sizeof(piece_buf),
                                       /*lstrip=*/0, /*special=*/false);
        if (len > 0) result.append(piece_buf, static_cast<std::size_t>(len));
    }
    return result;
}

// ---------------------------------------------------------------------------
// generate
// ---------------------------------------------------------------------------
bool GgmlEngine::matches_stop(const std::string& text) const
{
    for (const auto& stop : smp_cfg_.stop) {
        if (!stop.empty() && text.size() >= stop.size()) {
            if (text.compare(text.size() - stop.size(), stop.size(), stop) == 0)
                return true;
        }
    }
    return false;
}

GenerateResult GgmlEngine::generate(const GenerateRequest& req)
{
    GenerateResult result{};

    if (req.tokens.empty()) {
        result.error         = ErrorCode::InvalidArgument;
        result.error_message = "Empty token list";
        return result;
    }

    if (!is_model_loaded()) {
        result.error         = ErrorCode::NotFound;
        result.error_message = "No model loaded";
        return result;
    }

    // Route through continuous batching scheduler when enabled
    if (scheduler_) {
        return scheduler_->submit(req).get();
    }

    // Reset the KV cache for a fresh generation
    ctx_->reset();

    // Rebuild the sampler with any per-request overrides.
    // For now we use config defaults; per-request temp/top_p override
    // would require a temporary SamplingConfig here.
    // (Handled at the router level — see openai_router.cpp)

    spdlog::debug("[Engine] generate  prompt_tokens={} max_new={}",
                  req.tokens.size(),
                  (req.max_tokens > 0) ? req.max_tokens : smp_cfg_.max_tokens);

    auto t_start = std::chrono::steady_clock::now();

    // ---- Prefill ----
    std::vector<llama_token> prompt_tokens(req.tokens.begin(), req.tokens.end());
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(),
                                            static_cast<int32_t>(prompt_tokens.size()));

    auto t_prefill_start = std::chrono::steady_clock::now();
    if (llama_decode(ctx_->raw(), batch) != 0) {
        result.error         = ErrorCode::InternalError;
        result.error_message = "llama_decode (prefill) failed";
        spdlog::error("[Engine] llama_decode (prefill) failed");
        return result;
    }
    float prefill_ms = static_cast<float>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t_prefill_start).count()) / 1000.f;
    spdlog::debug("[Engine] prefill  {} tokens in {:.1f}ms ({:.0f} tok/s)",
                  prompt_tokens.size(), prefill_ms,
                  prefill_ms > 0 ? prompt_tokens.size() / (prefill_ms / 1000.f) : 0.f);

    result.prompt_tokens = static_cast<int32_t>(prompt_tokens.size());

    // ---- Decode loop ----
    const int max_new = (req.max_tokens > 0) ? req.max_tokens : smp_cfg_.max_tokens;
    const llama_vocab* vocab = model_->vocab();

    std::string accumulated;
    accumulated.reserve(static_cast<std::size_t>(max_new) * 4);

    int n_generated = 0;
    while (n_generated < max_new) {
        llama_token id = llama_sampler_sample(ctx_->sampler(), ctx_->raw(), -1);

        // EOG check — log the first token so we can diagnose silent early-exit
        if (n_generated == 0) {
            char piece_dbg[64] = {};
            llama_token_to_piece(vocab, id, piece_dbg, sizeof(piece_dbg) - 1, 0, true);
            spdlog::info("[Engine] first_token id={} text='{}' is_eog={}",
                         static_cast<int>(id), piece_dbg, llama_vocab_is_eog(vocab, id));
        }
        if (llama_vocab_is_eog(vocab, id)) break;

        // Decode token to text
        char piece_buf[256];
        int  piece_len = llama_token_to_piece(vocab, id, piece_buf, sizeof(piece_buf),
                                              0, false);
        std::string piece;
        if (piece_len > 0) piece.assign(piece_buf, static_cast<std::size_t>(piece_len));

        spdlog::trace("[Engine] tok[{:4d}] id={:6d}  '{}'",
                      n_generated + 1, static_cast<int>(id),
                      piece.empty() ? "<empty>" : piece);

        // Streaming callback
        if (req.on_token) req.on_token(static_cast<int32_t>(id), piece);

        accumulated += piece;
        result.tokens.push_back(static_cast<int32_t>(id));
        ++n_generated;

        // Stop token IDs
        bool stop_by_id = false;
        for (int32_t st : req.stop_tokens) {
            if (static_cast<llama_token>(st) == id) { stop_by_id = true; break; }
        }
        if (stop_by_id) break;

        // Stop strings (config-level)
        if (matches_stop(accumulated)) break;

        // Stop strings (per-request)
        bool stop_by_req_str = false;
        for (const auto& s : req.stop) {
            if (!s.empty() && accumulated.size() >= s.size() &&
                accumulated.compare(accumulated.size() - s.size(), s.size(), s) == 0) {
                stop_by_req_str = true;
                break;
            }
        }
        if (stop_by_req_str) break;

        // Feed generated token back for next decode step
        llama_token next_id = id;
        batch = llama_batch_get_one(&next_id, 1);
        if (llama_decode(ctx_->raw(), batch) != 0) {
            result.error         = ErrorCode::InternalError;
            result.error_message = "llama_decode (decode step) failed";
            spdlog::error("[Engine] llama_decode failed at step {}", n_generated);
            break;
        }
    }

    auto t_end = std::chrono::steady_clock::now();
    result.text             = std::move(accumulated);
    result.generated_tokens = n_generated;
    result.time_ms          = static_cast<float>(
        std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count());

    float decode_ms = result.time_ms - prefill_ms;
    float tps = (decode_ms > 0 && n_generated > 0)
              ? n_generated / (decode_ms / 1000.f) : 0.f;

    spdlog::debug("[Engine] done  prompt={} gen={} prefill={:.1f}ms "
                  "decode={:.1f}ms total={:.1f}ms ({:.1f} tok/s)",
                  result.prompt_tokens, n_generated,
                  prefill_ms, decode_ms, result.time_ms, tps);

    if (spdlog::should_log(spdlog::level::debug)) {
        std::string preview = result.text.substr(0, 120);
        if (result.text.size() > 120) preview += "…";
        spdlog::debug("[Engine] response preview: {}", preview);
    }

    return result;
}

// ---------------------------------------------------------------------------
// get_gpu_stats
// ---------------------------------------------------------------------------
GpuStats GgmlEngine::get_gpu_stats() const
{
    // Return empty stats on CPU-only builds — prevents native_router from
    // emitting a spurious GPU block with zero VRAM and an empty device name.
    if (GpuDevice::device_count() == 0) {
        return {};
    }

    GpuStats s;
    s.device_id = hw_cfg_.gpu.device_id;

    // Determine how many layers are actually on the GPU
    if (is_model_loaded()) {
        int cfg = hw_cfg_.offload.gpu_layers;
        s.layers_on_gpu = (cfg < 0) ? model_->n_layers()
                                    : std::min(cfg, model_->n_layers());
    }

    // Live VRAM figures (zeros in CPU-only builds)
    auto info = GpuDevice::query(s.device_id);
    s.device_name      = std::move(info.name);
    s.vram_total_bytes = info.vram_total_bytes;
    s.vram_free_bytes  = info.vram_free_bytes;
    return s;
}

// ---------------------------------------------------------------------------
// api_minor >= 1 — compression / RLM data access
// ---------------------------------------------------------------------------

const void* GgmlEngine::get_hidden_states(int32_t layer_idx,
                                          int64_t* out_size_bytes)
{
    if (out_size_bytes) *out_size_bytes = 0;
    if (!is_model_loaded() || !ctx_) return nullptr;

    // Returns the sequence-level embedding (pooled output) from the last forward
    // pass via llama_get_embeddings(). Layer-specific hidden states require a
    // custom llama.cpp build with intermediate tensor access — Phase C2 will
    // add a hook if the model was compiled with LLAMA_API_EMBEDDINGS.
    // layer_idx is recorded for future use; Phase C1 exposes the final layer only.
    (void)layer_idx;
    auto* lctx = ctx_->raw();
    if (!lctx) return nullptr;

    // llama_get_embeddings() returns NULL when the context was not built with
    // embeddings enabled (LLAMA_POOLING_TYPE_NONE). If NULL, Presis will fall
    // back to TF-IDF automatically.
    float* emb = llama_get_embeddings(lctx);
    if (!emb) return nullptr;

    const auto* lmodel = llama_get_model(lctx);
    int32_t n_embd = llama_model_n_embd(lmodel);
    // n_tokens: use llama_n_seq_max() on newer llama.cpp; fall back to n_ctx.
    int32_t n_tokens = llama_n_ctx(lctx);
    if (out_size_bytes) *out_size_bytes = static_cast<int64_t>(n_embd) * n_tokens * sizeof(float);
    return emb;
}

const void* GgmlEngine::get_attention_weights(int32_t layer_idx,
                                              int64_t* out_size_bytes)
{
    // CPU path: llama.cpp does not expose per-layer attention weight matrices
    // at inference time. Presis falls back to TF-IDF scoring automatically
    // when this returns nullptr.
    (void)layer_idx;
    if (out_size_bytes) *out_size_bytes = 0;
    return nullptr;
}

const void* GgmlEngine::get_kv_cache_tensor(int32_t layer_idx, int32_t type,
                                            int64_t out_shape[4])
{
    // GgmlEngine uses llama.cpp's unified KV cache — no paged-block layout.
    // VQ / KIVI GPU kernels are not applicable on the CPU path.
    (void)layer_idx; (void)type;
    if (out_shape) { out_shape[0]=out_shape[1]=out_shape[2]=out_shape[3]=0; }
    return nullptr;
}

} // namespace truellm
