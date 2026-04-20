#pragma once
// ---------------------------------------------------------------------------
// compression_scoring_pass.h — Pre-prefill token reduction (engine-internal)
//
// Handles Presis (attention-weight-based token pruning) and ToMe (hidden-state
// bipartite token merging).  FastV and PyramidDrop are NOT handled here —
// run_forward_megakernel() reads them directly from inf_cfg_.compression and
// passes them as MegakernelArgs fields (already correct).
//
// Flow inside CudaEngine::generate():
//
//   1. compress_tokens() is called with the incoming token sequence.
//      If attention weights / hidden states from a PRIOR forward pass are
//      cached (see update_cache()), use them for scoring.
//      Otherwise fall back to TF-IDF for Presis.  ToMe is skipped on first call.
//
//   2. The (possibly reduced) token sequence is passed to the prefill forward.
//
//   3. After prefill, update_cache() copies the newly captured d_mk_attn_weights_
//      and d_mk_hidden_states_ device pointers so they are available for the
//      next generate() call.
//
// Why engine-internal (not a plugin):
//   compression_preprocessor.so accessed attention weights / hidden states
//   via host->get_attention_weights() / host->get_hidden_states(), adding
//   vtable overhead and requiring three ABI entries (get_attention_weights,
//   get_hidden_states, set_context_metadata) that exist solely for this
//   first-party component.  As an engine member, direct pointer access
//   replaces all of that.
//
// External plugin authors who need attention weight access (interpretability
// tools, custom scoring) should use host->get_attention_weights() — that
// vtable entry is intentionally preserved.
// ---------------------------------------------------------------------------

#include <truellm/config_schema.h>
#include <vector>
#include <cstdint>

namespace truellm {

class CompressionScoringPass {
public:
    CompressionScoringPass(const CompressionConfig& cfg,
                           int n_heads, int head_dim, int hidden_size);

    // Called before prefill inside generate().
    // Returns a (possibly shortened) token sequence.
    //   tokens    : raw tokenised prompt
    //   seq_len   : tokens.size()
    //   kv_seq    : KV slots currently occupied (= seq_len for a new conversation)
    std::vector<int32_t> compress_tokens(
        const std::vector<int32_t>& tokens,
        int seq_len, int kv_seq);

    // Called after a successful megakernel forward pass.
    // Copies the device attention / hidden-state pointers into host-side caches
    // so they are ready for compress_tokens() on the next call.
    //   d_attn_weights  : device ptr [n_heads, seq, kv_seq] F32 (may be null)
    //   d_hidden_states : device ptr [seq, hidden_size] F16     (may be null)
    //   seq_len, kv_seq : dimensions of the captured tensors
    void update_cache(const void* d_attn_weights,
                      const void* d_hidden_states,
                      int seq_len, int kv_seq);

    bool presis_enabled() const { return presis_cfg_.enabled; }
    bool tome_enabled()   const { return tome_cfg_.enabled;   }
    bool any_enabled()    const { return presis_cfg_.enabled || tome_cfg_.enabled; }

    // ---- Test hook ----------------------------------------------------------
    // Populates the host-side caches directly, skipping the cudaMemcpy that
    // update_cache() performs from device pointers.  Unit tests use this to
    // exercise compress_tokens() without needing a live CUDA forward pass.
    void set_host_caches_for_test(std::vector<float> attn,
                                  std::vector<uint16_t> hidden,
                                  int seq_len, int kv_seq) {
        h_attn_cache_   = std::move(attn);
        h_hidden_cache_ = std::move(hidden);
        cached_seq_len_ = seq_len;
        cached_kv_seq_  = kv_seq;
    }

private:
    // ---- Presis -------------------------------------------------------------
    // Score each token by column-sum of attention weights (or TF-IDF fallback).
    std::vector<float> score_attention(int seq_len, int kv_seq) const;
    std::vector<float> score_tfidf   (const std::vector<int32_t>& tokens) const;
    std::vector<int32_t> apply_presis(const std::vector<int32_t>& tokens,
                                      const std::vector<float>& scores) const;

    // ---- ToMe ---------------------------------------------------------------
    // Bipartite soft matching: merge r pairs of most similar adjacent tokens.
    std::vector<int32_t> apply_tome(const std::vector<int32_t>& tokens,
                                    int seq_len) const;

    // Config
    PresisConfig presis_cfg_;
    ToMeConfig   tome_cfg_;
    int n_heads_, head_dim_, hidden_size_;

    // Host-side caches populated by update_cache() after each forward pass.
    // These hold copies of the device tensors brought to host for CPU scoring.
    std::vector<float>    h_attn_cache_;    // [n_heads * seq * kv_seq] F32
    std::vector<uint16_t> h_hidden_cache_;  // [seq * hidden_size] as raw F16 bits
    int cached_seq_len_ = 0;
    int cached_kv_seq_  = 0;
};

} // namespace truellm
