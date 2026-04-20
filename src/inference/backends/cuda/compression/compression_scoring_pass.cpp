// ---------------------------------------------------------------------------
// compression_scoring_pass.cpp — Presis + ToMe token reduction
// ---------------------------------------------------------------------------

#include "compression_scoring_pass.h"
#include <spdlog/spdlog.h>

#ifdef TRUELLM_HAS_CUDA
#  include <cuda_runtime.h>
#  include <cuda_fp16.h>
#endif

#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>

namespace truellm {

CompressionScoringPass::CompressionScoringPass(
    const CompressionConfig& cfg,
    int n_heads, int head_dim, int hidden_size)
    : presis_cfg_(cfg.presis)
    , tome_cfg_  (cfg.tome)
    , n_heads_   (n_heads)
    , head_dim_  (head_dim)
    , hidden_size_(hidden_size)
{}

// ---------------------------------------------------------------------------
// update_cache — copy device tensors to host after each megakernel forward
// ---------------------------------------------------------------------------
void CompressionScoringPass::update_cache(
    const void* d_attn_weights,
    const void* d_hidden_states,
    int seq_len, int kv_seq)
{
#ifdef TRUELLM_HAS_CUDA
    cached_seq_len_ = seq_len;
    cached_kv_seq_  = kv_seq;

    if (d_attn_weights && presis_cfg_.enabled) {
        size_t bytes = static_cast<size_t>(n_heads_) * seq_len * kv_seq * sizeof(float);
        h_attn_cache_.resize(static_cast<size_t>(n_heads_) * seq_len * kv_seq);
        cudaMemcpy(h_attn_cache_.data(), d_attn_weights,
                   bytes, cudaMemcpyDeviceToHost);
    }

    if (d_hidden_states && tome_cfg_.enabled) {
        size_t bytes = static_cast<size_t>(seq_len) * hidden_size_ * sizeof(uint16_t);
        h_hidden_cache_.resize(static_cast<size_t>(seq_len) * hidden_size_);
        cudaMemcpy(h_hidden_cache_.data(), d_hidden_states,
                   bytes, cudaMemcpyDeviceToHost);
    }
#else
    (void)d_attn_weights; (void)d_hidden_states; (void)seq_len; (void)kv_seq;
#endif
}

// ---------------------------------------------------------------------------
// compress_tokens — entry point called before prefill
// ---------------------------------------------------------------------------
std::vector<int32_t> CompressionScoringPass::compress_tokens(
    const std::vector<int32_t>& tokens,
    int seq_len, int kv_seq)
{
    if (tokens.empty()) return tokens;

    std::vector<int32_t> result = tokens;

    // ToMe first (Arch Invariant #11: ToMe before Presis).
    // The cached hidden-state buffer was captured at cached_seq_len_ tokens.
    // If the current batch differs in length, the position ↔ embedding mapping
    // is wrong; skip ToMe rather than merge truncated / misaligned tokens.
    if (tome_cfg_.enabled && cached_seq_len_ > 0 && !h_hidden_cache_.empty()) {
        if (cached_seq_len_ != seq_len) {
            spdlog::debug("[CompressionScoringPass] ToMe skipped: cached_seq={} "
                          "!= current seq={} (cache mismatch)",
                          cached_seq_len_, seq_len);
        } else {
            result = apply_tome(result, static_cast<int>(result.size()));
            spdlog::debug("[CompressionScoringPass] ToMe: {} → {} tokens",
                          seq_len, result.size());
        }
    }

    // Presis
    if (presis_cfg_.enabled) {
        std::vector<float> scores;
        // Attention cache is only valid when dimensions match the current
        // batch AND the cache is non-empty.  A prior request with a different
        // seq_len leaves stale weights; fall back to TF-IDF instead.
        bool have_attn = (!h_attn_cache_.empty()
                          && cached_seq_len_ > 0
                          && cached_seq_len_ == static_cast<int>(result.size())
                          && seq_len <= presis_cfg_.max_seqlen_for_attn);
        const char* src = "tfidf";
        if (have_attn) {
            scores = score_attention(static_cast<int>(result.size()), kv_seq);
            // If attention weights were all zero (degenerate / NaN forward),
            // column sums give a max of 0 and scores stay at 0, which would
            // make Presis pick tokens arbitrarily.  Detect and fall back.
            float mx = 0.f;
            for (float s : scores) if (s > mx) mx = s;
            if (mx <= 0.f) {
                scores = score_tfidf(result);
                src = "tfidf(attn-degenerate)";
            } else {
                src = "attn";
            }
        } else {
            scores = score_tfidf(result);
        }
        auto pruned = apply_presis(result, scores);
        spdlog::debug("[CompressionScoringPass] Presis({}): {} → {} tokens",
                      src, result.size(), pruned.size());
        result = std::move(pruned);
    }

    return result;
}

// ---------------------------------------------------------------------------
// score_attention — column-sum importance from cached attention weights
//
// h_attn_cache_ layout: [n_heads, cached_seq_len_, cached_kv_seq_] F32
// We want per-token importance for `seq_len` tokens from the current call.
// If dimensions don't match (e.g. conversation grew), fall back to TF-IDF.
// ---------------------------------------------------------------------------
std::vector<float> CompressionScoringPass::score_attention(
    int seq_len, int /*kv_seq*/) const
{
    std::vector<float> scores(seq_len, 0.f);

    int cached_s  = cached_seq_len_;
    int cached_kv = cached_kv_seq_;

    // Cache malformed / smaller than declared dims → caller will fall back.
    if (static_cast<int>(h_attn_cache_.size())
        < static_cast<size_t>(n_heads_) * cached_s * cached_kv)
        return scores;  // all zeros → caller falls back to TF-IDF

    // Column-sum across all heads and query positions:
    // importance[kv_pos] = sum over (head, q_pos) of attn[head, q_pos, kv_pos]
    int use_kv = std::min(seq_len, cached_kv);
    for (int h = 0; h < n_heads_; ++h) {
        for (int q = 0; q < cached_s; ++q) {
            const float* row = h_attn_cache_.data()
                             + (h * cached_s + q) * cached_kv;
            for (int k = 0; k < use_kv; ++k)
                scores[k] += row[k];
        }
    }

    // Normalise so max score = 1.  Leave scores at 0 if everything is zero —
    // the caller detects this and falls back to TF-IDF.
    float mx = *std::max_element(scores.begin(), scores.end());
    if (mx > 0.f) for (auto& s : scores) s /= mx;
    return scores;
}

// ---------------------------------------------------------------------------
// score_tfidf — fallback: higher token_id → rarer → higher importance
// ---------------------------------------------------------------------------
std::vector<float> CompressionScoringPass::score_tfidf(
    const std::vector<int32_t>& tokens) const
{
    if (tokens.empty()) return {};
    // Term frequency: count each token id
    std::unordered_map<int32_t,int> freq;
    for (int32_t t : tokens) freq[t]++;
    int N = static_cast<int>(tokens.size());

    std::vector<float> scores(tokens.size());
    for (int i = 0; i < N; ++i) {
        float tf  = 1.f / static_cast<float>(freq[tokens[i]]);
        // IDF heuristic: rarer token ids are more informative
        float idf = std::log(1.f + static_cast<float>(tokens[i]) / 32000.f);
        scores[i] = tf * idf;
    }
    // Normalise
    float mx = *std::max_element(scores.begin(), scores.end());
    if (mx > 0.f) for (auto& s : scores) s /= mx;
    return scores;
}

// ---------------------------------------------------------------------------
// apply_presis — remove tokens whose score < threshold, keeping the top K%
// System tokens (first token) and last N tokens are always kept.
// ---------------------------------------------------------------------------
std::vector<int32_t> CompressionScoringPass::apply_presis(
    const std::vector<int32_t>& tokens,
    const std::vector<float>&   scores) const
{
    int N = static_cast<int>(tokens.size());
    if (N == 0 || scores.size() != static_cast<size_t>(N)) return tokens;

    int keep_n = static_cast<int>(std::ceil(N * presis_cfg_.keep_fraction));
    // Always keep at least 1, never exceed N
    keep_n = std::max(1, std::min(keep_n, N));

    // Build sorted index by score descending
    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + keep_n, idx.end(),
        [&](int a, int b){ return scores[a] > scores[b]; });

    // Mark which positions to keep
    std::vector<bool> keep(N, false);
    for (int i = 0; i < keep_n; ++i) keep[idx[i]] = true;

    // Always keep first token (BOS / system) and last 8 tokens
    keep[0] = true;
    for (int i = std::max(0, N - 8); i < N; ++i) keep[i] = true;

    std::vector<int32_t> result;
    result.reserve(keep_n);
    for (int i = 0; i < N; ++i)
        if (keep[i]) result.push_back(tokens[i]);
    return result;
}

// ---------------------------------------------------------------------------
// apply_tome — bipartite soft matching: merge r pairs of most-similar tokens
//
// Uses cosine similarity between adjacent hidden-state vectors.
// Partitions tokens into even (A-set) and odd (B-set), greedily matches
// each A token to its most similar B token, removes the B token (merge).
// Falls back gracefully if hidden cache is empty or wrong size.
// ---------------------------------------------------------------------------
std::vector<int32_t> CompressionScoringPass::apply_tome(
    const std::vector<int32_t>& tokens,
    int seq_len) const
{
    int r = tome_cfg_.r;
    if (r <= 0 || h_hidden_cache_.empty()) return tokens;

    int cache_seq = cached_seq_len_;
    int use_seq   = std::min(seq_len, cache_seq);
    if (use_seq < 2) return tokens;

    // Get F32 norms for similarity computation from F16 hidden cache
    auto get_vec = [&](int pos, std::vector<float>& out) {
        out.resize(hidden_size_);
        const uint16_t* src = h_hidden_cache_.data() + pos * hidden_size_;
        for (int d = 0; d < hidden_size_; ++d) {
            // manual F16 → F32 without __half2float (CPU-only, no CUDA headers needed)
            uint16_t bits = src[d];
            uint32_t sign = (bits >> 15) & 1;
            uint32_t exp  = (bits >> 10) & 0x1F;
            uint32_t mant = bits & 0x3FF;
            float f;
            if (exp == 0 && mant == 0) { f = 0.f; }
            else if (exp == 31) { f = (mant == 0) ? (sign ? -INFINITY : INFINITY) : NAN; }
            else {
                uint32_t fbits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
                memcpy(&f, &fbits, 4);
            }
            out[d] = f;
        }
    };

    auto cosine = [&](const std::vector<float>& a, const std::vector<float>& b) -> float {
        float dot = 0.f, na = 0.f, nb = 0.f;
        for (int d = 0; d < hidden_size_; ++d) {
            dot += a[d] * b[d]; na += a[d]*a[d]; nb += b[d]*b[d];
        }
        constexpr float kEps = 1e-12f;
        // Both vectors ~zero → treat as identical (avoids the biased
        // "always merge zero pairs" behaviour of returning 0).  Exactly one
        // side zero → 0 (they carry no comparable signal).
        if (na < kEps && nb < kEps) return 1.f;
        if (na < kEps || nb < kEps) return 0.f;
        return dot / std::sqrt(na * nb);
    };

    // Bipartite: A = even indices, B = odd indices (within use_seq)
    std::vector<bool> removed(use_seq, false);
    int merges = 0;

    for (int a = 0; a < use_seq - 1 && merges < r; a += 2) {
        int b_best = -1;
        float best_sim = -2.f;
        for (int b = 1; b < use_seq && merges < r; b += 2) {
            if (removed[b]) continue;
            std::vector<float> va, vb;
            get_vec(a, va); get_vec(b, vb);
            float sim = cosine(va, vb);
            if (sim > best_sim) { best_sim = sim; b_best = b; }
        }
        if (b_best >= 0 && best_sim > 0.f) {
            removed[b_best] = true;
            ++merges;
        }
    }

    std::vector<int32_t> result;
    result.reserve(seq_len - merges);
    for (int i = 0; i < seq_len; ++i) {
        if (i < use_seq && removed[i]) continue;
        result.push_back(tokens[i]);
    }
    return result;
}

} // namespace truellm
