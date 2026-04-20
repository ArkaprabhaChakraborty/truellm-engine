// ---------------------------------------------------------------------------
// test_compression_scoring_pass.cpp — host-side tests for CompressionScoringPass
//
// Exercises the Presis / ToMe logic without needing a live CUDA forward pass.
// The pass reads pre-captured attention weights and hidden states from host
// caches; set_host_caches_for_test() lets us inject them directly.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "backends/cuda/compression/compression_scoring_pass.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

using truellm::CompressionConfig;
using truellm::CompressionScoringPass;

namespace {

// Encode a float as raw IEEE-754 half-precision bits (matches the CPU-side
// decoder in compression_scoring_pass.cpp).  Not a full-fidelity conversion
// — adequate for the limited range used in these tests.
uint16_t f32_to_f16_bits(float f)
{
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 31) & 0x1;
    int32_t  exp  = static_cast<int32_t>((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (x >> 13) & 0x3FF;
    if (exp <= 0) return static_cast<uint16_t>(sign << 15);  // flush to zero
    if (exp >= 31) return static_cast<uint16_t>((sign << 15) | (31 << 10));
    return static_cast<uint16_t>((sign << 15) | (exp << 10) | mant);
}

CompressionConfig make_cfg(bool presis_on, bool tome_on,
                           float keep_fraction = 0.5f, int r = 1)
{
    CompressionConfig cc;
    cc.enabled = presis_on || tome_on;
    cc.presis.enabled       = presis_on;
    cc.presis.keep_fraction = keep_fraction;
    cc.presis.max_seqlen_for_attn = 4096;
    cc.tome.enabled = tome_on;
    cc.tome.r       = r;
    return cc;
}

constexpr int kNHeads     = 2;
constexpr int kHeadDim    = 8;
constexpr int kHiddenSize = 16;

std::vector<float> flat_attn(int n_heads, int seq, int kv, float value)
{
    return std::vector<float>(static_cast<size_t>(n_heads) * seq * kv, value);
}

std::vector<uint16_t> hidden_uniform(int seq, int hidden, float value)
{
    std::vector<uint16_t> out(static_cast<size_t>(seq) * hidden,
                              f32_to_f16_bits(value));
    return out;
}

} // namespace

// ---------------------------------------------------------------------------
// Presis with no cached attention weights falls back to TF-IDF.  TF-IDF
// produces deterministic scores per token id, so the output is a strict
// subset of the input with keep_fraction enforced.
// ---------------------------------------------------------------------------
TEST(CompressionScoringPassTest, PresisFallsBackToTfidfWhenNoAttentionCache)
{
    CompressionConfig cc = make_cfg(/*presis=*/true, /*tome=*/false, 0.5f);
    CompressionScoringPass pass(cc, kNHeads, kHeadDim, kHiddenSize);

    std::vector<int32_t> tokens = {1, 2, 3, 4, 5, 6, 7, 8};
    auto out = pass.compress_tokens(tokens, static_cast<int>(tokens.size()),
                                    static_cast<int>(tokens.size()));

    EXPECT_LE(out.size(), tokens.size());
    EXPECT_GE(out.size(), 1u);
    // BOS (index 0) and last 8 are always preserved per apply_presis.
    ASSERT_FALSE(out.empty());
    EXPECT_EQ(out.front(), tokens.front());
}

// ---------------------------------------------------------------------------
// When attention weights are all zero (degenerate forward / NaN), Presis must
// detect a zero max and fall back to TF-IDF rather than returning an all-zero
// score vector that would let apply_presis pick tokens arbitrarily.
// Fix corresponds to edge case #2 in the audit.
// ---------------------------------------------------------------------------
TEST(CompressionScoringPassTest, ZeroRangeAttentionFallsBackToTfidf)
{
    CompressionConfig cc = make_cfg(true, false, 0.5f);
    CompressionScoringPass pass(cc, kNHeads, kHeadDim, kHiddenSize);

    std::vector<int32_t> tokens = {10, 20, 30, 40, 50, 60};
    const int S = static_cast<int>(tokens.size());
    pass.set_host_caches_for_test(flat_attn(kNHeads, S, S, 0.f),
                                  /*hidden=*/{}, S, S);

    auto out = pass.compress_tokens(tokens, S, S);

    // With zero attention, code must still have produced a non-empty result
    // via the tfidf fallback; size should respect keep_fraction (+protected
    // BOS / tail tokens).
    EXPECT_FALSE(out.empty());
    EXPECT_LE(out.size(), tokens.size());
    EXPECT_EQ(out.front(), tokens.front());
}

// ---------------------------------------------------------------------------
// ToMe must be skipped when the current batch size differs from the cached
// hidden-state sequence length (the cache is from a prior forward pass that
// saw a different batch).  Fix corresponds to edge case #5.
// ---------------------------------------------------------------------------
TEST(CompressionScoringPassTest, ToMeSkippedOnCacheSeqLenMismatch)
{
    CompressionConfig cc = make_cfg(/*presis=*/false, /*tome=*/true,
                                    /*keep=*/1.0f, /*r=*/2);
    CompressionScoringPass pass(cc, kNHeads, kHeadDim, kHiddenSize);

    // Cache captured with seq_len = 6, but we pass 4 tokens now.
    pass.set_host_caches_for_test(/*attn=*/{}, hidden_uniform(6, kHiddenSize, 1.f),
                                  /*seq_len=*/6, /*kv_seq=*/6);

    std::vector<int32_t> tokens = {1, 2, 3, 4};
    auto out = pass.compress_tokens(tokens, static_cast<int>(tokens.size()),
                                    static_cast<int>(tokens.size()));
    // Skipped → no merges → identity.
    EXPECT_EQ(out, tokens);
}

// ---------------------------------------------------------------------------
// ToMe applied: identical hidden vectors for every position → every pair is
// maximally similar → exactly `r` merges occur.
// ---------------------------------------------------------------------------
TEST(CompressionScoringPassTest, ToMeMergesRPairsWhenCacheMatches)
{
    const int r = 2;
    CompressionConfig cc = make_cfg(false, true, 1.0f, r);
    CompressionScoringPass pass(cc, kNHeads, kHeadDim, kHiddenSize);

    const int S = 6;
    pass.set_host_caches_for_test(/*attn=*/{}, hidden_uniform(S, kHiddenSize, 1.f),
                                  S, S);

    std::vector<int32_t> tokens = {11, 12, 13, 14, 15, 16};
    auto out = pass.compress_tokens(tokens, S, S);
    // Exactly r=2 merges on a 6-token sequence → 4 tokens survive.
    EXPECT_EQ(out.size(), tokens.size() - r);
}

// ---------------------------------------------------------------------------
// Cosine similarity with both vectors ~zero now returns 1.0 (treated as
// identical, pair is eligible for merging).  Previously the function
// returned 0, biasing ToMe away from merging zero-norm pairs despite them
// being semantically indistinguishable.  Fix corresponds to edge case #12.
// Verifies behaviour indirectly: an all-zero hidden cache still produces
// the configured number of ToMe merges.
// ---------------------------------------------------------------------------
TEST(CompressionScoringPassTest, ToMeCosineZeroNormStillMerges)
{
    const int r = 1;
    CompressionConfig cc = make_cfg(false, true, 1.0f, r);
    CompressionScoringPass pass(cc, kNHeads, kHeadDim, kHiddenSize);

    const int S = 4;
    pass.set_host_caches_for_test(/*attn=*/{}, hidden_uniform(S, kHiddenSize, 0.f),
                                  S, S);

    std::vector<int32_t> tokens = {1, 2, 3, 4};
    auto out = pass.compress_tokens(tokens, S, S);
    // With cosine(0,0)=1, at least one merge must have happened.
    EXPECT_LT(out.size(), tokens.size());
    EXPECT_EQ(out.size(), tokens.size() - r);
}

// ---------------------------------------------------------------------------
// Stale attention weights from a prior forward pass with a different seq_len
// must be ignored; Presis falls back to TF-IDF instead of slicing / aliasing
// the cache.  Fix corresponds to edge case #10.
// ---------------------------------------------------------------------------
TEST(CompressionScoringPassTest, StaleAttentionCacheFallsBack)
{
    CompressionConfig cc = make_cfg(true, false, 0.5f);
    CompressionScoringPass pass(cc, kNHeads, kHeadDim, kHiddenSize);

    // Cache captured with seq=16, but incoming batch has seq=8.  Populating a
    // non-empty attention cache at a mismatched dim would, without the guard,
    // pass the non-empty check and be used to score the wrong positions.
    pass.set_host_caches_for_test(flat_attn(kNHeads, 16, 16, 1.f),
                                  /*hidden=*/{}, 16, 16);

    std::vector<int32_t> tokens = {1, 2, 3, 4, 5, 6, 7, 8};
    auto out = pass.compress_tokens(tokens, static_cast<int>(tokens.size()),
                                    static_cast<int>(tokens.size()));

    // Must still produce a reasonable (non-empty, bounded) output.
    EXPECT_FALSE(out.empty());
    EXPECT_LE(out.size(), tokens.size());
}

// ---------------------------------------------------------------------------
// Empty tokens → empty result (guard against spurious fallbacks writing past
// the end of TF-IDF's score vector).
// ---------------------------------------------------------------------------
TEST(CompressionScoringPassTest, EmptyTokensPassThrough)
{
    CompressionConfig cc = make_cfg(true, true);
    CompressionScoringPass pass(cc, kNHeads, kHeadDim, kHiddenSize);
    auto out = pass.compress_tokens({}, 0, 0);
    EXPECT_TRUE(out.empty());
}

// ---------------------------------------------------------------------------
// Presis keep_fraction=1.0 returns all tokens (plus protected BOS/tail).
// Guards the ceiling math in apply_presis.
// ---------------------------------------------------------------------------
TEST(CompressionScoringPassTest, PresisKeepAllFractionReturnsFullSequence)
{
    CompressionConfig cc = make_cfg(true, false, /*keep_fraction=*/1.0f);
    CompressionScoringPass pass(cc, kNHeads, kHeadDim, kHiddenSize);

    std::vector<int32_t> tokens = {1, 2, 3, 4, 5};
    auto out = pass.compress_tokens(tokens, static_cast<int>(tokens.size()),
                                    static_cast<int>(tokens.size()));
    EXPECT_EQ(out.size(), tokens.size());
}
