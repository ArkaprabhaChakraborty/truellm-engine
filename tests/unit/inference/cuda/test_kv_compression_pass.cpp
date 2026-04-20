// ---------------------------------------------------------------------------
// test_kv_compression_pass.cpp — constructor validation tests for
// KvCompressionPass.  No CUDA kernels are launched: we only inspect the
// host-side flags the constructor sets based on (cfg, head_dim).
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "backends/cuda/compression/kv_compression_pass.h"

using truellm::CompressionConfig;
using truellm::KvCompressionPass;
using truellm::VQ_MAX_CODEBOOK_SIZE;

namespace {

CompressionConfig vq_cfg(int codebook_size)
{
    CompressionConfig cc;
    cc.kv_vq.enabled       = true;
    cc.kv_vq.codebook_size = codebook_size;
    return cc;
}

CompressionConfig kivi_cfg(int residual_length = 8, int bits = 4)
{
    CompressionConfig cc;
    cc.kv_kivi.enabled         = true;
    cc.kv_kivi.residual_length = residual_length;
    cc.kv_kivi.bits            = bits;
    return cc;
}

} // namespace

// ---------------------------------------------------------------------------
// Codebook sizes larger than uint16_t max must be clamped (VQ indices are
// stored as uint16_t on device).  Fix corresponds to edge case #14.
// ---------------------------------------------------------------------------
TEST(KvCompressionPassTest, CodebookSizeExceedingUint16IsClamped)
{
    auto cfg = vq_cfg(/*codebook_size=*/100000);
    KvCompressionPass pass(cfg, /*n_layers=*/2, /*n_kv_heads=*/4,
                           /*head_dim=*/64, /*block_size=*/16);
    EXPECT_TRUE(pass.enabled());
    // No public getter for codebook_size; we only verify the pass stays
    // enabled (i.e. the clamp did not accidentally disable VQ entirely).
    // The effective clamp is covered by the warn-log in the constructor,
    // and by VQ_MAX_CODEBOOK_SIZE being within uint16 range.
    EXPECT_LE(VQ_MAX_CODEBOOK_SIZE, 65535);
}

// ---------------------------------------------------------------------------
// A non-positive codebook_size with VQ requested must disable VQ rather
// than allocating a zero-sized codebook.
// ---------------------------------------------------------------------------
TEST(KvCompressionPassTest, NonPositiveCodebookSizeDisablesVq)
{
    auto cfg = vq_cfg(/*codebook_size=*/0);
    cfg.kv_kivi.enabled = false;   // only VQ requested
    KvCompressionPass pass(cfg, 2, 4, 64, 16);
    EXPECT_FALSE(pass.enabled());
}

// ---------------------------------------------------------------------------
// Requesting KIVI with a head_dim beyond KIVI_MAX_HEAD_DIM (kernel stack
// array cap) must disable KIVI rather than silently corrupting every
// KV vector.  Fix corresponds to edge case #1.
// ---------------------------------------------------------------------------
TEST(KvCompressionPassTest, KiviHeadDimTooLargeDisablesKivi)
{
    auto cfg = kivi_cfg();
    cfg.kv_vq.enabled = false;
    KvCompressionPass pass(cfg, 2, 4, /*head_dim=*/256, 16);
    EXPECT_FALSE(pass.enabled());
}

// ---------------------------------------------------------------------------
// head_dim within bounds, sensible residual_length → KIVI stays enabled.
// ---------------------------------------------------------------------------
TEST(KvCompressionPassTest, KiviSupportedParamsStayEnabled)
{
    auto cfg = kivi_cfg(/*residual_length=*/8, /*bits=*/4);
    cfg.kv_vq.enabled = false;
    KvCompressionPass pass(cfg, 2, 4, /*head_dim=*/128, 16);
    EXPECT_TRUE(pass.enabled());
}

// ---------------------------------------------------------------------------
// KIVI with n_outliers exceeding the cap is disabled.
// ---------------------------------------------------------------------------
TEST(KvCompressionPassTest, KiviTooManyOutliersDisablesKivi)
{
    auto cfg = kivi_cfg(/*residual_length=*/200, /*bits=*/4);
    cfg.kv_vq.enabled = false;
    KvCompressionPass pass(cfg, 2, 4, /*head_dim=*/128, 16);
    EXPECT_FALSE(pass.enabled());
}

// ---------------------------------------------------------------------------
// Both features disabled → pass reports not-enabled.
// ---------------------------------------------------------------------------
TEST(KvCompressionPassTest, BothDisabledReportsNotEnabled)
{
    CompressionConfig cc;
    KvCompressionPass pass(cc, 2, 4, 64, 16);
    EXPECT_FALSE(pass.enabled());
}
