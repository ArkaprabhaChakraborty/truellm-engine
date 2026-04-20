// ---------------------------------------------------------------------------
// test_kivi_bounds.cpp — pure bounds / size-math tests for KIVI
//
// These exercise host-callable helpers only; no CUDA launches are performed.
// The kivi_cache.cuh header pulls in <cuda_fp16.h> so this TU still must be
// compiled with a CUDA toolkit available, but no device code runs.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "backends/cuda/compression/kivi_cache.cuh"

using truellm::KIVI_MAX_HEAD_DIM;
using truellm::KIVI_MAX_OUTLIERS;
using truellm::kivi_compressed_bytes;
using truellm::kivi_params_supported;

// ---------------------------------------------------------------------------
// kivi_compressed_bytes math check.
// For head_dim=128, n_outliers=8, group_size=16:
//   outliers   : 8 * 3 = 24 bytes
//   residuals  : (120 / 16) groups × (8 + 4) bytes = 7.5 → ceil(8) × 12 = 96
//     Wait — n_groups = ceil(120 / 16) = 8; 8 × 12 = 96.  Total 120 bytes.
// ---------------------------------------------------------------------------
TEST(KiviBoundsTest, CompressedBytesMatchesLayout)
{
    EXPECT_EQ(kivi_compressed_bytes(128, 8, 16), 24u + 8u * 12u);
    EXPECT_EQ(kivi_compressed_bytes(64,  0, 16), 4u * 12u);  // no outliers
    EXPECT_EQ(kivi_compressed_bytes(64, 64, 16), 64u * 3u);  // all outliers
}

// ---------------------------------------------------------------------------
// kivi_params_supported enforces the stack-array caps.  Fix for edge case #1.
// ---------------------------------------------------------------------------
TEST(KiviBoundsTest, ParamsSupportedAcceptsDefaults)
{
    EXPECT_TRUE(kivi_params_supported(128, 8, 16));
    EXPECT_TRUE(kivi_params_supported(64,  4, 16));
    EXPECT_TRUE(kivi_params_supported(KIVI_MAX_HEAD_DIM,
                                      KIVI_MAX_OUTLIERS, 16));
}

TEST(KiviBoundsTest, ParamsSupportedRejectsOutOfRange)
{
    // head_dim > max → kernel stack arrays would overflow
    EXPECT_FALSE(kivi_params_supported(KIVI_MAX_HEAD_DIM + 1, 8, 16));
    // n_outliers > max
    EXPECT_FALSE(kivi_params_supported(128, KIVI_MAX_OUTLIERS + 1, 16));
    // n_outliers > head_dim (nonsense)
    EXPECT_FALSE(kivi_params_supported(16, 32, 16));
    // zero / negative group_size → division by zero downstream
    EXPECT_FALSE(kivi_params_supported(128, 8, 0));
    EXPECT_FALSE(kivi_params_supported(128, 8, -4));
    // non-positive head_dim
    EXPECT_FALSE(kivi_params_supported(0, 0, 16));
    EXPECT_FALSE(kivi_params_supported(-1, 0, 16));
}

// ---------------------------------------------------------------------------
// KIVI_MAX_HEAD_DIM / KIVI_MAX_OUTLIERS must match the on-stack array sizes
// inside kivi_cache.cu.  If someone bumps the constants in the header but
// forgets the arrays (or vice-versa), this catches it.  Not a deep check,
// but documents the intended invariant.
// ---------------------------------------------------------------------------
TEST(KiviBoundsTest, ConstantsAreExpectedValues)
{
    EXPECT_EQ(KIVI_MAX_HEAD_DIM, 128);
    EXPECT_EQ(KIVI_MAX_OUTLIERS, 64);
}
