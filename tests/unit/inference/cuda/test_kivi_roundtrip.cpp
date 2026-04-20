// ---------------------------------------------------------------------------
// test_kivi_roundtrip.cpp — GPU round-trip tests for KIVI compress/decompress.
//
// Exercises the full kernel path (compress → decompress) and checks the
// reconstruction error stays within the quantization bound dictated by the
// INT4 group size.  Skips via GTEST_SKIP() when no CUDA device is present.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "backends/cuda/compression/kivi_cache.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <vector>

using truellm::kivi_compress;
using truellm::kivi_decompress;
using truellm::kivi_compressed_bytes;

namespace {

class KiviRoundtripTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        if (err != cudaSuccess || count <= 0) {
            GTEST_SKIP() << "No CUDA device available";
        }
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess);
    }
    void TearDown() override
    {
        if (stream_) cudaStreamDestroy(stream_);
        for (void* p : allocs_) cudaFree(p);
    }

    __half* upload_f16(const std::vector<float>& host)
    {
        std::vector<__half> tmp(host.size());
        for (size_t i = 0; i < host.size(); ++i) tmp[i] = __float2half(host[i]);
        __half* d = nullptr;
        size_t bytes = tmp.size() * sizeof(__half);
        EXPECT_EQ(cudaMalloc(&d, bytes), cudaSuccess);
        EXPECT_EQ(cudaMemcpy(d, tmp.data(), bytes, cudaMemcpyHostToDevice),
                  cudaSuccess);
        allocs_.push_back(d);
        return d;
    }

    template <typename T> T* alloc_device(size_t n)
    {
        T* d = nullptr;
        EXPECT_EQ(cudaMalloc(&d, n * sizeof(T)), cudaSuccess);
        allocs_.push_back(d);
        return d;
    }

    std::vector<float> download_f16_as_f32(const __half* d, size_t n)
    {
        std::vector<__half> tmp(n);
        EXPECT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
        EXPECT_EQ(cudaMemcpy(tmp.data(), d, n * sizeof(__half),
                             cudaMemcpyDeviceToHost), cudaSuccess);
        std::vector<float> out(n);
        for (size_t i = 0; i < n; ++i) out[i] = __half2float(tmp[i]);
        return out;
    }

    cudaStream_t       stream_ = nullptr;
    std::vector<void*> allocs_;
};

// ---------------------------------------------------------------------------
// Round-trip on smooth data: max abs error should be ~range/15 per group,
// dominated by INT4 granularity.  We pick head_dim=64, n_outliers=0, group=16.
// ---------------------------------------------------------------------------
TEST_F(KiviRoundtripTest, RoundtripNoOutliersRespectsInt4Granularity)
{
    constexpr int N_VECS = 3;
    constexpr int HEAD_DIM = 64;
    constexpr int N_OUT = 0;
    constexpr int GROUP = 16;

    std::vector<float> kv_h(N_VECS * HEAD_DIM);
    for (int v = 0; v < N_VECS; ++v) {
        for (int d = 0; d < HEAD_DIM; ++d) {
            // smooth ramp per vector, different offset per vec
            kv_h[v * HEAD_DIM + d] = 0.1f * d + 0.5f * v;
        }
    }

    __half*  d_in  = upload_f16(kv_h);
    size_t   pack_bytes = kivi_compressed_bytes(HEAD_DIM, N_OUT, GROUP);
    uint8_t* d_pack = alloc_device<uint8_t>(N_VECS * pack_bytes);
    __half*  d_out = alloc_device<__half>(N_VECS * HEAD_DIM);

    kivi_compress(d_in, d_pack, N_VECS, HEAD_DIM, N_OUT, GROUP, stream_);
    kivi_decompress(d_pack, d_out, N_VECS, HEAD_DIM, N_OUT, GROUP, stream_);

    auto got = download_f16_as_f32(d_out, N_VECS * HEAD_DIM);

    // Per-group max range for our data: 0.1f * (GROUP-1) ≈ 1.5, so worst-case
    // step ≈ 1.5/15 = 0.1. F16 rounding + step slop → 0.15.
    for (size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], kv_h[i], 0.15f) << "i=" << i;
    }
}

// ---------------------------------------------------------------------------
// With outliers: a handful of channels have |value| much larger than the
// rest.  Those should survive the round trip nearly exactly (only F16
// rounding), because they're stored verbatim.  Residual channels obey the
// INT4 quantization bound.
// ---------------------------------------------------------------------------
TEST_F(KiviRoundtripTest, RoundtripWithOutliersPreservesLargeChannels)
{
    constexpr int N_VECS = 2;
    constexpr int HEAD_DIM = 64;
    constexpr int N_OUT = 4;
    constexpr int GROUP = 16;

    std::vector<float> kv_h(N_VECS * HEAD_DIM);
    // Base: small noise ramp.
    for (int i = 0; i < N_VECS * HEAD_DIM; ++i)
        kv_h[i] = 0.01f * (i % HEAD_DIM);
    // Inject 4 outliers per vector at known positions.
    const int outlier_channels[N_OUT] = {3, 17, 33, 55};
    for (int v = 0; v < N_VECS; ++v) {
        for (int k = 0; k < N_OUT; ++k) {
            int ch = outlier_channels[k];
            kv_h[v * HEAD_DIM + ch] = 50.0f + k + v;  // large
        }
    }

    __half*  d_in   = upload_f16(kv_h);
    size_t   pack_bytes = kivi_compressed_bytes(HEAD_DIM, N_OUT, GROUP);
    uint8_t* d_pack = alloc_device<uint8_t>(N_VECS * pack_bytes);
    __half*  d_out  = alloc_device<__half>(N_VECS * HEAD_DIM);

    kivi_compress(d_in, d_pack, N_VECS, HEAD_DIM, N_OUT, GROUP, stream_);
    kivi_decompress(d_pack, d_out, N_VECS, HEAD_DIM, N_OUT, GROUP, stream_);

    auto got = download_f16_as_f32(d_out, N_VECS * HEAD_DIM);

    for (int v = 0; v < N_VECS; ++v) {
        for (int k = 0; k < N_OUT; ++k) {
            int ch = outlier_channels[k];
            int idx = v * HEAD_DIM + ch;
            // Outlier stored as F16 verbatim → only rounding error.
            EXPECT_NEAR(got[idx], kv_h[idx], 0.1f)
                << "outlier v=" << v << " ch=" << ch;
        }
        for (int d = 0; d < HEAD_DIM; ++d) {
            bool is_out = false;
            for (int k = 0; k < N_OUT; ++k) if (outlier_channels[k] == d) is_out = true;
            if (is_out) continue;
            // Residual tolerance: the non-outlier ramp has range ~0.63 per
            // group, step ~0.042, so 0.1 leaves slack for F16 rounding.
            int idx = v * HEAD_DIM + d;
            EXPECT_NEAR(got[idx], kv_h[idx], 0.1f)
                << "residual v=" << v << " d=" << d;
        }
    }
}

// ---------------------------------------------------------------------------
// Degenerate group (all residual values equal) must round-trip to the
// constant.  This exercises the `range <= 1e-6f` branch in the compress
// kernel, which writes scale=0 and zero=gmin.  Decompress must reproduce
// gmin for every channel in the group.
// ---------------------------------------------------------------------------
TEST_F(KiviRoundtripTest, RoundtripDegenerateGroupProducesConstant)
{
    constexpr int N_VECS = 1;
    constexpr int HEAD_DIM = 32;
    constexpr int N_OUT = 0;
    constexpr int GROUP = 16;

    // All residual values = 2.5 → range == 0 per group.
    std::vector<float> kv_h(HEAD_DIM, 2.5f);

    __half*  d_in   = upload_f16(kv_h);
    size_t   pack_bytes = kivi_compressed_bytes(HEAD_DIM, N_OUT, GROUP);
    uint8_t* d_pack = alloc_device<uint8_t>(N_VECS * pack_bytes);
    __half*  d_out  = alloc_device<__half>(N_VECS * HEAD_DIM);

    kivi_compress(d_in, d_pack, N_VECS, HEAD_DIM, N_OUT, GROUP, stream_);
    kivi_decompress(d_pack, d_out, N_VECS, HEAD_DIM, N_OUT, GROUP, stream_);

    auto got = download_f16_as_f32(d_out, HEAD_DIM);
    for (size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], 2.5f, 1e-2f) << "i=" << i;
    }
}

// ---------------------------------------------------------------------------
// Out-of-range params must degrade host wrapper to no-op (see
// kivi_params_supported).  The output buffer we zero beforehand should stay
// zero — i.e. no kernel launch happened.
// ---------------------------------------------------------------------------
TEST_F(KiviRoundtripTest, CompressNoOpsOnUnsupportedParams)
{
    constexpr int N_VECS = 1;
    constexpr int HEAD_DIM = 256;   // > KIVI_MAX_HEAD_DIM (128)
    constexpr int N_OUT = 0;
    constexpr int GROUP = 16;

    std::vector<float> kv_h(HEAD_DIM, 1.0f);
    __half* d_in = upload_f16(kv_h);

    size_t   pack_bytes = kivi_compressed_bytes(HEAD_DIM, N_OUT, GROUP);
    uint8_t* d_pack = alloc_device<uint8_t>(N_VECS * pack_bytes);
    ASSERT_EQ(cudaMemset(d_pack, 0xAB, N_VECS * pack_bytes), cudaSuccess);

    kivi_compress(d_in, d_pack, N_VECS, HEAD_DIM, N_OUT, GROUP, stream_);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    std::vector<uint8_t> pack_h(pack_bytes);
    ASSERT_EQ(cudaMemcpy(pack_h.data(), d_pack, pack_bytes,
                         cudaMemcpyDeviceToHost), cudaSuccess);
    // Sentinel still present → wrapper was a no-op, as intended.
    for (uint8_t b : pack_h) EXPECT_EQ(b, 0xAB);
}

} // namespace
