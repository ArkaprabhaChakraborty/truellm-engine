// ---------------------------------------------------------------------------
// test_codebook_init.cpp — GPU convergence test for codebook_init_kmeans.
//
// Generates two far-apart clusters of calibration vectors, seeds the
// codebook with one point near each cluster, and runs a handful of Lloyd
// iterations.  After convergence, each codebook entry should sit at (or
// very near) its cluster centroid, and every sample must be assigned to
// its own cluster by vq_encode.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "backends/cuda/compression/kv_compression_pass.h"  // codebook_init_kmeans
#include "backends/cuda/compression/vq_cache.cuh"           // vq_encode
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <vector>

using truellm::codebook_init_kmeans;
using truellm::vq_encode;

namespace {

class CodebookInitTest : public ::testing::Test {
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
// Two clusters in R^4: one centered near (0,0,0,0), one near (10,10,10,10).
// With a decent initial seed, a few Lloyd iterations should land the book
// entries on each centroid.
// ---------------------------------------------------------------------------
TEST_F(CodebookInitTest, KmeansConvergesOnTwoSeparableClusters)
{
    constexpr int HEAD_DIM = 4;
    constexpr int CBOOK = 2;
    constexpr int N_PER = 32;
    constexpr int N_SAMPLES = CBOOK * N_PER;

    std::vector<float> data(N_SAMPLES * HEAD_DIM);
    for (int i = 0; i < N_PER; ++i) {
        float jitter = 0.05f * (i - N_PER / 2);
        // cluster 0 near origin
        for (int d = 0; d < HEAD_DIM; ++d)
            data[i * HEAD_DIM + d] = jitter + 0.01f * d;
        // cluster 1 near (10,...)
        for (int d = 0; d < HEAD_DIM; ++d)
            data[(N_PER + i) * HEAD_DIM + d] = 10.f + jitter + 0.01f * d;
    }

    // Seed: one near each cluster, but offset enough to make iterations matter.
    std::vector<float> book = {
        0.5f, 0.5f, 0.5f, 0.5f,
        9.0f, 9.0f, 9.0f, 9.0f,
    };

    __half* d_data = upload_f16(data);
    __half* d_book = upload_f16(book);

    codebook_init_kmeans(d_data, d_book,
                         N_SAMPLES, CBOOK, HEAD_DIM,
                         /*n_iterations=*/10, stream_);

    auto got_book = download_f16_as_f32(d_book, CBOOK * HEAD_DIM);

    // Each centroid should be within ~0.2 of the true mean.
    for (int d = 0; d < HEAD_DIM; ++d) {
        EXPECT_NEAR(got_book[0 * HEAD_DIM + d], 0.0f + 0.01f * d, 0.2f)
            << "cluster-0 d=" << d;
        EXPECT_NEAR(got_book[1 * HEAD_DIM + d], 10.0f + 0.01f * d, 0.2f)
            << "cluster-1 d=" << d;
    }

    // Encode every sample: first N_PER should all land on one index, the
    // remaining on the other.  We don't know which is 0 vs 1 (depends on
    // seed ordering) — just check that the split is clean.
    uint16_t* d_idx = alloc_device<uint16_t>(N_SAMPLES);
    vq_encode(d_data, d_book, d_idx, /*seq_len=*/N_SAMPLES, /*n_kv_heads=*/1,
              HEAD_DIM, CBOOK, stream_);
    std::vector<uint16_t> idx(N_SAMPLES);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(idx.data(), d_idx, N_SAMPLES * sizeof(uint16_t),
                         cudaMemcpyDeviceToHost), cudaSuccess);

    uint16_t a = idx[0];
    uint16_t b = idx[N_SAMPLES - 1];
    EXPECT_NE(a, b) << "both clusters collapsed to one codebook entry";
    for (int i = 0; i < N_PER; ++i)        EXPECT_EQ(idx[i],            a);
    for (int i = N_PER; i < N_SAMPLES; ++i) EXPECT_EQ(idx[i],           b);
}

// ---------------------------------------------------------------------------
// Identical seeds + identical data → after one iteration, every centroid
// collapses to the mean of the samples assigned to it.  With one cluster
// (codebook_size=1), the resulting centroid must equal the data mean.
// ---------------------------------------------------------------------------
TEST_F(CodebookInitTest, KmeansSingleClusterConvergesToMean)
{
    constexpr int HEAD_DIM = 4;
    constexpr int CBOOK = 1;
    constexpr int N_SAMPLES = 16;

    std::vector<float> data(N_SAMPLES * HEAD_DIM);
    std::vector<float> mean(HEAD_DIM, 0.f);
    for (int i = 0; i < N_SAMPLES; ++i) {
        for (int d = 0; d < HEAD_DIM; ++d) {
            float v = 0.1f * i + 0.2f * d;
            data[i * HEAD_DIM + d] = v;
            mean[d] += v;
        }
    }
    for (int d = 0; d < HEAD_DIM; ++d) mean[d] /= N_SAMPLES;

    std::vector<float> book(HEAD_DIM, 0.f);   // arbitrary seed
    __half* d_data = upload_f16(data);
    __half* d_book = upload_f16(book);

    codebook_init_kmeans(d_data, d_book,
                         N_SAMPLES, CBOOK, HEAD_DIM,
                         /*n_iterations=*/3, stream_);
    auto got = download_f16_as_f32(d_book, HEAD_DIM);
    for (int d = 0; d < HEAD_DIM; ++d) {
        EXPECT_NEAR(got[d], mean[d], 0.05f) << "d=" << d;
    }
}

// ---------------------------------------------------------------------------
// Zero iterations → codebook should remain exactly equal to the seed (no
// centroid update ever runs).  Guards against stray writes in the kernel
// launch loop.
// ---------------------------------------------------------------------------
TEST_F(CodebookInitTest, KmeansZeroIterationsLeavesSeedUntouched)
{
    constexpr int HEAD_DIM = 4;
    constexpr int CBOOK = 2;
    constexpr int N_SAMPLES = 8;

    std::vector<float> data(N_SAMPLES * HEAD_DIM, 42.0f);
    std::vector<float> book = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};

    __half* d_data = upload_f16(data);
    __half* d_book = upload_f16(book);

    codebook_init_kmeans(d_data, d_book,
                         N_SAMPLES, CBOOK, HEAD_DIM,
                         /*n_iterations=*/0, stream_);
    auto got = download_f16_as_f32(d_book, CBOOK * HEAD_DIM);
    for (size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], book[i], 1e-2f) << "i=" << i;
    }
}

} // namespace
