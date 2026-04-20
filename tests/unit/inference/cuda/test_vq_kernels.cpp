// ---------------------------------------------------------------------------
// test_vq_kernels.cpp — GPU correctness tests for vq_encode / vq_decode.
//
// Encode-decode round trip: when every KV vector exactly matches a codebook
// entry, encode must pick that entry's index, and decode must reproduce the
// vector bit-for-bit.  A second test injects small perturbations and checks
// the nearest-entry invariant.
//
// Tests skip via GTEST_SKIP() when no CUDA device is available.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "backends/cuda/compression/vq_cache.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstring>
#include <vector>

using truellm::vq_encode;
using truellm::vq_decode;

namespace {

class VqKernelTest : public ::testing::Test {
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

    // Upload host float[] as device __half[].
    __half* upload_f16(const std::vector<float>& host)
    {
        std::vector<__half> tmp(host.size());
        for (size_t i = 0; i < host.size(); ++i) tmp[i] = __float2half(host[i]);
        __half* d_ptr = nullptr;
        size_t bytes = tmp.size() * sizeof(__half);
        EXPECT_EQ(cudaMalloc(&d_ptr, bytes), cudaSuccess);
        EXPECT_EQ(cudaMemcpy(d_ptr, tmp.data(), bytes, cudaMemcpyHostToDevice),
                  cudaSuccess);
        allocs_.push_back(d_ptr);
        return d_ptr;
    }

    template <typename T>
    T* alloc_device(size_t n)
    {
        T* d_ptr = nullptr;
        EXPECT_EQ(cudaMalloc(&d_ptr, n * sizeof(T)), cudaSuccess);
        allocs_.push_back(d_ptr);
        return d_ptr;
    }

    std::vector<uint16_t> download_u16(const uint16_t* d_ptr, size_t n)
    {
        std::vector<uint16_t> host(n);
        EXPECT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
        EXPECT_EQ(cudaMemcpy(host.data(), d_ptr, n * sizeof(uint16_t),
                             cudaMemcpyDeviceToHost), cudaSuccess);
        return host;
    }

    std::vector<float> download_f16_as_f32(const __half* d_ptr, size_t n)
    {
        std::vector<__half> tmp(n);
        EXPECT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
        EXPECT_EQ(cudaMemcpy(tmp.data(), d_ptr, n * sizeof(__half),
                             cudaMemcpyDeviceToHost), cudaSuccess);
        std::vector<float> host(n);
        for (size_t i = 0; i < n; ++i) host[i] = __half2float(tmp[i]);
        return host;
    }

    cudaStream_t       stream_ = nullptr;
    std::vector<void*> allocs_;
};

// ---------------------------------------------------------------------------
// Feed in KV vectors that are literal copies of codebook entries.  Encode
// must pick each entry's own index; decode must return the entry unchanged.
// ---------------------------------------------------------------------------
TEST_F(VqKernelTest, EncodeDecodeExactMatchIsIdentity)
{
    constexpr int kSeq = 4;
    constexpr int kHeads = 2;
    constexpr int kDim = 8;
    constexpr int kBook = 4;

    std::vector<float> book_h(kBook * kDim);
    for (int c = 0; c < kBook; ++c)
        for (int d = 0; d < kDim; ++d)
            book_h[c * kDim + d] = static_cast<float>(c * 10 + d);

    // kv vector at (tok, head) := codebook entry (tok*kHeads + head) % kBook.
    std::vector<float> kv_h(kSeq * kHeads * kDim);
    std::vector<uint16_t> expect_idx(kSeq * kHeads);
    for (int t = 0; t < kSeq; ++t) {
        for (int h = 0; h < kHeads; ++h) {
            int want = (t * kHeads + h) % kBook;
            expect_idx[t * kHeads + h] = static_cast<uint16_t>(want);
            for (int d = 0; d < kDim; ++d)
                kv_h[(t * kHeads + h) * kDim + d] = book_h[want * kDim + d];
        }
    }

    __half*   d_kv   = upload_f16(kv_h);
    __half*   d_book = upload_f16(book_h);
    uint16_t* d_idx  = alloc_device<uint16_t>(kSeq * kHeads);

    vq_encode(d_kv, d_book, d_idx, kSeq, kHeads, kDim, kBook, stream_);
    auto got_idx = download_u16(d_idx, kSeq * kHeads);
    for (size_t i = 0; i < got_idx.size(); ++i) {
        EXPECT_EQ(got_idx[i], expect_idx[i]) << "i=" << i;
    }

    // Now round-trip: decode into a fresh buffer and compare to the input.
    __half* d_out = alloc_device<__half>(kSeq * kHeads * kDim);
    vq_decode(d_out, d_book, d_idx, kSeq, kHeads, kDim, stream_);
    auto out_h = download_f16_as_f32(d_out, kSeq * kHeads * kDim);
    ASSERT_EQ(out_h.size(), kv_h.size());
    for (size_t i = 0; i < out_h.size(); ++i) {
        EXPECT_NEAR(out_h[i], kv_h[i], 1e-2f) << "i=" << i;
    }
}

// ---------------------------------------------------------------------------
// Small perturbation < half-distance between entries must still land on the
// nearest entry.  Book entries spaced 1000 apart, perturbation ±1.
// ---------------------------------------------------------------------------
TEST_F(VqKernelTest, EncodePicksNearestUnderSmallPerturbation)
{
    constexpr int kSeq = 3, kHeads = 1, kDim = 4, kBook = 3;
    std::vector<float> book_h = {
        0.f,    0.f,    0.f,    0.f,
        1000.f, 1000.f, 1000.f, 1000.f,
        2000.f, 2000.f, 2000.f, 2000.f,
    };
    // Each token drifts slightly from one of the three centroids.
    std::vector<float> kv_h = {
        0.5f,    -0.5f,   0.2f,   -0.2f,      // nearest: 0
        1001.f,  999.f,   1000.5f, 999.5f,    // nearest: 1
        1999.f,  2001.f,  1999.5f, 2000.5f,   // nearest: 2
    };
    std::vector<uint16_t> expect = {0, 1, 2};

    __half*   d_kv   = upload_f16(kv_h);
    __half*   d_book = upload_f16(book_h);
    uint16_t* d_idx  = alloc_device<uint16_t>(kSeq * kHeads);

    vq_encode(d_kv, d_book, d_idx, kSeq, kHeads, kDim, kBook, stream_);
    auto got = download_u16(d_idx, kSeq * kHeads);
    for (size_t i = 0; i < got.size(); ++i) {
        EXPECT_EQ(got[i], expect[i]) << "i=" << i;
    }
}

// ---------------------------------------------------------------------------
// Decode only depends on the index stream — feeding a controlled index vector
// should read out the exact codebook rows, regardless of any earlier encode.
// Catches row-indexing bugs in vq_decode_kernel.
// ---------------------------------------------------------------------------
TEST_F(VqKernelTest, DecodeFromSyntheticIndicesReadsExpectedRows)
{
    constexpr int kSeq = 2, kHeads = 2, kDim = 4, kBook = 5;
    std::vector<float> book_h(kBook * kDim);
    for (int c = 0; c < kBook; ++c)
        for (int d = 0; d < kDim; ++d)
            book_h[c * kDim + d] = 100.f * c + d;

    std::vector<uint16_t> idx_h = {4, 0, 2, 3};  // (tok,head) → row

    __half*   d_book = upload_f16(book_h);
    uint16_t* d_idx  = alloc_device<uint16_t>(idx_h.size());
    ASSERT_EQ(cudaMemcpy(d_idx, idx_h.data(), idx_h.size() * sizeof(uint16_t),
                         cudaMemcpyHostToDevice), cudaSuccess);
    __half* d_out = alloc_device<__half>(kSeq * kHeads * kDim);

    vq_decode(d_out, d_book, d_idx, kSeq, kHeads, kDim, stream_);
    auto out = download_f16_as_f32(d_out, kSeq * kHeads * kDim);

    for (int i = 0; i < kSeq * kHeads; ++i) {
        int row = idx_h[i];
        for (int d = 0; d < kDim; ++d) {
            EXPECT_NEAR(out[i * kDim + d], book_h[row * kDim + d], 1e-2f)
                << "i=" << i << " d=" << d;
        }
    }
}

} // namespace
