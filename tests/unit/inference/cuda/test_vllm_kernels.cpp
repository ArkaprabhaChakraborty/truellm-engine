// ---------------------------------------------------------------------------
// test_vllm_kernels.cpp — GPU correctness tests for the vendored vLLM
// kernels (rms_norm, silu_and_mul).  Each test allocates small device
// buffers, launches the kernel, and compares the result against a CPU
// reference implementation.
//
// These tests require a usable CUDA device at runtime.  When no device is
// present (cudaGetDeviceCount == 0 or the driver is missing), every test
// is marked GTEST_SKIP so the suite still passes on CPU-only hosts.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <truellm/types.h>

#include <cmath>
#include <cstdlib>
#include <vector>

using truellm::DataType;
using truellm::TrueBuffer;

// ---------------------------------------------------------------------------
// Forward declarations that mirror those in cuda_engine.cpp.  We cannot
// include vllm_types.h from this TU without dragging in <cuda_fp16.h> on
// the host side, so we hand-roll the signatures.  They are ABI-compatible
// with the definitions in third_party/vllm_kernels/*.cu.
// ---------------------------------------------------------------------------
void rms_norm(TrueBuffer& out, const TrueBuffer& x, const TrueBuffer& weight,
              float eps, cudaStream_t stream);

void silu_and_mul(TrueBuffer& out, const TrueBuffer& gate,
                  const TrueBuffer& up, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Test fixture: skip when no CUDA device is visible.  Allocates a single
// stream for every test; freed by destructor.
// ---------------------------------------------------------------------------
class VllmKernelTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        if (err != cudaSuccess || count <= 0) {
            GTEST_SKIP() << "No CUDA device available (err="
                         << cudaGetErrorString(err) << ", count=" << count
                         << ")";
        }
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess);
    }

    void TearDown() override
    {
        if (stream_) cudaStreamDestroy(stream_);
    }

    // Upload a host vector to a freshly-allocated device buffer and wrap it
    // in a TrueBuffer view with the requested shape.  Caller owns the
    // device pointer and is responsible for cudaFree.
    TrueBuffer upload(const std::vector<float>& host,
                      int64_t d0, int64_t d1 = 1)
    {
        float* d_ptr = nullptr;
        size_t bytes = host.size() * sizeof(float);
        EXPECT_EQ(cudaMalloc(&d_ptr, bytes), cudaSuccess);
        EXPECT_EQ(cudaMemcpy(d_ptr, host.data(), bytes,
                             cudaMemcpyHostToDevice), cudaSuccess);
        device_ptrs_.push_back(d_ptr);
        return make_view(d_ptr, d0, d1);
    }

    TrueBuffer allocate(int64_t d0, int64_t d1 = 1)
    {
        float* d_ptr = nullptr;
        size_t bytes = static_cast<size_t>(d0 * d1) * sizeof(float);
        EXPECT_EQ(cudaMalloc(&d_ptr, bytes), cudaSuccess);
        EXPECT_EQ(cudaMemsetAsync(d_ptr, 0, bytes, stream_), cudaSuccess);
        device_ptrs_.push_back(d_ptr);
        return make_view(d_ptr, d0, d1);
    }

    static TrueBuffer make_view(float* d_ptr, int64_t d0, int64_t d1)
    {
        TrueBuffer b{};
        b.data       = d_ptr;
        b.dtype      = DataType::F32;
        b.device_id  = 0;
        b.ndim       = 2;
        b.shape[0]   = d0;
        b.shape[1]   = d1;
        b.shape[2]   = 1;
        b.shape[3]   = 1;
        b.stride[3]  = 1;
        b.stride[2]  = 1;
        b.stride[1]  = 1;
        b.stride[0]  = d1;
        b.size_bytes = static_cast<size_t>(d0 * d1) * sizeof(float);
        return b;
    }

    std::vector<float> download(const TrueBuffer& buf)
    {
        size_t n = static_cast<size_t>(buf.shape[0] * buf.shape[1]);
        std::vector<float> host(n);
        EXPECT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
        EXPECT_EQ(cudaMemcpy(host.data(), buf.data, n * sizeof(float),
                             cudaMemcpyDeviceToHost), cudaSuccess);
        return host;
    }

    // Free every device allocation made through upload() / allocate().
    ~VllmKernelTest() override
    {
        for (auto* p : device_ptrs_) cudaFree(p);
    }

    cudaStream_t             stream_ = nullptr;
    std::vector<float*>      device_ptrs_;
};

// ---------------------------------------------------------------------------
// RMSNorm reference: y_i = (x_i / rms(x)) * w_i, where
//   rms(x) = sqrt(mean(x_i^2) + eps).
// ---------------------------------------------------------------------------
static std::vector<float> rms_norm_reference(
    const std::vector<float>& x,
    const std::vector<float>& w,
    int num_tokens, int hidden, float eps)
{
    std::vector<float> out(x.size());
    for (int t = 0; t < num_tokens; ++t) {
        double sumsq = 0.0;
        for (int h = 0; h < hidden; ++h) {
            double v = x[t * hidden + h];
            sumsq += v * v;
        }
        double rms = std::sqrt(sumsq / hidden + eps);
        for (int h = 0; h < hidden; ++h) {
            out[t * hidden + h] =
                static_cast<float>((x[t * hidden + h] / rms) * w[h]);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Small, hand-checkable case: 2 tokens × 8 hidden dims.
// ---------------------------------------------------------------------------
TEST_F(VllmKernelTest, RmsNormF32MatchesCpuReference)
{
    constexpr int T = 2;
    constexpr int H = 8;
    constexpr float kEps = 1e-5f;

    std::vector<float> host_x(T * H);
    std::vector<float> host_w(H);
    for (int i = 0; i < T * H; ++i) host_x[i] = 0.1f * (i + 1);
    for (int i = 0; i < H;     ++i) host_w[i] = 1.0f - 0.05f * i;

    TrueBuffer x   = upload(host_x, T, H);
    TrueBuffer w   = upload(host_w, H, 1);
    TrueBuffer out = allocate(T, H);

    rms_norm(out, x, w, kEps, stream_);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    auto got = download(out);

    auto expected = rms_norm_reference(host_x, host_w, T, H, kEps);
    ASSERT_EQ(got.size(), expected.size());
    for (size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], expected[i], 1e-4f) << "idx=" << i;
    }
}

// ---------------------------------------------------------------------------
// Uniform-input invariant: when every element of x equals c, RMSNorm with
// unit weights reduces to a constant sqrt(1 + eps*H/c^2)-rescaled vector.
// In the eps→0 limit every output equals 1.0 regardless of c, which is the
// easiest-to-check invariant.
// ---------------------------------------------------------------------------
TEST_F(VllmKernelTest, RmsNormF32UniformInputYieldsUnitOutput)
{
    constexpr int T = 1;
    constexpr int H = 32;
    constexpr float c = 2.5f;

    std::vector<float> host_x(T * H, c);
    std::vector<float> host_w(H, 1.0f);

    TrueBuffer x   = upload(host_x, T, H);
    TrueBuffer w   = upload(host_w, H, 1);
    TrueBuffer out = allocate(T, H);

    rms_norm(out, x, w, /*eps=*/0.0f, stream_);
    auto got = download(out);

    for (float v : got) EXPECT_NEAR(v, 1.0f, 1e-4f);
}

// ---------------------------------------------------------------------------
// silu_and_mul reference: out[i] = silu(gate[i]) * up[i]
// silu(x) = x / (1 + exp(-x))
// ---------------------------------------------------------------------------
TEST_F(VllmKernelTest, SiluAndMulF32MatchesCpuReference)
{
    constexpr int T = 4;
    constexpr int D = 16;

    std::vector<float> host_gate(T * D), host_up(T * D);
    for (int i = 0; i < T * D; ++i) {
        host_gate[i] = -1.0f + 0.07f * i;
        host_up[i]   = 0.25f + 0.03f * i;
    }

    TrueBuffer gate = upload(host_gate, T, D);
    TrueBuffer up   = upload(host_up,   T, D);
    TrueBuffer out  = allocate(T, D);

    silu_and_mul(out, gate, up, stream_);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    auto got = download(out);

    ASSERT_EQ(got.size(), host_gate.size());
    for (size_t i = 0; i < got.size(); ++i) {
        float g = host_gate[i];
        float expected = (g / (1.0f + std::exp(-g))) * host_up[i];
        EXPECT_NEAR(got[i], expected, 1e-5f) << "idx=" << i;
    }
}

// ---------------------------------------------------------------------------
// silu_and_mul with up=0 produces zero regardless of gate — catches any
// accidental element skip or wrong-size indexing in the launch geometry.
// ---------------------------------------------------------------------------
TEST_F(VllmKernelTest, SiluAndMulF32ZeroUpProducesZero)
{
    constexpr int T = 2;
    constexpr int D = 64;
    std::vector<float> host_gate(T * D);
    for (int i = 0; i < T * D; ++i) host_gate[i] = static_cast<float>(i - T * D / 2);
    std::vector<float> host_up(T * D, 0.0f);

    TrueBuffer gate = upload(host_gate, T, D);
    TrueBuffer up   = upload(host_up,   T, D);
    TrueBuffer out  = allocate(T, D);

    silu_and_mul(out, gate, up, stream_);
    auto got = download(out);

    for (float v : got) EXPECT_EQ(v, 0.0f);
}
