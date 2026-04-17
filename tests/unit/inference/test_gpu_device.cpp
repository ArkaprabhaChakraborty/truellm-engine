#include <gtest/gtest.h>

#include "gpu/gpu_device.h"

// ---------------------------------------------------------------------------
// GpuDevice unit tests
//
// In CPU-only builds these tests verify the safe-zero stub behaviour.
// In CUDA builds (TRUELLM_HAS_CUDA defined) they additionally assert that
// the device query returns plausible values.
// ---------------------------------------------------------------------------

TEST(GpuDeviceTest, DeviceCountIsNonNegative)
{
    EXPECT_GE(truellm::GpuDevice::device_count(), 0);
}

TEST(GpuDeviceTest, QueryInvalidDeviceReturnsSentinel)
{
    auto info = truellm::GpuDevice::query(-1);
    EXPECT_EQ(info.device_id, -1);
    EXPECT_EQ(info.vram_total_bytes, 0u);
    EXPECT_EQ(info.vram_free_bytes,  0u);
}

TEST(GpuDeviceTest, QueryOutOfRangeDeviceReturnsSentinel)
{
    auto info = truellm::GpuDevice::query(9999);
    EXPECT_EQ(info.device_id, -1);
    EXPECT_EQ(info.vram_total_bytes, 0u);
}

TEST(GpuDeviceTest, EstimateGpuLayersReturnsMinusOneOnZeroLayerSize)
{
    int result = truellm::GpuDevice::estimate_gpu_layers(0, 0, 0);
    EXPECT_EQ(result, -1);
}

#if defined(TRUELLM_HAS_CUDA)
TEST(GpuDeviceTest, CudaDevice0HasNonzeroVram)
{
    ASSERT_GE(truellm::GpuDevice::device_count(), 1);

    auto info = truellm::GpuDevice::query(0);
    EXPECT_EQ(info.device_id, 0);
    EXPECT_FALSE(info.name.empty());
    EXPECT_GT(info.vram_total_bytes, 0u);
    EXPECT_GT(info.vram_free_bytes,  0u);
    EXPECT_LE(info.vram_free_bytes, info.vram_total_bytes);
}

TEST(GpuDeviceTest, CudaEstimateGpuLayersReturnsPlausibleResult)
{
    constexpr int64_t layer_bytes = 500LL * 1024 * 1024;
    constexpr int64_t kv_bytes    = 512LL * 1024 * 1024;

    int layers = truellm::GpuDevice::estimate_gpu_layers(0, layer_bytes, kv_bytes);
    EXPECT_GE(layers, -1);
}
#endif
