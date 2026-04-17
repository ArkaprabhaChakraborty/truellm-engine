// ---------------------------------------------------------------------------
// multi_gpu.cpp — Tensor parallelism coordinator (Phase 6 stub)
// ---------------------------------------------------------------------------

#include "multi_gpu.h"
#include "gpu/gpu_device.h"

#include <spdlog/spdlog.h>
#include <stdexcept>

namespace truellm {

MultiGpu::MultiGpu(int tp_size, int primary_device_id)
    : tp_size_(tp_size), primary_device_id_(primary_device_id)
{
    if (tp_size_ < 1) tp_size_ = 1;

    if (tp_size_ > 1) {
        int dev_count = GpuDevice::device_count();
        if (tp_size_ > dev_count) {
            throw std::invalid_argument(
                "[MultiGpu] tensor_parallel_size=" + std::to_string(tp_size_) +
                " exceeds available device count=" + std::to_string(dev_count));
        }
        // NCCL communicator initialisation is reserved for Phase 6 extension.
        // Setting tp_size > 1 without NCCL compiled in is a configuration error.
        throw std::runtime_error(
            "[MultiGpu] tensor_parallel_size > 1 requires NCCL — "
            "not yet implemented. Set tensor_parallel_size = 1.");
    }

    spdlog::info("[MultiGpu] tp_size={} device={}", tp_size_, primary_device_id_);
}

MultiGpu::~MultiGpu()
{
    // NCCL communicator cleanup reserved for Phase 6.
}

#ifdef TRUELLM_HAS_CUDA
void MultiGpu::all_reduce(void* /*buf*/, size_t /*n_elements*/,
                           DataType /*dtype*/, cudaStream_t /*stream*/)
{
    // No-op for tp_size == 1 — single device needs no reduction.
    // Multi-device: ncclAllReduce(buf, buf, n_elements, ncclFloat16,
    //               ncclSum, nccl_comm_, stream);
}
#endif

TrueBuffer MultiGpu::scatter_weight(const TrueBuffer& weight, int /*split_axis*/)
{
    // No-op for single device — return the full weight tensor unchanged.
    return weight;
}

} // namespace truellm
