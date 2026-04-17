#pragma once
// ---------------------------------------------------------------------------
// multi_gpu.h — Tensor parallelism coordinator stub
//
// When tensor_parallel_size = 1 (default), all operations are no-ops and
// the single-device path in CudaEngine is used without modification.
//
// When tensor_parallel_size > 1 (future), this class manages NCCL
// communicators and provides all-reduce primitives for attention and FFN.
//
// Phase 6 implements the single-device (tp_size=1) path only.
// Multi-device support is reserved for a follow-on phase.
// ---------------------------------------------------------------------------

#include <truellm/types.h>
#include <stdexcept>
#include <string>

#ifdef TRUELLM_HAS_CUDA
#  include <cuda_runtime.h>
#endif

namespace truellm {

class MultiGpu {
public:
    // Constructs the coordinator. Validates tp_size against device count.
    // Throws InvalidArgument if tp_size > device_count().
    explicit MultiGpu(int tp_size, int primary_device_id);
    ~MultiGpu();

    MultiGpu(const MultiGpu&)            = delete;
    MultiGpu& operator=(const MultiGpu&) = delete;

    // -----------------------------------------------------------------------
    // Properties
    // -----------------------------------------------------------------------
    int tp_size()   const { return tp_size_; }
    int local_rank() const { return 0; }   // single-device always rank 0
    bool is_single_device() const { return tp_size_ == 1; }

    // -----------------------------------------------------------------------
    // Collective operations (no-ops when tp_size == 1)
    // -----------------------------------------------------------------------

    // All-reduce across tensor-parallel ranks after attention output projection
    // or FFN down projection.
    // buf : device buffer holding local partial result [tokens, hidden]
    // Returns immediately when tp_size == 1.
#ifdef TRUELLM_HAS_CUDA
    void all_reduce(void* buf, size_t n_elements, DataType dtype,
                    cudaStream_t stream);
#endif

    // Scatter weights across devices during model load (reserved, no-op now).
    // Returns the slice of the weight tensor that belongs to local_rank.
    TrueBuffer scatter_weight(const TrueBuffer& weight, int split_axis);

private:
    int tp_size_;
    int primary_device_id_;

    // NCCL communicator — initialised when tp_size > 1 (reserved).
    void* nccl_comm_ = nullptr;
};

} // namespace truellm
