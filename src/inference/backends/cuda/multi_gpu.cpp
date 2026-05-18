// ---------------------------------------------------------------------------
// multi_gpu.cpp — Tensor parallelism coordinator
//
// Status / scope: see multi_gpu.h header banner.  This file implements the
// constructor pair (single-process and multi-process), NCCL communicator
// init/destroy, ncclAllReduce, and the simple weight-scatter view helper.
// The kernel-level integration (sharded GEMMs, all-reduce sites in the
// megakernel) is not in this commit — multi_gpu_ in CudaEngine still holds
// a tp_size==1 instance for production traffic.
// ---------------------------------------------------------------------------

#include "multi_gpu.h"
#include "gpu/gpu_device.h"

#include <spdlog/spdlog.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace truellm {

namespace {

#ifdef TRUELLM_HAS_NCCL
// Map our DataType to ncclDataType_t.  Returns ncclNumTypes when no
// equivalent NCCL type exists; the caller treats that as InvalidArgument.
ncclDataType_t to_nccl_dtype(DataType dt)
{
    switch (dt) {
        case DataType::F32:  return ncclFloat32;
        case DataType::F16:  return ncclFloat16;
        case DataType::BF16: return ncclBfloat16;
        case DataType::I32:  return ncclInt32;
        case DataType::I8:   return ncclInt8;
        default:             return ncclNumTypes;
    }
}

size_t nccl_elem_size(ncclDataType_t dt)
{
    switch (dt) {
        case ncclFloat32:   return 4;
        case ncclFloat16:   return 2;
        case ncclBfloat16:  return 2;
        case ncclInt32:     return 4;
        case ncclInt8:      return 1;
        default:            return 0;
    }
}
#endif // TRUELLM_HAS_NCCL

} // anonymous namespace

// ---------------------------------------------------------------------------
// Single-process constructor — owns one comm per local rank in [0..tp_size)
// ---------------------------------------------------------------------------
MultiGpu::MultiGpu(int tp_size, int primary_device_id)
    : tp_size_(tp_size)
    , rank_(0)
    , local_device_id_(primary_device_id)
    , init_mode_(TpInitMode::SingleProcessAllRanks)
{
    if (tp_size_ < 1) tp_size_ = 1;

    if (tp_size_ == 1) {
        spdlog::info("[MultiGpu] tp_size=1 device={} (single-device path)",
                     local_device_id_);
        return;
    }

#ifdef TRUELLM_HAS_NCCL
    int dev_count = GpuDevice::device_count();
    if (tp_size_ > dev_count) {
        throw std::invalid_argument(
            "[MultiGpu] tensor_parallel_size=" + std::to_string(tp_size_) +
            " exceeds available device count=" + std::to_string(dev_count));
    }
    init_single_process();
    spdlog::info("[MultiGpu] tp_size={} (single-process NCCL ready)",
                 tp_size_);
#else
    throw std::runtime_error(
        "[MultiGpu] tensor_parallel_size=" + std::to_string(tp_size_) +
        " requires the engine to be built with TRUELLM_HAS_NCCL.  "
        "Either set tensor_parallel_size=1 or rebuild with NCCL support "
        "(install libnccl-dev / nccl_<ver>-cuda<ver> and re-run cmake).");
#endif
}

// ---------------------------------------------------------------------------
// Multi-process constructor — caller distributed the unique-id out-of-band.
// ---------------------------------------------------------------------------
MultiGpu::MultiGpu(int tp_size, int rank, int local_device_id,
                    const void* unique_id, size_t unique_id_size)
    : tp_size_(tp_size)
    , rank_(rank)
    , local_device_id_(local_device_id)
    , init_mode_(TpInitMode::MultiProcessRank)
{
    if (tp_size_ < 1) tp_size_ = 1;
    if (tp_size_ == 1) {
        spdlog::info("[MultiGpu] tp_size=1 (multi-process constructor "
                     "called but world is single-rank)");
        return;
    }
    if (rank_ < 0 || rank_ >= tp_size_) {
        throw std::invalid_argument(
            "[MultiGpu] rank=" + std::to_string(rank_) +
            " is outside [0, tp_size=" + std::to_string(tp_size_) + ")");
    }

#ifdef TRUELLM_HAS_NCCL
    if (!unique_id || unique_id_size != sizeof(ncclUniqueId)) {
        throw std::invalid_argument(
            "[MultiGpu] multi-process init: unique_id must be a "
            "sizeof(ncclUniqueId)-byte blob produced by create_unique_id()");
    }
    init_multi_process(unique_id, unique_id_size);
    spdlog::info("[MultiGpu] tp_size={} rank={} device={} "
                 "(multi-process NCCL ready)",
                 tp_size_, rank_, local_device_id_);
#else
    (void)unique_id; (void)unique_id_size;
    throw std::runtime_error(
        "[MultiGpu] multi-process tensor parallelism requires "
        "TRUELLM_HAS_NCCL.  Set tensor_parallel_size=1 or rebuild "
        "with NCCL support.");
#endif
}

MultiGpu::~MultiGpu()
{
    destroy_comms();
}

// ---------------------------------------------------------------------------
// init_single_process — populate comms_[0..tp_size-1] via ncclCommInitAll
// ---------------------------------------------------------------------------
void MultiGpu::init_single_process()
{
#ifdef TRUELLM_HAS_NCCL
    comms_.assign(tp_size_, nullptr);
    std::vector<int> devices(tp_size_);
    for (int i = 0; i < tp_size_; ++i) devices[i] = i;
    ncclResult_t r = ncclCommInitAll(comms_.data(), tp_size_, devices.data());
    if (r != ncclSuccess) {
        throw std::runtime_error(
            std::string("[MultiGpu] ncclCommInitAll failed: ") +
            ncclGetErrorString(r));
    }
    nccl_ready_ = true;
#endif
}

// ---------------------------------------------------------------------------
// init_multi_process — single comm per process via ncclCommInitRank
// ---------------------------------------------------------------------------
void MultiGpu::init_multi_process(const void* unique_id,
                                   size_t /*unique_id_size*/)
{
#ifdef TRUELLM_HAS_NCCL
    comms_.assign(1, nullptr);
    ncclUniqueId id;
    std::memcpy(&id, unique_id, sizeof(ncclUniqueId));

    if (cudaSetDevice(local_device_id_) != cudaSuccess) {
        throw std::runtime_error(
            "[MultiGpu] cudaSetDevice(" + std::to_string(local_device_id_) +
            ") failed before NCCL init");
    }
    ncclResult_t r = ncclCommInitRank(&comms_[0], tp_size_, id, rank_);
    if (r != ncclSuccess) {
        throw std::runtime_error(
            std::string("[MultiGpu] ncclCommInitRank failed: ") +
            ncclGetErrorString(r));
    }
    nccl_ready_ = true;
#else
    (void)unique_id;
#endif
}

void MultiGpu::destroy_comms()
{
#ifdef TRUELLM_HAS_NCCL
    for (auto* c : comms_) {
        if (c) ncclCommDestroy(c);
    }
    comms_.clear();
    nccl_ready_ = false;
#endif
}

// ---------------------------------------------------------------------------
// all_reduce — sum-reduce a device buffer across the TP world
// ---------------------------------------------------------------------------
#ifdef TRUELLM_HAS_CUDA
ErrorCode MultiGpu::all_reduce(void* buf, size_t n_elements, DataType dtype,
                                cudaStream_t stream)
{
    if (tp_size_ <= 1) return ErrorCode::Ok;     // single-rank no-op
    if (!nccl_ready_)  return ErrorCode::Unavailable;
    if (!buf || n_elements == 0) return ErrorCode::InvalidArgument;

#ifdef TRUELLM_HAS_NCCL
    ncclDataType_t nt = to_nccl_dtype(dtype);
    if (nt == ncclNumTypes) return ErrorCode::InvalidArgument;

    ncclComm_t comm = nullptr;
    if (init_mode_ == TpInitMode::SingleProcessAllRanks) {
        if (rank_ < 0 || static_cast<size_t>(rank_) >= comms_.size())
            return ErrorCode::InternalError;
        comm = comms_[rank_];
    } else {
        if (comms_.empty()) return ErrorCode::InternalError;
        comm = comms_.front();
    }

    ncclResult_t r = ncclAllReduce(buf, buf, n_elements, nt,
                                    ncclSum, comm, stream);
    if (r != ncclSuccess) {
        spdlog::error("[MultiGpu] ncclAllReduce failed: {}",
                      ncclGetErrorString(r));
        return ErrorCode::BackendError;
    }
    (void)nccl_elem_size;  // referenced only for documentation
    return ErrorCode::Ok;
#else
    (void)buf; (void)n_elements; (void)dtype; (void)stream;
    return ErrorCode::Unavailable;
#endif
}
#endif

// ---------------------------------------------------------------------------
// scatter_weight — return this rank's slice of a weight tensor
// ---------------------------------------------------------------------------
TrueBuffer MultiGpu::scatter_weight(const TrueBuffer& weight, int split_axis)
{
    if (tp_size_ == 1) return weight;
    if (split_axis < 0 || split_axis >= weight.ndim) {
        throw std::invalid_argument(
            "[MultiGpu] scatter_weight: split_axis=" + std::to_string(split_axis) +
            " out of range for ndim=" + std::to_string(weight.ndim));
    }

    int64_t axis_len = weight.shape[split_axis];
    if (axis_len % tp_size_ != 0) {
        throw std::invalid_argument(
            "[MultiGpu] scatter_weight: axis length=" + std::to_string(axis_len) +
            " is not divisible by tp_size=" + std::to_string(tp_size_));
    }
    int64_t shard_len = axis_len / tp_size_;
    int64_t offset    = shard_len * rank_;

    TrueBuffer view = weight;
    view.shape[split_axis] = shard_len;

    // Compute byte offset using the original stride; this assumes contiguous
    // layout along split_axis (the common case for our F16 GEMM weights).
    int64_t stride_elems = view.stride[split_axis];
    if (stride_elems == 0) stride_elems = 1;

    size_t elem_size = 0;
    switch (weight.dtype) {
        case DataType::F32:  elem_size = 4; break;
        case DataType::F16:  elem_size = 2; break;
        case DataType::BF16: elem_size = 2; break;
        case DataType::I32:  elem_size = 4; break;
        case DataType::I16:  elem_size = 2; break;
        case DataType::I8:   elem_size = 1; break;
        default:             elem_size = 0; break;
    }
    if (elem_size == 0) {
        throw std::invalid_argument(
            "[MultiGpu] scatter_weight: cannot stride a quantised dtype");
    }
    int64_t byte_offset = offset * stride_elems * static_cast<int64_t>(elem_size);
    view.data = static_cast<char*>(weight.data) + byte_offset;
    view.size_bytes = static_cast<size_t>(shard_len)
                    * stride_elems * elem_size;

    return view;
}

// ---------------------------------------------------------------------------
// create_unique_id — caller distributes this blob to all peer ranks
// ---------------------------------------------------------------------------
std::vector<uint8_t> MultiGpu::create_unique_id()
{
#ifdef TRUELLM_HAS_NCCL
    ncclUniqueId id;
    ncclResult_t r = ncclGetUniqueId(&id);
    if (r != ncclSuccess) {
        throw std::runtime_error(
            std::string("[MultiGpu] ncclGetUniqueId failed: ") +
            ncclGetErrorString(r));
    }
    std::vector<uint8_t> out(sizeof(ncclUniqueId));
    std::memcpy(out.data(), &id, sizeof(ncclUniqueId));
    return out;
#else
    throw std::runtime_error(
        "[MultiGpu] create_unique_id requires the engine to be built "
        "with TRUELLM_HAS_NCCL.");
#endif
}

} // namespace truellm
