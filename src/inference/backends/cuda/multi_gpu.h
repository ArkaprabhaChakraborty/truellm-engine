#pragma once
// ---------------------------------------------------------------------------
// multi_gpu.h — Tensor-parallelism coordinator
//
// Status (this commit)
//   tp_size == 1                                         — fully supported
//   tp_size  > 1, TRUELLM_HAS_NCCL defined, single-proc  — communicator is
//                                                          initialised; the
//                                                          all_reduce()
//                                                          primitive uses
//                                                          ncclAllReduce
//                                                          across the local
//                                                          ranks
//   tp_size  > 1, TRUELLM_HAS_NCCL defined, multi-proc   — caller supplies
//                                                          a unique-id; the
//                                                          rank/world-size
//                                                          pair is set via
//                                                          ncclCommInitRank
//   tp_size  > 1, TRUELLM_HAS_NCCL undefined             — the constructor
//                                                          throws so the
//                                                          server fails
//                                                          loudly at boot
//
// What is *not* yet wired
//   The CUDA megakernel still treats every weight matrix as if it lived on
//   one device.  A real tensor-parallel forward pass requires:
//     - column-parallel sharding of W_q / W_k / W_v / W_gate / W_up
//     - row-parallel sharding of W_o / W_down
//     - all_reduce after attention output projection and after FFN down
//     - a TP-aware paged KV allocator (one shard per rank)
//   None of those exist in CudaEngine today.  This file holds the framework
//   they will plug into when that work lands; until then, configuring
//   tensor_parallel_size > 1 should be considered experimental.
// ---------------------------------------------------------------------------

#include <truellm/types.h>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef TRUELLM_HAS_CUDA
#  include <cuda_runtime.h>
#endif

#ifdef TRUELLM_HAS_NCCL
#  include <nccl.h>
#endif

namespace truellm {

// Init mode for a multi-rank communicator.  Single-process multi-GPU is the
// common in-server case; multi-process is reserved for distributed inference
// where the caller supplies a unique-id.
enum class TpInitMode {
    SingleProcessAllRanks = 0,   // ncclCommInitAll across [0..tp_size-1]
    MultiProcessRank      = 1    // ncclCommInitRank with externally-supplied
                                  // unique id and rank
};

class MultiGpu {
public:
    // Single-process construction: builds tp_size local communicators
    // covering CUDA devices [0..tp_size-1] when NCCL is present.
    explicit MultiGpu(int tp_size, int primary_device_id);

    // Multi-process construction: caller supplies the unique id (broadcast
    // out-of-band) plus this rank's index in [0, tp_size).  The local CUDA
    // device id can differ from rank when devices are partitioned.
    MultiGpu(int tp_size, int rank, int local_device_id,
             const void* unique_id, size_t unique_id_size);

    ~MultiGpu();

    MultiGpu(const MultiGpu&)            = delete;
    MultiGpu& operator=(const MultiGpu&) = delete;

    // -----------------------------------------------------------------------
    // Properties
    // -----------------------------------------------------------------------
    int  tp_size()         const { return tp_size_; }
    int  rank()            const { return rank_; }
    int  local_rank()      const { return rank_; }
    int  local_device_id() const { return local_device_id_; }
    bool is_single_device() const { return tp_size_ == 1; }
    bool nccl_ready()       const { return nccl_ready_; }

    // -----------------------------------------------------------------------
    // Collective operations
    // -----------------------------------------------------------------------

    // All-reduce across tensor-parallel ranks (sum reduction).
    // tp_size == 1                : returns immediately
    // NCCL ready                  : ncclAllReduce in-place
    // tp_size > 1, NCCL not ready : returns ErrorCode::Unavailable
#ifdef TRUELLM_HAS_CUDA
    ErrorCode all_reduce(void* buf, size_t n_elements, DataType dtype,
                         cudaStream_t stream);
#endif

    // Split a weight tensor along split_axis and return this rank's slice.
    // Returns the input unchanged when tp_size == 1.  When tp_size > 1 but
    // the axis is not divisible by tp_size, throws std::invalid_argument.
    // The returned TrueBuffer is a view — the caller still owns the source
    // device memory.
    TrueBuffer scatter_weight(const TrueBuffer& weight, int split_axis);

    // Generate a fresh ncclUniqueId for distribution to peer ranks.  The
    // returned bytes are what each peer must pass to MultiGpu(... rank ...
    // unique_id ...).  Throws when NCCL is not compiled in.
    std::vector<uint8_t> create_unique_id();

private:
    int tp_size_         = 1;
    int rank_            = 0;
    int local_device_id_ = 0;
    bool nccl_ready_     = false;
    TpInitMode init_mode_ = TpInitMode::SingleProcessAllRanks;

#ifdef TRUELLM_HAS_NCCL
    // For single-process mode we own one comm per local rank; only
    // comms_[rank_] is actually used by this instance, but ncclCommInitAll
    // requires the full array.
    std::vector<ncclComm_t> comms_;
#endif

    void  init_single_process();
    void  init_multi_process(const void* unique_id, size_t unique_id_size);
    void  destroy_comms();
};

} // namespace truellm
