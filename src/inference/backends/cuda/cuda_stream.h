#pragma once
// ---------------------------------------------------------------------------
// cuda_stream.h — CUDA stream pool manager
//
// Manages a fixed set of CUDA streams with named roles:
//   Stream 0 (PREFILL) : full-sequence prefill forward pass
//   Stream 1 (DECODE)  : single-token decode forward pass
//   Stream 2 (COPY)    : block table / slot mapping host→device transfers,
//                        eviction swaps, staging buffer uploads
//   Stream 3+ (AUX)    : reserved for future async weight loading
//
// Inter-stream ordering is enforced via cudaEvent_t barriers.
// The caller inserts events after copy operations and waits on them
// before launching compute kernels that depend on the copied data.
// ---------------------------------------------------------------------------

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef TRUELLM_HAS_CUDA
#  include <cuda_runtime.h>
#endif

namespace truellm {

// Named stream slots — used as indices into the stream pool.
enum class StreamRole : int {
    Prefill = 0,
    Decode  = 1,
    Copy    = 2,
};

class CudaStream {
public:
    // Creates `num_streams` CUDA streams on the specified device.
    // Minimum is 3 (Prefill, Decode, Copy).
    explicit CudaStream(int num_streams = 3, int device_id = 0);
    ~CudaStream();

    CudaStream(const CudaStream&)            = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    // -----------------------------------------------------------------------
    // Stream accessors
    // -----------------------------------------------------------------------
#ifdef TRUELLM_HAS_CUDA
    cudaStream_t get(StreamRole role) const;
    cudaStream_t get(int index) const;
#endif

    // -----------------------------------------------------------------------
    // Synchronisation
    // -----------------------------------------------------------------------

    // Block the calling CPU thread until the given stream completes.
    void sync(StreamRole role) const;
    void sync_all() const;

    // -----------------------------------------------------------------------
    // CUDA events for cross-stream dependency tracking
    // -----------------------------------------------------------------------

    // Record an event on `from` stream; it will be waited on `to` stream.
    // Returns an opaque event handle (index into internal event pool).
    int record_event(StreamRole from);

    // Make `to` stream wait for the event identified by `event_handle`.
    void wait_event(StreamRole to, int event_handle);

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------
    int num_streams() const { return static_cast<int>(streams_.size()); }
    int device_id()   const { return device_id_; }

private:
#ifdef TRUELLM_HAS_CUDA
    std::vector<cudaStream_t> streams_;
    std::vector<cudaEvent_t>  events_;    // reusable event pool
    int                       next_event_ = 0;
#else
    std::vector<int> streams_;  // placeholder
#endif
    int device_id_;
};

} // namespace truellm
