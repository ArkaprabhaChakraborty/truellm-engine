// ---------------------------------------------------------------------------
// cuda_stream.cpp — CUDA stream pool implementation
// ---------------------------------------------------------------------------

#include "cuda_stream.h"

#include <spdlog/spdlog.h>
#include <algorithm>
#include <cassert>
#include <stdexcept>

#ifdef TRUELLM_HAS_CUDA
#  define CUDA_CHECK(expr)                                                   \
    do {                                                                     \
        cudaError_t _e = (expr);                                             \
        if (_e != cudaSuccess) {                                             \
            throw std::runtime_error(std::string("[CudaStream] CUDA error: ")\
                + cudaGetErrorString(_e));                                   \
        }                                                                    \
    } while (0)
#endif

namespace truellm {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
CudaStream::CudaStream(int num_streams, int device_id)
    : device_id_(device_id)
{
#ifndef TRUELLM_HAS_CUDA
    throw std::runtime_error("[CudaStream] Built without CUDA support");
#else
    CUDA_CHECK(cudaSetDevice(device_id_));

    int n = std::max(num_streams, 3);  // enforce minimum 3
    streams_.resize(n);
    for (int i = 0; i < n; ++i) {
        // cudaStreamNonBlocking: streams do not synchronise with stream 0
        // (the legacy default stream).  We manage dependencies explicitly.
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams_[i],
                                             cudaStreamNonBlocking));
    }

    // Pre-allocate an event pool (16 slots; wraps around via modulo)
    events_.resize(16);
    for (auto& e : events_) {
        CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    }

    spdlog::info("[CudaStream] Created {} streams on device {}", n, device_id_);
#endif
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------
CudaStream::~CudaStream()
{
#ifdef TRUELLM_HAS_CUDA
    // Drain all streams before destroying
    for (auto s : streams_) {
        cudaStreamSynchronize(s);
        cudaStreamDestroy(s);
    }
    for (auto e : events_) {
        cudaEventDestroy(e);
    }
#endif
}

// ---------------------------------------------------------------------------
// Stream accessors
// ---------------------------------------------------------------------------
#ifdef TRUELLM_HAS_CUDA

cudaStream_t CudaStream::get(StreamRole role) const
{
    return get(static_cast<int>(role));
}

cudaStream_t CudaStream::get(int index) const
{
    assert(index >= 0 && index < static_cast<int>(streams_.size()));
    return streams_[index];
}

#endif // TRUELLM_HAS_CUDA

// ---------------------------------------------------------------------------
// Synchronisation
// ---------------------------------------------------------------------------
void CudaStream::sync(StreamRole role) const
{
#ifdef TRUELLM_HAS_CUDA
    CUDA_CHECK(cudaStreamSynchronize(streams_[static_cast<int>(role)]));
#endif
}

void CudaStream::sync_all() const
{
#ifdef TRUELLM_HAS_CUDA
    for (auto s : streams_) {
        CUDA_CHECK(cudaStreamSynchronize(s));
    }
#endif
}

// ---------------------------------------------------------------------------
// Event management
// ---------------------------------------------------------------------------
int CudaStream::record_event(StreamRole from)
{
#ifndef TRUELLM_HAS_CUDA
    return -1;
#else
    int handle = next_event_ % static_cast<int>(events_.size());
    next_event_ = (next_event_ + 1) % static_cast<int>(events_.size());
    CUDA_CHECK(cudaEventRecord(events_[handle], streams_[static_cast<int>(from)]));
    return handle;
#endif
}

void CudaStream::wait_event(StreamRole to, int event_handle)
{
#ifdef TRUELLM_HAS_CUDA
    if (event_handle < 0 || event_handle >= static_cast<int>(events_.size())) return;
    CUDA_CHECK(cudaStreamWaitEvent(streams_[static_cast<int>(to)],
                                   events_[event_handle], 0));
#else
    (void)to; (void)event_handle;
#endif
}

} // namespace truellm
