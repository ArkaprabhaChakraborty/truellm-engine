#pragma once
// ---------------------------------------------------------------------------
// cuda_allocator.h — Paged KV block pool allocator
//
// Owns the single large VRAM region used by all active sequences' KV caches.
// Sequences interact with the allocator exclusively through integer block IDs.
// Raw device pointers are only materialised when launching CUDA kernels.
//
// Block layout in device memory (per layer, all layers contiguous):
//   [num_layers, 2, num_blocks, num_kv_heads, head_size, block_size]
// Layer l K-pool: base + l * 2 * num_blocks * n_kv * head_size * block_size * elem
// Layer l V-pool: base + (l*2+1) * num_blocks * n_kv * head_size * block_size * elem
//
// Thread safety: all public methods are protected by an internal mutex.
// The scheduler serialises sequence lifecycle transitions so contention
// is low; the mutex is defensive rather than performance-critical.
// ---------------------------------------------------------------------------

#include <truellm/types.h>
#include <truellm/config_schema.h>
#include "../../engine_interface.h"   // KvCacheStats, GpuStats

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef TRUELLM_HAS_CUDA
#  include <cuda_runtime.h>
#endif

namespace truellm {

// ---------------------------------------------------------------------------
// CudaAllocator
// ---------------------------------------------------------------------------
class CudaAllocator {
public:
    CudaAllocator(const CudaEngineConfig& cfg,
                  int num_layers,
                  int num_kv_heads,
                  int head_size,
                  DataType kv_dtype,
                  int device_id);
    ~CudaAllocator();

    // Non-copyable / non-movable (owns raw CUDA allocations).
    CudaAllocator(const CudaAllocator&)            = delete;
    CudaAllocator& operator=(const CudaAllocator&) = delete;

    // -----------------------------------------------------------------------
    // Block-level operations
    // -----------------------------------------------------------------------

    // Allocate blocks for a new sequence.  Returns false when the pool is full.
    // Pre-allocates ceil(num_prompt_tokens / block_size) blocks.
    bool allocate_sequence(uint64_t seq_id, int num_prompt_tokens);

    // Allocate one additional block when a sequence grows past a block boundary.
    // Returns the new physical block ID, or -1 on out-of-memory.
    int extend_sequence(uint64_t seq_id);

    // Release all blocks owned by a sequence back to the free pool.
    void free_sequence(uint64_t seq_id);

    // Shallow-copy a sequence's block table (beam search / prefix sharing).
    // Blocks are reference-counted; both sequences share the same physicals.
    bool copy_sequence(uint64_t src_id, uint64_t dst_id);

    // -----------------------------------------------------------------------
    // Device pointer accessors (called from CudaEngine kernel launch sites)
    // -----------------------------------------------------------------------

    // Copies the host-side block table for seq_id to the GPU staging buffer
    // and returns the device pointer.  Called once per decode step.
    // Stream on which the copy is issued must be synced before kernel launch.
    const int32_t* get_device_block_table(uint64_t seq_id,
                                          cudaStream_t copy_stream);

    // Return a slot_mapping buffer (one int32 per token) for a prefill batch.
    // slot = physical_block_id * block_size + offset_within_block.
    // Caller provides the token-count; method fills and uploads.
    const int32_t* get_device_slot_mapping(uint64_t seq_id,
                                           int num_tokens,
                                           int seq_start_pos,
                                           cudaStream_t copy_stream);

    // -----------------------------------------------------------------------
    // Per-layer KV cache device pointers
    //
    // Full pool layout: [num_layers, 2, num_blocks, n_kv, head_size, block_size]
    //   Layer l K-pool: base + l * 2 * kv_k_pool_bytes()
    //   Layer l V-pool: base + l * 2 * kv_k_pool_bytes() + kv_k_pool_bytes()
    //
    // kv_layer_kv_ptr(l) — combined K+V pointer for reshape_and_cache
    //                      (shape [num_blocks*2, n_kv, head_size, block_size])
    // kv_layer_k_ptr(l)  — K-pool pointer for paged_attention_v1/v2
    // kv_layer_v_ptr(l)  — V-pool pointer for paged_attention_v1/v2
    // -----------------------------------------------------------------------
    void* kv_layer_kv_ptr(int layer) const {
        return static_cast<char*>(kv_cache_dev_)
               + static_cast<size_t>(layer) * 2 * kv_k_pool_bytes();
    }
    void* kv_layer_k_ptr(int layer) const {
        return kv_layer_kv_ptr(layer);
    }
    void* kv_layer_v_ptr(int layer) const {
        return static_cast<char*>(kv_layer_kv_ptr(layer)) + kv_k_pool_bytes();
    }

    // -----------------------------------------------------------------------
    // Capacity queries
    // -----------------------------------------------------------------------
    int  num_free_blocks()  const;
    int  num_total_blocks() const { return max_num_blocks_; }
    int  block_size()       const { return block_size_; }
    int  max_blocks_per_seq() const { return max_blocks_per_seq_; }

    KvCacheStats stats() const;

    // -----------------------------------------------------------------------
    // Eviction (host-pinned swap)
    // -----------------------------------------------------------------------

    // Move all blocks of seq_id to the pinned host buffer.
    // The sequence's block table entry is marked as evicted.
    bool evict_sequence(uint64_t seq_id, cudaStream_t copy_stream);

    // Reload a previously evicted sequence back to GPU.
    bool reload_sequence(uint64_t seq_id, cudaStream_t copy_stream);

private:
    // Config
    int block_size_;
    int max_num_blocks_;
    int max_blocks_per_seq_;
    int num_layers_;
    int num_kv_heads_;
    int head_size_;
    DataType kv_dtype_;
    int device_id_;
    bool fp8_enabled_;

    // VRAM pool — one contiguous allocation
    void* kv_cache_dev_ = nullptr;       // [num_blocks, 2, block_size, num_kv_heads, head_size]

    // Host-pinned eviction buffer (allocated lazily on first evict call)
    void* eviction_host_buf_  = nullptr;
    int   eviction_buf_blocks_ = 0;

    // GPU staging buffers for block tables and slot mappings
    // Sized to max_blocks_per_seq_ and max context tokens respectively.
    int32_t* bt_dev_buf_   = nullptr;  // [max_blocks_per_seq]   per-decode step
    int32_t* sm_dev_buf_   = nullptr;  // [max_context_tokens]   per-prefill

    // Host-side block tables: seq_id → [logical_block_idx → physical_block_id]
    std::unordered_map<uint64_t, std::vector<int32_t>> block_tables_;

    // Ref-counting for beam search block sharing
    std::vector<int> block_ref_counts_;  // [max_num_blocks]

    // Free block stack
    std::vector<int32_t> free_list_;

    // Evicted sequences: seq_id → host-buffer offset (in blocks)
    std::unordered_map<uint64_t, int> evicted_seqs_;

    mutable std::mutex mu_;

    // Internal helpers
    int    alloc_block_locked();     // must hold mu_
    void   free_block_locked(int id);
    int    bytes_per_block() const;

    // Bytes for one layer's K-pool (= same for V-pool):
    //   num_blocks * n_kv * head_size * block_size * elem_size
    size_t kv_k_pool_bytes() const {
        return static_cast<size_t>(max_num_blocks_) * num_kv_heads_
               * head_size_ * block_size_ * (fp8_enabled_ ? 1 : 2);
    }
};

} // namespace truellm
