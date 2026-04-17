// ---------------------------------------------------------------------------
// cuda_allocator.cpp — Paged KV block pool implementation
// ---------------------------------------------------------------------------

#include "cuda_allocator.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <numeric>
#include <stdexcept>

#ifdef TRUELLM_HAS_CUDA
#  include <cuda_runtime.h>
#  define CUDA_CHECK(expr)                                                   \
    do {                                                                     \
        cudaError_t _e = (expr);                                             \
        if (_e != cudaSuccess) {                                             \
            throw std::runtime_error(std::string("[CudaAllocator] CUDA error: ") \
                + cudaGetErrorString(_e));                                   \
        }                                                                    \
    } while (0)
#endif

namespace truellm {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
CudaAllocator::CudaAllocator(const CudaEngineConfig& cfg,
                             int num_layers,
                             int num_kv_heads,
                             int head_size,
                             DataType kv_dtype,
                             int device_id)
    : block_size_    (cfg.paged_kv_block_size)
    , max_num_blocks_(cfg.max_num_blocks)
    , num_layers_    (num_layers)
    , num_kv_heads_  (num_kv_heads)
    , head_size_     (head_size)
    , kv_dtype_      (kv_dtype)
    , device_id_     (device_id)
    , fp8_enabled_   (cfg.use_fp8_kv)
{
#ifndef TRUELLM_HAS_CUDA
    throw std::runtime_error("[CudaAllocator] Built without CUDA support");
#else
    CUDA_CHECK(cudaSetDevice(device_id_));

    // Auto-size block pool when max_num_blocks == 0
    if (max_num_blocks_ <= 0) {
        size_t free_bytes = 0, total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));

        float util   = (cfg.gpu_memory_utilization > 0.f)
                     ? cfg.gpu_memory_utilization : 0.90f;
        size_t budget = static_cast<size_t>(free_bytes * util);
        int    bpb    = bytes_per_block();  // already includes num_layers_
        max_num_blocks_ = static_cast<int>(budget / bpb);
        spdlog::info("[CudaAllocator] Auto-sized pool: {} blocks "
                     "({:.0f} MB, {:.0f} MB free, {:.0f}% utilization)",
                     max_num_blocks_,
                     (static_cast<double>(max_num_blocks_) * bpb) / (1 << 20),
                     free_bytes / (1.0 * (1 << 20)),
                     util * 100.f);
    }

    if (max_num_blocks_ <= 0) {
        throw std::runtime_error("[CudaAllocator] No VRAM available for KV pool");
    }

    // Compute max_blocks_per_seq for a plausible maximum context
    // (2048 blocks is more than enough for 32768 tokens with block_size=16)
    max_blocks_per_seq_ = std::min(max_num_blocks_, 2048);

    // ---------------------------------------------------------------------------
    // Allocate KV cache device buffer
    // Layout: [num_blocks, 2, block_size, num_kv_heads, head_size]
    // Stored flattened — one physical block covers ALL layers via
    // per-layer pointer arithmetic in the engine.
    // ---------------------------------------------------------------------------
    size_t pool_bytes = static_cast<size_t>(max_num_blocks_) * bytes_per_block();
    spdlog::info("[CudaAllocator] Allocating {:.0f} MB for {} KV blocks",
                 pool_bytes / (1.0 * (1 << 20)), max_num_blocks_);
    CUDA_CHECK(cudaMalloc(&kv_cache_dev_, pool_bytes));
    CUDA_CHECK(cudaMemset(kv_cache_dev_, 0, pool_bytes));

    // Staging buffers (pinned host ↔ device block table transfers)
    CUDA_CHECK(cudaMalloc(&bt_dev_buf_,
                          max_blocks_per_seq_ * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&sm_dev_buf_,
                          max_blocks_per_seq_ * block_size_ * sizeof(int32_t)));

    // Initialise ref-counts and free list
    block_ref_counts_.assign(max_num_blocks_, 0);
    free_list_.resize(max_num_blocks_);
    std::iota(free_list_.begin(), free_list_.end(), 0);  // 0,1,2,...,N-1
    std::reverse(free_list_.begin(), free_list_.end());   // pop from back = block 0 first

    spdlog::info("[CudaAllocator] Ready: {} blocks, block_size={}, "
                 "kv_heads={}, head_size={}, fp8={}",
                 max_num_blocks_, block_size_, num_kv_heads_, head_size_,
                 fp8_enabled_ ? "on" : "off");
#endif
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------
CudaAllocator::~CudaAllocator()
{
#ifdef TRUELLM_HAS_CUDA
    if (kv_cache_dev_)     cudaFree(kv_cache_dev_);
    if (bt_dev_buf_)       cudaFree(bt_dev_buf_);
    if (sm_dev_buf_)       cudaFree(sm_dev_buf_);
    if (eviction_host_buf_) cudaFreeHost(eviction_host_buf_);
#endif
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

int CudaAllocator::bytes_per_block() const
{
    size_t elem_size = (fp8_enabled_) ? 1 : 2;  // FP8=1B, F16=2B
    // 2 = K and V;  num_layers_ = all transformer layers stored per block
    return static_cast<int>(
        2LL * num_layers_ * block_size_ * num_kv_heads_ * head_size_ * elem_size);
}

int CudaAllocator::alloc_block_locked()
{
    if (free_list_.empty()) return -1;
    int id = free_list_.back();
    free_list_.pop_back();
    block_ref_counts_[id] = 1;
    return id;
}

void CudaAllocator::free_block_locked(int id)
{
    assert(id >= 0 && id < max_num_blocks_);
    if (--block_ref_counts_[id] <= 0) {
        block_ref_counts_[id] = 0;
        free_list_.push_back(id);
    }
}

// ---------------------------------------------------------------------------
// Sequence lifecycle
// ---------------------------------------------------------------------------

bool CudaAllocator::allocate_sequence(uint64_t seq_id, int num_prompt_tokens)
{
    std::lock_guard<std::mutex> lock(mu_);
    if (block_tables_.count(seq_id)) return false;  // already exists

    int num_blocks = (num_prompt_tokens + block_size_ - 1) / block_size_;
    num_blocks = std::max(num_blocks, 1);

    if (static_cast<int>(free_list_.size()) < num_blocks) {
        spdlog::warn("[CudaAllocator] Out of blocks: need {}, have {}",
                     num_blocks, free_list_.size());
        return false;
    }

    std::vector<int32_t> bt(num_blocks);
    for (int i = 0; i < num_blocks; ++i) {
        int id = alloc_block_locked();
        if (id < 0) {
            // Roll back on partial failure
            for (int j = 0; j < i; ++j) free_block_locked(bt[j]);
            return false;
        }
        bt[i] = id;
    }

    block_tables_[seq_id] = std::move(bt);
    return true;
}

int CudaAllocator::extend_sequence(uint64_t seq_id)
{
    std::lock_guard<std::mutex> lock(mu_);
    auto it = block_tables_.find(seq_id);
    if (it == block_tables_.end()) return -1;

    int id = alloc_block_locked();
    if (id < 0) return -1;

    it->second.push_back(id);
    return id;
}

void CudaAllocator::free_sequence(uint64_t seq_id)
{
    std::lock_guard<std::mutex> lock(mu_);
    auto it = block_tables_.find(seq_id);
    if (it == block_tables_.end()) return;

    for (int32_t bid : it->second)
        free_block_locked(bid);

    block_tables_.erase(it);
    evicted_seqs_.erase(seq_id);
}

bool CudaAllocator::copy_sequence(uint64_t src_id, uint64_t dst_id)
{
    std::lock_guard<std::mutex> lock(mu_);
    auto it = block_tables_.find(src_id);
    if (it == block_tables_.end()) return false;

    // Shallow copy — increment ref counts
    block_tables_[dst_id] = it->second;
    for (int32_t bid : it->second)
        ++block_ref_counts_[bid];

    return true;
}

// ---------------------------------------------------------------------------
// Device pointer accessors
// ---------------------------------------------------------------------------

#ifdef TRUELLM_HAS_CUDA

const int32_t* CudaAllocator::get_device_block_table(uint64_t seq_id,
                                                      cudaStream_t copy_stream)
{
    std::lock_guard<std::mutex> lock(mu_);
    auto it = block_tables_.find(seq_id);
    if (it == block_tables_.end()) return nullptr;

    const auto& bt = it->second;
    int n = static_cast<int>(bt.size());
    assert(n <= max_blocks_per_seq_);

    // Copy block table to device staging buffer (async on copy_stream)
    cudaMemcpyAsync(bt_dev_buf_, bt.data(),
                    n * sizeof(int32_t),
                    cudaMemcpyHostToDevice, copy_stream);
    // Pad remainder to -1 so kernels skip empty slots
    if (n < max_blocks_per_seq_) {
        cudaMemsetAsync(bt_dev_buf_ + n, 0xFF,
                        (max_blocks_per_seq_ - n) * sizeof(int32_t),
                        copy_stream);
    }
    return bt_dev_buf_;
}

const int32_t* CudaAllocator::get_device_slot_mapping(uint64_t seq_id,
                                                       int num_tokens,
                                                       int seq_start_pos,
                                                       cudaStream_t copy_stream)
{
    std::lock_guard<std::mutex> lock(mu_);
    auto it = block_tables_.find(seq_id);
    if (it == block_tables_.end()) return nullptr;

    const auto& bt = it->second;

    // Build host-side slot mapping and copy to device
    std::vector<int32_t> slots(num_tokens);
    for (int i = 0; i < num_tokens; ++i) {
        int pos      = seq_start_pos + i;
        int blk_idx  = pos / block_size_;
        int blk_off  = pos % block_size_;
        assert(blk_idx < static_cast<int>(bt.size()));
        slots[i] = bt[blk_idx] * block_size_ + blk_off;
    }

    cudaMemcpyAsync(sm_dev_buf_, slots.data(),
                    num_tokens * sizeof(int32_t),
                    cudaMemcpyHostToDevice, copy_stream);
    return sm_dev_buf_;
}

#else // !TRUELLM_HAS_CUDA

const int32_t* CudaAllocator::get_device_block_table(uint64_t, cudaStream_t) { return nullptr; }
const int32_t* CudaAllocator::get_device_slot_mapping(uint64_t, int, int, cudaStream_t) { return nullptr; }

#endif

// ---------------------------------------------------------------------------
// Capacity queries
// ---------------------------------------------------------------------------

int CudaAllocator::num_free_blocks() const
{
    std::lock_guard<std::mutex> lock(mu_);
    return static_cast<int>(free_list_.size());
}

KvCacheStats CudaAllocator::stats() const
{
    std::lock_guard<std::mutex> lock(mu_);
    int used = max_num_blocks_ - static_cast<int>(free_list_.size());
    KvCacheStats s;
    s.block_size_tokens = block_size_;
    s.blocks_total      = max_num_blocks_;
    s.blocks_used       = used;
    s.blocks_free       = static_cast<int>(free_list_.size());
    s.utilization_pct   = (max_num_blocks_ > 0)
                         ? (100.f * used / max_num_blocks_) : 0.f;
    s.fp8_enabled       = fp8_enabled_;
    return s;
}

// ---------------------------------------------------------------------------
// Eviction
// ---------------------------------------------------------------------------

bool CudaAllocator::evict_sequence(uint64_t seq_id, cudaStream_t copy_stream)
{
#ifndef TRUELLM_HAS_CUDA
    return false;
#else
    std::lock_guard<std::mutex> lock(mu_);
    auto it = block_tables_.find(seq_id);
    if (it == block_tables_.end()) return false;
    if (evicted_seqs_.count(seq_id)) return true;  // already evicted

    const auto& bt = it->second;
    int n = static_cast<int>(bt.size());

    // Lazy-allocate host pinned buffer if needed
    if (eviction_buf_blocks_ < n) {
        if (eviction_host_buf_) cudaFreeHost(eviction_host_buf_);
        int new_size = n * 2;  // over-provision
        cudaMallocHost(&eviction_host_buf_,
                       static_cast<size_t>(new_size) * bytes_per_block());
        eviction_buf_blocks_ = new_size;
    }

    // Copy each block from device to host
    int bpb = bytes_per_block();
    for (int i = 0; i < n; ++i) {
        int32_t phys = bt[i];
        char* src  = static_cast<char*>(kv_cache_dev_) + phys * bpb;
        char* dst  = static_cast<char*>(eviction_host_buf_) + i * bpb;
        cudaMemcpyAsync(dst, src, bpb, cudaMemcpyDeviceToHost, copy_stream);
        free_block_locked(phys);
    }

    evicted_seqs_[seq_id] = n;   // store actual block count for reload
    it->second.clear();          // block table cleared; reload will refill it
    return true;
#endif
}

bool CudaAllocator::reload_sequence(uint64_t seq_id, cudaStream_t copy_stream)
{
#ifndef TRUELLM_HAS_CUDA
    return false;
#else
    std::lock_guard<std::mutex> lock(mu_);
    if (!evicted_seqs_.count(seq_id)) return false;

    auto& bt = block_tables_[seq_id];
    int n = evicted_seqs_[seq_id];  // actual count stored by evict_sequence

    int bpb = bytes_per_block();
    for (int i = 0; i < n; ++i) {
        int id = alloc_block_locked();
        if (id < 0) return false;
        bt.push_back(id);
        char* src = static_cast<char*>(eviction_host_buf_) + i * bpb;
        char* dst = static_cast<char*>(kv_cache_dev_)     + id * bpb;
        cudaMemcpyAsync(dst, src, bpb, cudaMemcpyHostToDevice, copy_stream);
    }

    evicted_seqs_.erase(seq_id);
    return true;
#endif
}

} // namespace truellm
