// ---------------------------------------------------------------------------
// kv_compression_pass.cpp — KvCompressionPass implementation
// ---------------------------------------------------------------------------

#include "kv_compression_pass.h"
#include "vq_cache.cuh"
#include "kivi_cache.cuh"
#include "../cuda_allocator.h"

#include <spdlog/spdlog.h>
#include <stdexcept>
#include <cstring>

#ifdef TRUELLM_HAS_CUDA
#  include <cuda_runtime.h>
#  include <cuda_fp16.h>
#  define CUDA_CHECK(expr)                                                \
     do {                                                                 \
         cudaError_t _e = (expr);                                         \
         if (_e != cudaSuccess)                                           \
             throw std::runtime_error(                                    \
                 std::string("[KvCompressionPass] CUDA: ")                \
                 + cudaGetErrorString(_e));                               \
     } while (0)
#endif

namespace truellm {

KvCompressionPass::KvCompressionPass(const CompressionConfig& cfg,
                                     int n_layers, int n_kv_heads,
                                     int head_dim, int block_size)
    : vq_enabled_  (cfg.kv_vq.enabled)
    , kivi_enabled_(cfg.kv_kivi.enabled)
    , n_layers_    (n_layers)
    , n_kv_heads_  (n_kv_heads)
    , head_dim_    (head_dim)
    , block_size_  (block_size)
    , codebook_size_(cfg.kv_vq.codebook_size)
    , kivi_n_outliers_(cfg.kv_kivi.residual_length)   // residual_length = recent N kept full-precision
    , kivi_group_size_(cfg.kv_kivi.bits == 4 ? 16 : 8) // group_size depends on bit-width
    , vq_cfg_  (cfg.kv_vq)
    , kivi_cfg_(cfg.kv_kivi)
{
    // VQ indices are uint16; clamp so the codebook fits the index type.
    if (codebook_size_ > VQ_MAX_CODEBOOK_SIZE) {
        spdlog::warn("[KvCompressionPass] codebook_size={} exceeds uint16 "
                     "index range; clamping to {}",
                     codebook_size_, VQ_MAX_CODEBOOK_SIZE);
        codebook_size_ = VQ_MAX_CODEBOOK_SIZE;
    }
    if (codebook_size_ <= 0 && vq_enabled_) {
        spdlog::warn("[KvCompressionPass] codebook_size={} invalid; disabling VQ",
                     codebook_size_);
        vq_enabled_ = false;
    }

    // KIVI kernels have fixed-size stack buffers (KIVI_MAX_HEAD_DIM = 128).
    // If the model's head_dim exceeds that, enabling KIVI would silently
    // truncate every vector — refuse rather than corrupt.
    if (kivi_enabled_ &&
        !kivi_params_supported(head_dim_, kivi_n_outliers_, kivi_group_size_)) {
        spdlog::warn("[KvCompressionPass] KiVi params unsupported "
                     "(head_dim={} n_outliers={} group_size={}); disabling",
                     head_dim_, kivi_n_outliers_, kivi_group_size_);
        kivi_enabled_ = false;
    }
}

KvCompressionPass::~KvCompressionPass()
{
#ifdef TRUELLM_HAS_CUDA
    if (d_codebook_k_) { cudaFree(d_codebook_k_); d_codebook_k_ = nullptr; }
    if (d_codebook_v_) { cudaFree(d_codebook_v_); d_codebook_v_ = nullptr; }
    if (d_vq_indices_) { cudaFree(d_vq_indices_); d_vq_indices_ = nullptr; }
    if (d_kivi_buf_)   { cudaFree(d_kivi_buf_);   d_kivi_buf_   = nullptr; }
#endif
}

void KvCompressionPass::init(CudaAllocator* /*allocator*/, cudaStream_t stream)
{
#ifndef TRUELLM_HAS_CUDA
    return;
#else
    if (vq_enabled_) {
        size_t book_bytes = static_cast<size_t>(codebook_size_) * head_dim_ * sizeof(__half);
        CUDA_CHECK(cudaMalloc(&d_codebook_k_, book_bytes));
        CUDA_CHECK(cudaMalloc(&d_codebook_v_, book_bytes));
        // Seed with small random values — real calibration happens on first batch
        CUDA_CHECK(cudaMemsetAsync(d_codebook_k_, 0, book_bytes, stream));
        CUDA_CHECK(cudaMemsetAsync(d_codebook_v_, 0, book_bytes, stream));

        // Index scratch: sized for one full-context prefill worth of vectors
        // [n_kv_heads * max_seq] uint16, shared across layers (one layer at a time)
        int max_seq_scratch = 4096;
        CUDA_CHECK(cudaMalloc(&d_vq_indices_,
            static_cast<size_t>(n_kv_heads_) * max_seq_scratch * sizeof(uint16_t)));

        spdlog::info("[KvCompressionPass] VQ codebook {}×{} F16 allocated "
                     "({}MB each)", codebook_size_, head_dim_,
                     book_bytes / (1024 * 1024));
    }

    if (kivi_enabled_) {
        // Scratch for one layer's compressed output
        // Worst case: full block (block_size * n_kv_heads) vectors
        int n_vecs = block_size_ * n_kv_heads_;
        kivi_buf_bytes_ = static_cast<size_t>(n_vecs)
            * kivi_compressed_bytes(head_dim_, kivi_n_outliers_, kivi_group_size_);
        CUDA_CHECK(cudaMalloc(&d_kivi_buf_, kivi_buf_bytes_));

        spdlog::info("[KvCompressionPass] KiVi scratch {}KB "
                     "(outliers={} group={})",
                     kivi_buf_bytes_ / 1024,
                     kivi_n_outliers_, kivi_group_size_);
    }
    (void)stream;
#endif
}

void KvCompressionPass::run(int layer_idx, int seq_len,
                            void* k_ptr, void* v_ptr,
                            cudaStream_t stream)
{
#ifndef TRUELLM_HAS_CUDA
    (void)layer_idx; (void)seq_len; (void)k_ptr; (void)v_ptr; (void)stream;
    return;
#else
    if (vq_enabled_)   run_vq  (layer_idx, seq_len, k_ptr, v_ptr, stream);
    if (kivi_enabled_) run_kivi(layer_idx, seq_len, k_ptr, v_ptr, stream);
#endif
}

void KvCompressionPass::run_vq(int /*layer_idx*/, int seq_len,
                               void* k_ptr, void* v_ptr,
                               cudaStream_t stream)
{
#ifdef TRUELLM_HAS_CUDA
    if (!d_codebook_k_ || seq_len <= 0) return;

    auto* k = reinterpret_cast<__half*>(k_ptr);
    auto* v = reinterpret_cast<__half*>(v_ptr);

    // Codebook seeding: serialise concurrent callers via codebook_init_mu_
    // with double-check so the fast path (codebook_ready_) stays lock-free.
    // The real sample count is seq_len * n_kv_heads_ (one KV vector per head
    // per token), not seq_len alone — gate on that so small prefill batches
    // on multi-head models still get to initialise the codebook.
    if (!codebook_ready_) {
        int n_samples = seq_len * n_kv_heads_;
        if (n_samples < codebook_size_) return;  // wait for a bigger batch
        std::lock_guard<std::mutex> lock(codebook_init_mu_);
        if (!codebook_ready_) {
            codebook_init_kmeans(k, d_codebook_k_,
                                 n_samples, codebook_size_,
                                 head_dim_, /*n_iterations=*/5, stream);
            codebook_init_kmeans(v, d_codebook_v_,
                                 n_samples, codebook_size_,
                                 head_dim_, /*n_iterations=*/5, stream);
            codebook_ready_ = true;
        }
    }

    int n_vecs = seq_len * n_kv_heads_;
    auto* idx  = reinterpret_cast<uint16_t*>(d_vq_indices_);

    // Encode K
    vq_encode(k, reinterpret_cast<const __half*>(d_codebook_k_),
              idx, seq_len, n_kv_heads_, head_dim_, codebook_size_, stream);
    // Decode K back in-place (replaces raw values with nearest codebook entry)
    vq_decode(k, reinterpret_cast<const __half*>(d_codebook_k_),
              idx, seq_len, n_kv_heads_, head_dim_, stream);

    // Same for V
    vq_encode(v, reinterpret_cast<const __half*>(d_codebook_v_),
              idx, seq_len, n_kv_heads_, head_dim_, codebook_size_, stream);
    vq_decode(v, reinterpret_cast<const __half*>(d_codebook_v_),
              idx, seq_len, n_kv_heads_, head_dim_, stream);
    (void)n_vecs;
#else
    (void)seq_len; (void)k_ptr; (void)v_ptr; (void)stream;
#endif
}

void KvCompressionPass::run_kivi(int /*layer_idx*/, int seq_len,
                                 void* k_ptr, void* v_ptr,
                                 cudaStream_t stream)
{
#ifdef TRUELLM_HAS_CUDA
    if (!d_kivi_buf_ || seq_len <= 0) return;

    int n_vecs = seq_len * n_kv_heads_;
    size_t needed = static_cast<size_t>(n_vecs)
                  * kivi_compressed_bytes(head_dim_, kivi_n_outliers_, kivi_group_size_);
    if (needed > kivi_buf_bytes_) {
        // Scratch too small — skip rather than OOM (conservative policy).
        // Resize at next load_model() call is the right fix.
        spdlog::warn("[KvCompressionPass] KiVi scratch too small "
                     "({} > {}), skipping layer", needed, kivi_buf_bytes_);
        return;
    }

    auto* k    = reinterpret_cast<__half*>(k_ptr);
    auto* v    = reinterpret_cast<__half*>(v_ptr);
    auto* buf  = reinterpret_cast<uint8_t*>(d_kivi_buf_);

    // Compress K → scratch → decompress back in-place
    kivi_compress  (k, buf,  n_vecs, head_dim_, kivi_n_outliers_, kivi_group_size_, stream);
    kivi_decompress(buf, k,  n_vecs, head_dim_, kivi_n_outliers_, kivi_group_size_, stream);

    // Same for V
    kivi_compress  (v, buf,  n_vecs, head_dim_, kivi_n_outliers_, kivi_group_size_, stream);
    kivi_decompress(buf, v,  n_vecs, head_dim_, kivi_n_outliers_, kivi_group_size_, stream);
#else
    (void)seq_len; (void)k_ptr; (void)v_ptr; (void)stream;
#endif
}

} // namespace truellm
