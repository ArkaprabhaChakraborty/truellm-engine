#pragma once
// ---------------------------------------------------------------------------
// kv_compression_pass.h — Post-prefill KV cache compression (engine-internal)
//
// Instantiated by CudaEngine when kv_vq.enabled or kv_kivi.enabled.
// Called from run_forward_megakernel() after KV slots are written.
//
// Why engine-internal (not a plugin):
//   kv_compression_kernel.so was the original plugin implementation.  It
//   accessed device KV pointers via host->get_kv_cache_tensor(), adding
//   vtable call overhead on every layer of every forward pass.  As a
//   first-party component with direct CudaAllocator access, those calls
//   are replaced with direct pointer arithmetic.
//
// External plugin authors who genuinely need KV cache access (research,
// custom compression schemes) should use host->get_kv_cache_tensor()
// through the plugin vtable — that API is intentionally preserved.
// ---------------------------------------------------------------------------

#include <truellm/config_schema.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <mutex>

namespace truellm {

// VQ indices are stored as uint16_t on device, so the codebook cannot exceed
// 2^16 entries.  Callers passing larger configs are clamped at init() time.
constexpr int VQ_MAX_CODEBOOK_SIZE = 65535;

class CudaAllocator;

class KvCompressionPass {
public:
    // cfg     : full inference compression config (reads kv_vq + kv_kivi)
    // n_layers, n_kv_heads, head_dim, block_size: from CudaModel at load time
    KvCompressionPass(const CompressionConfig& cfg,
                      int n_layers, int n_kv_heads,
                      int head_dim, int block_size);
    ~KvCompressionPass();

    // Called once after load_model(); allocates device codebook/index/scratch.
    // For VQ: runs k-means seeding if calibration_text is non-empty (cold start
    // uses random centroid seeding from the first batch of KV vectors seen).
    void init(CudaAllocator* allocator, cudaStream_t stream);

    // Called per-layer inside run_forward_megakernel() after each KV write.
    //   layer_idx : transformer layer index (0 … n_layers-1)
    //   seq_len   : number of new KV vectors written (= current prompt length
    //               for prefill, = 1 for decode)
    //   k_ptr     : device pointer to K tensor for this layer
    //   v_ptr     : device pointer to V tensor for this layer
    void run(int layer_idx, int seq_len,
             void* k_ptr, void* v_ptr,
             cudaStream_t stream);

    bool enabled() const { return vq_enabled_ || kivi_enabled_; }

private:
    void run_vq  (int layer_idx, int seq_len,
                  void* k_ptr, void* v_ptr, cudaStream_t stream);
    void run_kivi(int layer_idx, int seq_len,
                  void* k_ptr, void* v_ptr, cudaStream_t stream);

    bool vq_enabled_;
    bool kivi_enabled_;
    int  n_layers_, n_kv_heads_, head_dim_, block_size_;

    // VQ per-layer state
    // d_codebook_k_/v_: [codebook_size, head_dim] F16, one copy shared across layers
    void*    d_codebook_k_  = nullptr;
    void*    d_codebook_v_  = nullptr;
    // d_vq_indices_: [n_layers, max_seq_per_compress, n_kv_heads] uint16
    // Sized at init() for the maximum prefill batch we expect to compress.
    void*    d_vq_indices_  = nullptr;
    int      codebook_size_ = 0;
    bool     codebook_ready_ = false;
    // Serialises the one-shot k-means seeding against concurrent streams.
    // Multiple decode/prefill streams may race into run_vq() before the first
    // batch finishes; without this, they double-write d_codebook_k_/v_.
    std::mutex codebook_init_mu_;

    // KiVi per-layer scratch
    // d_kivi_buf_: single-layer scratch [n_kv_heads * compressed_bytes_per_vec]
    void*    d_kivi_buf_    = nullptr;
    size_t   kivi_buf_bytes_ = 0;
    int      kivi_n_outliers_ = 0;
    int      kivi_group_size_ = 0;

    VqConfig   vq_cfg_;
    KiviConfig kivi_cfg_;
};

// codebook_init_kmeans — K-means codebook seeding on device F16 data.
// Pointers are device __half* passed as void* to avoid pulling cuda_fp16.h
// into MSVC-compiled translation units (cuda_fp16.h is incompatible with
// /Zc:preprocessor when included outside of an NVCC compilation unit).
void codebook_init_kmeans(
    const void*  d_data,     // device __half*
    void*        d_book,     // device __half*
    int n_samples, int codebook_size, int head_dim,
    int n_iterations, cudaStream_t stream);

} // namespace truellm
