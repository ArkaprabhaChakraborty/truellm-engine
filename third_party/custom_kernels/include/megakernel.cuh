#pragma once
// ---------------------------------------------------------------------------
// megakernel.cuh — Args struct and declarations for the cooperative transformer
// megakernel.
//
// MegakernelArgs encodes everything the single cudaLaunchCooperativeKernel
// invocation needs:  model weights (read-only), KV cache (read-write), per-
// request activation buffers, GQA topology, and optional compression hooks.
//
// GQA invariant: n_heads >= n_kv_heads, n_heads % n_kv_heads == 0.
// kv_heads_per_group = n_heads / n_kv_heads.
//
// Compression hooks (FastV / PyramidDrop):
//   fastv_start_layer == -1   → FastV disabled for this forward pass.
//   pyramid_drop_layers == 0  → PyramidDrop disabled.
//   attn_logit_mask: [seq_len] bool, set by megakernel, read by softmax stage.
//   attn_weight_out: [n_layers, n_heads, seq_len, seq_len] F32.
//     Written only when save_attn_weights == true (needed by Presis scorer).
//   hidden_state_out: [n_layers+1, seq_len, d_model] F16.
//     Written only when save_hidden_states == true (needed by ToMe).
//
// All pointer fields are device pointers (or nullptr to disable the feature).
// ---------------------------------------------------------------------------

#pragma once
#include <cstdint>
#include <cuda_fp16.h>

struct MegakernelArgs {
    // -----------------------------------------------------------------------
    // Model topology
    // -----------------------------------------------------------------------
    int32_t  n_layers;
    int32_t  n_heads;        // query heads
    int32_t  n_kv_heads;     // key/value heads (GQA: n_kv_heads <= n_heads)
    int32_t  head_dim;       // d_model / n_heads
    int32_t  d_model;        // hidden dimension = n_heads * head_dim
    int32_t  d_ffn;          // feed-forward inner dimension
    int32_t  vocab_size;

    // -----------------------------------------------------------------------
    // Sequence layout
    // -----------------------------------------------------------------------
    int32_t  seq_len;        // number of tokens in this forward pass
    int32_t  kv_seq_len;     // total tokens in KV cache (past + current)
    int32_t  batch_size;     // 1 for single-token decode

    // -----------------------------------------------------------------------
    // Per-layer weight pointers  (device, read-only, __ldg-accessed)
    // Layout: wq[layer] → [n_heads*head_dim, d_model] F16 row-major
    //         wk/wv[layer] → [n_kv_heads*head_dim, d_model] F16 row-major
    //         wo[layer] → [d_model, n_heads*head_dim] F16 row-major
    //         w_gate/w_up/w_down[layer] → FFN weights
    //         rms_att_w/rms_ffn_w[layer] → [d_model] F32 RMS-norm scales
    // -----------------------------------------------------------------------
    const __half* const* wq;          // [n_layers]
    const __half* const* wk;          // [n_layers]
    const __half* const* wv;          // [n_layers]
    const __half* const* wo;          // [n_layers]
    const __half* const* w_gate;      // [n_layers]
    const __half* const* w_up;        // [n_layers]
    const __half* const* w_down;      // [n_layers]
    const float*  const* rms_att_w;   // [n_layers]
    const float*  const* rms_ffn_w;   // [n_layers]
    const __half*        w_embed;     // [vocab_size, d_model]
    const __half*        w_unembed;   // [vocab_size, d_model]
    const float*         rms_final_w; // [d_model]

    // -----------------------------------------------------------------------
    // RoPE caches  [max_seq_len, head_dim/2] F32
    // -----------------------------------------------------------------------
    const float* rope_cos;
    const float* rope_sin;
    int32_t      rope_offset; // position id of first token in seq

    // -----------------------------------------------------------------------
    // KV cache  (read-write device buffer)
    // Layout: k_cache[layer][head][kv_seq_len][head_dim] F16
    //         v_cache[layer][head][kv_seq_len][head_dim] F16
    // Pointers to per-layer arrays.
    // -----------------------------------------------------------------------
    __half* const* k_cache;   // [n_layers]
    __half* const* v_cache;   // [n_layers]
    int32_t        kv_cache_capacity; // allocated kv_seq_len slots

    // -----------------------------------------------------------------------
    // Activation workspace  (device, pre-allocated by CudaEngine)
    // Size: at least 2 * seq_len * d_model * sizeof(__half)
    // -----------------------------------------------------------------------
    __half*  act_buf;         // double-buffer: [2, seq_len, d_model]
    float*   softmax_buf;     // [n_heads, seq_len, kv_seq_len] F32

    // -----------------------------------------------------------------------
    // Input / output token ids
    // -----------------------------------------------------------------------
    const int32_t* input_ids;   // [seq_len]
    int32_t*       output_logit_argmax; // single int, written at end

    // -----------------------------------------------------------------------
    // Output logits  (optional; nullptr → skip)
    // -----------------------------------------------------------------------
    __half*  logits_out;  // [seq_len, vocab_size] F16, nullable

    // -----------------------------------------------------------------------
    // Compression hooks
    // -----------------------------------------------------------------------

    // FastV: attention logit masking
    int32_t fastv_start_layer;    // -1 = disabled
    float   fastv_keep_ratio;     // fraction of tokens to keep
    bool*   attn_logit_mask;      // [seq_len] device bool, written by kernel

    // PyramidDrop: per-layer progressive drop
    int32_t pyramid_drop_layers;  // 0 = disabled
    float   pyramid_drop_lambda;  // keep ratio decay per layer

    // Attention weight capture (for Presis scorer, written by kernel)
    bool    save_attn_weights;
    float*  attn_weight_out;      // [n_layers, n_heads, seq_len, kv_seq_len]

    // Hidden state capture (for ToMe, written by kernel)
    bool    save_hidden_states;
    __half* hidden_state_out;     // [n_layers+1, seq_len, d_model]
};

// ---------------------------------------------------------------------------
// Forward declaration of the cooperative kernel.
// Must be launched via cudaLaunchCooperativeKernel.
// ---------------------------------------------------------------------------
extern "C" __global__
void truellm_transformer_megakernel(MegakernelArgs args);
