// ---------------------------------------------------------------------------
// host_api.h — Vtable the engine passes to truellm_plugin_init.
// Valid for the lifetime of the engine process.  Thread-safe: all functions
// may be called from any thread.
//
// API TIERS
// ─────────
// api_minor == 0  (base API)
//   Logging, config access, model metadata, per-request accessors,
//   status reporting, engine-managed heap.
//
// api_minor >= 1  (advanced API — check host->api_minor >= 1 before calling)
//   Tokenizer, engine config access, device tensor accessors, sub-calls,
//   token injection, context metadata, capability queries.
//
// ADVANCED DEVICE TENSOR APIS (api_minor >= 1)
// ─────────────────────────────────────────────
// get_attention_weights, get_hidden_states, get_kv_cache_tensor
//
//   These return device pointers into engine-owned GPU memory.  They are
//   intentionally preserved as a first-class extensibility surface so that
//   external plugin authors can build advanced plugins:
//
//     • Interpretability / probing tools (read attention patterns per layer)
//     • Custom attention visualisers
//     • Research compression schemes (custom VQ, learned quantisation)
//     • KV cache analysis and profiling plugins
//     • Activation steering plugins that read / write hidden states
//
//   NOTE: The engine's own Presis, ToMe, VQ, and KiVi implementations do NOT
//   use these vtable entries — they are compiled directly into CudaEngine
//   (see compression/) with direct device pointer access and zero overhead.
//   These vtable entries exist for EXTERNAL plugin authors, not for engine-
//   internal code.
//
//   Usage contract:
//     1. Call get_server_capability(ctx, TRUELLM_CAP_ATTN_WEIGHTS) first.
//        Returns "1" only when megakernel_save_attn=true and a CUDA backend
//        is active.  Returns NULL otherwise — do not call get_attention_weights.
//     2. Device pointers are valid only until the next generate() call.
//        Copy to host or use within the same plugin callback invocation.
//     3. Pointers are device-resident. You must have a CUDA context active on
//        the engine's device (query get_server_capability(ctx, "cuda.device_id"))
//        before issuing any CUDA operations on the returned pointer.
//     4. Do NOT free, cudaFree, or otherwise deallocate these pointers.
//        They are owned by the engine's buffer pool.
//
//   See truellm-native-plugins/plugins/examples/attn_probe/ for a reference
//   implementation demonstrating correct usage.
// ---------------------------------------------------------------------------
#pragma once
#ifndef TRUELLM_HOST_API_H
#define TRUELLM_HOST_API_H

#include "plugin.h"
#include "version.h"

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Server capability key constants (used with get_server_capability)
// ---------------------------------------------------------------------------
#define TRUELLM_CAP_MEGAKERNEL       "megakernel.enabled"
#define TRUELLM_CAP_MEGAKERNEL_TILE  "megakernel.tile_size"
#define TRUELLM_CAP_RLM              "rlm.enabled"
#define TRUELLM_CAP_RLM_MAX_DEPTH    "rlm.max_hierarchy_depth"
#define TRUELLM_CAP_ATTN_WEIGHTS     "compression.attention_weights_available"
#define TRUELLM_CAP_HIDDEN_STATES    "compression.hidden_states_available"
#define TRUELLM_CAP_COMPRESSION      "compression.enabled"

struct truellm_host_api_t {
    uint32_t api_version;   // TRUELLM_ABI_VERSION
    uint32_t api_minor;     // TRUELLM_ABI_MINOR — plugins check this before calling minor-1 entries

    // ── Logging ──────────────────────────────────────────────────────────────
    void (*log_debug)(const char* plugin_id, const char* msg);
    void (*log_info) (const char* plugin_id, const char* msg);
    void (*log_warn) (const char* plugin_id, const char* msg);
    void (*log_error)(const char* plugin_id, const char* msg);

    // ── Config access ────────────────────────────────────────────────────────
    // Returns the config value for key under [plugins.<plugin_id>] in server.toml.
    // Return value is engine-owned and valid only until the next call to get_config.
    // Returns NULL if the key does not exist.
    const char* (*get_config)(const char* plugin_id, const char* key);

    // ── Model metadata (valid after model load; NULL/"0" before) ─────────────
    const char* (*model_id)();            // alias or path
    const char* (*model_arch)();          // e.g. "llama", "qwen2"
    int64_t     (*model_context_len)();
    int64_t     (*model_vocab_size)();

    // ── Per-request accessors (take ctx pointer from callback args) ───────────
    const char* (*request_id)          (const truellm_context_t* ctx);
    int64_t     (*token_budget)        (const truellm_context_t* ctx);
    int32_t     (*request_max_tokens)  (const truellm_context_t* ctx);
    float       (*request_temperature) (const truellm_context_t* ctx);

    // ── Status reporting ─────────────────────────────────────────────────────
    // Sends a progress message to the streaming client (visible as a comment
    // or metadata chunk depending on the client).  No-op in non-streaming mode.
    void (*report_status)(const truellm_context_t* ctx, const char* message);

    // ── Engine-managed heap ──────────────────────────────────────────────────
    // Use these to allocate result payloads when you want the engine's allocator
    // to own the memory.  Pair with free_fn = host->dealloc in the result struct.
    void* (*alloc)  (size_t bytes);
    void  (*dealloc)(void*  ptr);

    // =========================================================================
    // api_minor >= 1 entries — check host->api_minor before calling these
    // =========================================================================

    // ── Tokenizer ─────────────────────────────────────────────────────────────
    // tokenize: engine-owned buffer; valid until next tokenize() call.
    // out_count receives the number of token IDs returned.
    const int32_t* (*tokenize)  (const char* text, int32_t* out_count);
    // detokenize: engine-owned string; valid until next detokenize() call.
    const char*    (*detokenize)(const int32_t* tokens, int32_t count);

    // ── Engine config access ──────────────────────────────────────────────────
    // Read any engine config value by dotted section + key.
    // Example: get_engine_config("inference.compression.presis", "keep_fraction") → "0.60"
    // Returns engine-owned string (stable for process lifetime). NULL if missing.
    const char* (*get_engine_config)(const char* section, const char* key);

    // ── Attention weights ─────────────────────────────────────────────────────
    // Returns device pointer to [num_heads × seq_len × kv_seq_len] F32 for the
    // given transformer layer from the most recent forward pass.
    // Buffer is engine-owned; valid until next generate() call completes.
    //
    // Prerequisites (check via get_server_capability before calling):
    //   TRUELLM_CAP_ATTN_WEIGHTS == "1"  (megakernel_save_attn=true + CUDA backend)
    //
    // Returns NULL when prerequisites not met.  *out_size_bytes set to byte length.
    //
    // USE CASE: interpretability probes, custom attention visualisers, research
    // compression scoring (e.g. custom alternatives to the engine's Presis pass).
    // The engine's own Presis scoring does NOT use this vtable entry.
    const void* (*get_attention_weights)(int32_t layer_idx, int64_t* out_size_bytes);

    // ── Hidden states ─────────────────────────────────────────────────────────
    // Returns device pointer to [seq_len × hidden_size] F16 residual stream
    // captured after the given layer's FFN from the most recent forward pass.
    // layer_idx == n_layers returns the final post-norm hidden state.
    // Engine-owned; valid until next generate(). NULL if unsupported.
    //
    // Prerequisites: TRUELLM_CAP_HIDDEN_STATES == "1"
    //
    // USE CASE: activation steering, probing classifiers, representation analysis,
    // research token merging (e.g. custom alternatives to the engine's ToMe pass).
    // The engine's own ToMe implementation does NOT use this vtable entry.
    const void* (*get_hidden_states)(int32_t layer_idx, int64_t* out_size_bytes);

    // ── KV cache tensor ───────────────────────────────────────────────────────
    // Returns device pointer to paged KV block tensor for the given layer.
    // out_shape[4] = {num_blocks, num_kv_heads, block_size, head_dim}.
    // type: 0 = keys, 1 = values.
    // Engine-owned; valid until next allocator reshape (rare).
    // Returns NULL on CPU backend or when paged-attention is not active.
    //
    // USE CASE: KV cache analysis plugins, custom research compression schemes,
    // codebook training on live KV data, cache inspection / profiling.
    // The engine's own VQ and KiVi passes do NOT use this vtable entry.
    const void* (*get_kv_cache_tensor)(int32_t layer_idx, int32_t type,
                                       int64_t out_shape[4]);

    // ── Sub-call (for ChunkEncoder / RLM) ─────────────────────────────────────
    // Direct C++ path to engine->generate(). ~1 µs overhead, no HTTP loopback.
    // Enforces ctx->recursion_depth < max_hierarchy_depth hard limit.
    // on_token: called for each generated token; done=1 on final token.
    // result_json_out: engine-allocated JSON string with completion result;
    //   caller must free with host->dealloc().
    // Returns TRUELLM_ERR_ABORT when depth limit exceeded.
    truellm_error_t (*call_engine)(
        const truellm_context_t*                   ctx,
        const char*                                messages_json,
        const char*                                gen_params_json,
        void (*on_token)(void* cb_data, const char* tok, int done),
        void*                                      cb_data,
        char**                                     result_json_out);

    // ── Token injection ───────────────────────────────────────────────────────
    // Insert tokens into the context at logical position pos.
    // tokens_json: JSON array of integer token IDs.
    // Returns TRUELLM_ERR_INVALID if pos is out of range.
    truellm_error_t (*inject_tokens)(
        truellm_context_t* ctx, int64_t pos, const char* tokens_json);

    // ── Per-request context metadata ─────────────────────────────────────────
    // Simple string key-value bag scoped to one request. Thread-safe for
    // concurrent plugin callbacks that share the same ctx.
    void        (*set_context_metadata)(truellm_context_t* ctx,
                                        const char* key, const char* val);
    const char* (*get_context_metadata)(const truellm_context_t* ctx,
                                        const char* key);

    // ── Server capability query ───────────────────────────────────────────────
    // Returns the runtime value for a named server capability.
    // Use TRUELLM_CAP_* constants above for key names.
    // Returns NULL when the capability is absent or not yet resolved.
    const char* (*get_server_capability)(const truellm_context_t* ctx,
                                         const char* cap_key);

    // =========================================================================
    // api_minor >= 2 entries — sandboxed plugin data dir + blob fs API.
    // Indexed-storage plugins (e.g. evidence_store) link libsqlite3 directly
    // through the SDK helper truellm/storage_sqlite.h; the engine just owns
    // the per-plugin sandbox path resolution.  See Code_Capability_design.md
    // §4.1 / §4.1b / §4.7 for the three storage tiers.
    //
    // Path policy
    //   * relpath is rejected if it contains "..", a leading separator, a
    //     drive letter (Windows), or a NUL byte.
    //   * The engine creates ${data_dir}/<plugin_id>/ at first use.
    //   * Writes are atomic: tmpfile + fsync + rename.  Concurrent writers
    //     to the same key serialise via flock(2) on a sibling .lock file —
    //     never via filesystem CAS, which fails under parallel fork agents.
    // =========================================================================

    // Returns the absolute, sandboxed per-plugin data directory.  The string
    // is engine-owned and stable for process lifetime.  NULL when plugins
    // have no data dir configured (operator turned the feature off).
    const char* (*plugin_data_dir)(const char* plugin_id);

    // Atomic blob write under <data_dir>/<plugin_id>/<relpath>.
    truellm_error_t (*plugin_fs_write)(const char* plugin_id,
                                        const char* relpath,
                                        const void* bytes, size_t n_bytes);

    // Read a blob.  *out_bytes is engine-allocated; caller frees with
    // host->dealloc().  Returns TRUELLM_ERR_NOT_FOUND if the file is absent.
    truellm_error_t (*plugin_fs_read)(const char* plugin_id,
                                       const char* relpath,
                                       void** out_bytes, size_t* out_n);

    // Remove a blob.  Returns TRUELLM_ERR_NOT_FOUND when absent (idempotent
    // unlink is the caller's responsibility — they can ignore NOT_FOUND).
    truellm_error_t (*plugin_fs_unlink)(const char* plugin_id,
                                         const char* relpath);

    // List a directory.  *out_names is a NULL-terminated engine-allocated
    // array of engine-allocated strings.  The caller frees each name and
    // the array via host->dealloc().  *out_count receives the entry count
    // (excluding the trailing NULL).
    truellm_error_t (*plugin_fs_listdir)(const char* plugin_id,
                                          const char* relpath,
                                          char*** out_names,
                                          size_t* out_count);

    // =========================================================================
    // api_minor >= 3 entries — cross-plugin tool dispatch.
    //
    // Generators in the federated set (research_orchestrator,
    // research_synthesizer, future agentic plugins) need to invoke other
    // plugins' tools as part of a single HTTP turn.  call_tool routes
    // through the same dispatch path the openai_router tool loop uses:
    //   - the qualified name is resolved against the registered tool set
    //     (short-name fallback supported);
    //   - the agent.allowed_tools enforcement (§3.3.2) applies just as it
    //     would for a model-emitted tool_call;
    //   - the tool's TRUELLM_OK / TRUELLM_ERR_PERMISSION / TRUELLM_ERR_*
    //     result is faithfully reported back.
    //
    // result_json_out is engine-allocated and shaped:
    //   { "ok": bool,            // true if the dispatch completed
    //     "error_code": int,     // tool's truellm_error_t value
    //     "error_msg": "...",    // empty on success
    //     "payload": "..." }     // tool's payload string (may be JSON)
    // Caller frees with host->dealloc().
    truellm_error_t (*call_tool)(
        const truellm_context_t* ctx,
        const char*              qualified_name,
        const char*              args_json,
        char**                   result_json_out);
};

#ifdef __cplusplus
}
#endif
#endif // TRUELLM_HOST_API_H
