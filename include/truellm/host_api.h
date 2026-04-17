// ---------------------------------------------------------------------------
// host_api.h — Vtable the engine passes to truellm_plugin_init.
// Valid for the lifetime of the engine process.  Thread-safe: all functions
// may be called from any thread.
// ---------------------------------------------------------------------------
#pragma once
#ifndef TRUELLM_HOST_API_H
#define TRUELLM_HOST_API_H

#include "plugin.h"

#ifdef __cplusplus
extern "C" {
#endif

struct truellm_host_api_t {
    uint32_t api_version;   // TRUELLM_ABI_VERSION
    uint32_t api_minor;

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
};

#ifdef __cplusplus
}
#endif
#endif // TRUELLM_HOST_API_H
