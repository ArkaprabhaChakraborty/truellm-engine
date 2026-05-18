// ---------------------------------------------------------------------------
// plugin.h — TrueLLM Plugin C ABI v1
// LOAD-BEARING: ground truth for all SDKs. Do not modify without a version bump.
// ---------------------------------------------------------------------------
#pragma once
#ifndef TRUELLM_PLUGIN_H
#define TRUELLM_PLUGIN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── ABI version ───────────────────────────────────────────────────────────────
#define TRUELLM_ABI_VERSION  1u
#define TRUELLM_ABI_MINOR    3u

// ── Export annotation ─────────────────────────────────────────────────────────
// TRUELLM_PLUGIN_EXPORT must decorate truellm_plugin_init in every plugin DLL.
// The engine side (which uses GetProcAddress / dlsym) does not need dllimport.
//
// Define TRUELLM_PLUGIN_BUILDING when compiling a plugin shared library so that
// the declaration becomes __declspec(dllexport) on Windows.  CMake's
// target_compile_definitions should set this for every plugin target.
#if defined(_WIN32)
#  if defined(TRUELLM_PLUGIN_BUILDING)
#    define TRUELLM_PLUGIN_EXPORT __declspec(dllexport)
#  else
#    define TRUELLM_PLUGIN_EXPORT
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define TRUELLM_PLUGIN_EXPORT __attribute__((visibility("default")))
#else
#  define TRUELLM_PLUGIN_EXPORT
#endif

// ── Forward declarations ──────────────────────────────────────────────────────
typedef struct truellm_host_api_t    truellm_host_api_t;
typedef struct truellm_context_t     truellm_context_t;

// ── Error codes (must mirror types.h ErrorCode values) ───────────────────────
typedef enum {
    TRUELLM_OK             = 0,
    TRUELLM_ERR_INVALID    = 1,
    TRUELLM_ERR_NOT_FOUND  = 2,
    TRUELLM_ERR_INTERNAL   = 3,
    TRUELLM_ERR_TIMEOUT    = 4,
    TRUELLM_ERR_PERMISSION = 5,
    TRUELLM_ERR_ABORT      = 6,
} truellm_error_t;

// ── Hook flags (combinable bitmask) ──────────────────────────────────────────
#define TRUELLM_HOOK_TOOL_PROVIDER 0x01u
#define TRUELLM_HOOK_PREPROCESSOR  0x02u
#define TRUELLM_HOOK_GENERATOR     0x04u
#define TRUELLM_HOOK_CONFIG_SCHEMA 0x08u

// ── Memory contract ───────────────────────────────────────────────────────────
// Strings the plugin returns to the engine are allocated by the plugin and
// freed by the engine through the free_fn pointer.  This lets each language
// runtime use its own allocator.
//
// Strings the engine passes to the plugin are owned by the engine and are
// valid only for the duration of that function call.  Copy if you need to
// retain them.

// ── Tool result ───────────────────────────────────────────────────────────────
typedef struct {
    truellm_error_t  error;
    const char*      payload;              // result text (JSON or plain); plugin-allocated
    void           (*free_fn)(void* arg);  // engine calls this to free payload
    void*            free_arg;             // passed verbatim to free_fn
} truellm_tool_result_t;

// ── Tool definition ───────────────────────────────────────────────────────────
typedef struct {
    const char* name;             // unique within the plugin's namespace
    const char* description;      // shown to the model
    const char* parameters_json;  // OpenAI-compatible JSON Schema string
} truellm_tool_def_t;

// ── Preprocessor decision ─────────────────────────────────────────────────────
typedef enum {
    TRUELLM_PREPROC_PASS    = 0,  // continue chain with (optionally modified) context
    TRUELLM_PREPROC_RESPOND = 1,  // bypass inference; return response_text directly
    TRUELLM_PREPROC_ABORT   = 2,  // return an error response
} truellm_preproc_decision_t;

typedef struct {
    truellm_preproc_decision_t  decision;
    const char*                 modified_messages_json; // NULL = no change; plugin-allocated
    const char*                 response_text;          // for RESPOND; plugin-allocated
    void                      (*free_fn)(void* arg);   // covers both strings above
    void*                       free_arg;
} truellm_preproc_result_t;

// ── ToolProvider provider struct ──────────────────────────────────────────────
typedef struct {
    uint32_t    abi_version;      // must equal TRUELLM_ABI_VERSION
    uint32_t    abi_minor;
    uint32_t    struct_size;      // sizeof(truellm_tool_provider_t)
    const char* plugin_id;        // "my_plugin" — unique, [a-z0-9_] only; plugin-owned
    const char* plugin_version;   // semver; plugin-owned

    // Return NULL-terminated array of tool definitions; array is plugin-owned.
    // Engine calls this once at registration; pointer must remain valid until destroy().
    const truellm_tool_def_t* const* (*get_tools)(void* state);

    // Execute a named tool call.
    // args_json is engine-owned and valid only during this call — copy if needed.
    truellm_tool_result_t (*call_tool)(
        void*                    state,
        const truellm_context_t* ctx,
        const char*              tool_name,
        const char*              args_json
    );

    void* state;                  // opaque plugin state; first arg to all callbacks
    void (*destroy)(void* state); // called on unload; must free all plugin memory
} truellm_tool_provider_t;

// ── Preprocessor provider struct ─────────────────────────────────────────────
typedef struct {
    uint32_t    abi_version;
    uint32_t    abi_minor;
    uint32_t    struct_size;
    const char* plugin_id;
    const char* plugin_version;
    int32_t     priority;         // lower = earlier in chain; 0 = default

    // messages_json: current messages array as JSON; engine-owned, call-duration only.
    truellm_preproc_result_t (*preprocess)(
        void*                    state,
        const truellm_context_t* ctx,
        const char*              messages_json
    );

    void* state;
    void (*destroy)(void* state);
} truellm_preprocessor_t;

// ── Generator provider struct ─────────────────────────────────────────────────
typedef struct {
    uint32_t    abi_version;
    uint32_t    abi_minor;
    uint32_t    struct_size;
    const char* plugin_id;
    const char* plugin_version;
    int32_t     priority;

    // Return non-zero if this generator claims the request.
    int (*claims)(void* state, const truellm_context_t* ctx,
                  const char* messages_json);

    // Produce a response.  Call on_token for each token (streaming path);
    // also return the full result for the non-streaming path.
    // on_token: token_text is the piece string; is_done=1 on the final call.
    truellm_tool_result_t (*generate)(
        void*                    state,
        const truellm_context_t* ctx,
        const char*              messages_json,
        void (*on_token)(void* cb_data, const char* token_text, int is_done),
        void*                    cb_data
    );

    void* state;
    void (*destroy)(void* state);
} truellm_generator_t;

// ── Plugin entry point ────────────────────────────────────────────────────────
// Every plugin shared library exports exactly this symbol.
// Set output pointers to NULL for hook types the plugin does not implement.
// The engine reads abi_version from each non-NULL provider before using it.
TRUELLM_PLUGIN_EXPORT truellm_error_t truellm_plugin_init(
    const truellm_host_api_t*  host,
    truellm_tool_provider_t**  tool_out,
    truellm_preprocessor_t**   prep_out,
    truellm_generator_t**      gen_out
);

#ifdef __cplusplus
}
#endif
#endif // TRUELLM_PLUGIN_H
