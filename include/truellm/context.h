// ---------------------------------------------------------------------------
// context.h — Opaque per-request context passed to all plugin callbacks.
// Access fields exclusively through the host_api vtable — never cast or
// dereference directly.  Layout is internal and may change between minor versions.
// ---------------------------------------------------------------------------
#pragma once
#ifndef TRUELLM_CONTEXT_H
#define TRUELLM_CONTEXT_H

#ifdef __cplusplus
extern "C" {
#endif

// Engine-side definition lives in plugin_bridge/plugin_context.h.
// The public header intentionally exposes only the opaque tag.
typedef struct truellm_context_t truellm_context_t;

#ifdef __cplusplus
}
#endif
#endif // TRUELLM_CONTEXT_H
