#pragma once

// ---------------------------------------------------------------------------
// version.h — TrueLLM version constants and ABI version
//
// TRUELLM_ABI_VERSION: bump on any breaking change to plugin.h / host_api.h.
// Major version bump = ABI break.  Minor = additive-only.
// ---------------------------------------------------------------------------

#define TRUELLM_VERSION_MAJOR 0
#define TRUELLM_VERSION_MINOR 1
#define TRUELLM_VERSION_PATCH 0

// ABI version: plugins compiled against ABI N refuse to load on ABI N-1 / N+1
#define TRUELLM_ABI_VERSION 1
// Minor ABI version: additive-only extensions (new vtable entries appended).
// Plugins may check api_minor to test for new host_api.h entries.
#define TRUELLM_ABI_MINOR   1

// Stringified version for display
#define TRUELLM_VERSION_STRING "0.1.0"

namespace truellm {

struct Version {
    static constexpr int major     = TRUELLM_VERSION_MAJOR;
    static constexpr int minor     = TRUELLM_VERSION_MINOR;
    static constexpr int patch     = TRUELLM_VERSION_PATCH;
    static constexpr int abi       = TRUELLM_ABI_VERSION;
    static constexpr const char* string = TRUELLM_VERSION_STRING;
};

} // namespace truellm
