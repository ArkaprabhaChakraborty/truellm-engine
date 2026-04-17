#pragma once
// ---------------------------------------------------------------------------
// plugin_context.h — Internal concrete type behind the opaque truellm_context_t.
// Not exported to SDK consumers; only plugin_bridge/ and openai_router.cpp use this.
// ---------------------------------------------------------------------------

#include <truellm/plugin.h>

#include <cstdint>
#include <functional>
#include <string>

// truellm_context_t is declared in the public context.h as an opaque C tag.
// We define the concrete struct here so C++ code in the engine can construct it.
struct truellm_context_t {
    std::string  request_id;
    std::string  model_id;
    std::string  model_arch;
    int64_t      token_budget  = 0;
    int32_t      max_tokens    = 512;
    float        temperature   = 0.7f;
    // Streaming sink for report_status; nullptr in non-streaming mode.
    std::function<void(const std::string&)> status_sink;
};
