#pragma once
// ---------------------------------------------------------------------------
// generator_registry.h — Phase D: priority-ordered generator registry.
// ---------------------------------------------------------------------------

#include <truellm/plugin.h>
#include "plugin_bridge/plugin_context.h"

#include <memory>
#include <string>

namespace truellm {

// ---------------------------------------------------------------------------
// GeneratorDispatchOutcome — result of attempting to dispatch through a
// priority-ordered generator chain.
// ---------------------------------------------------------------------------
struct GeneratorDispatchOutcome {
    bool            claimed       = false;     // did any generator claim the request
    bool            ok            = false;     // did generate() succeed
    std::string     text;                      // assembled output (non-streaming path)
    std::string     plugin_id;                 // id of the generator that claimed
    truellm_error_t error         = TRUELLM_OK;
    std::string     error_message;
};

// ---------------------------------------------------------------------------
// GeneratorRegistry
// ---------------------------------------------------------------------------
class GeneratorRegistry {
public:
    GeneratorRegistry();
    ~GeneratorRegistry();

    // Register a native generator provider. Thread-safe.
    void register_generator(truellm_generator_t* gen);

    // Unregister all generators for a plugin. Thread-safe.
    void unregister_plugin(const std::string& plugin_id);

    // Returns true if any generator is registered.
    bool has_generators() const;

    // Returns true if any registered generator claims this request.
    bool claims(const std::string& messages_json, truellm_context_t* ctx);

    // Walk generators in priority order; first one that claims the request
    // also produces the response.  Returns an outcome whose .claimed=false if
    // no generator matched (caller should fall through to the default engine
    // path).  Token callback receives each emitted piece; is_done=1 on final.
    GeneratorDispatchOutcome dispatch(
        const std::string&                              messages_json,
        truellm_context_t*                              ctx,
        void (*on_token)(void* cb_data,
                         const char* token_text,
                         int         is_done) = nullptr,
        void*                                           cb_data  = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace truellm
