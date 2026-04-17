#pragma once
// ---------------------------------------------------------------------------
// preprocessor_chain.h — Phase D: ordered, thread-safe preprocessor chain.
// ---------------------------------------------------------------------------

#include <truellm/plugin.h>
#include "plugin_bridge/plugin_context.h"

#include <memory>
#include <string>

namespace truellm {

// ---------------------------------------------------------------------------
// PreprocessResult — returned from run()
// ---------------------------------------------------------------------------
struct PreprocessResult {
    truellm_preproc_decision_t decision              = TRUELLM_PREPROC_PASS;
    std::string                modified_messages_json;  // "" = unchanged
    std::string                response_text;           // for RESPOND decision
};

// ---------------------------------------------------------------------------
// PreprocessorChain
// ---------------------------------------------------------------------------
class PreprocessorChain {
public:
    PreprocessorChain();
    ~PreprocessorChain();

    // Register a native preprocessor provider. Thread-safe.
    void register_preprocessor(truellm_preprocessor_t* prep);

    // Unregister all preprocessors for a plugin. Thread-safe.
    void unregister_plugin(const std::string& plugin_id);

    // Returns true if any preprocessor is registered.
    bool has_preprocessors() const;

    // Run the chain in priority order.
    // Returns PASS with updated messages_json, RESPOND with response_text,
    // or ABORT on a veto.
    PreprocessResult run(const std::string& messages_json, truellm_context_t* ctx);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace truellm
