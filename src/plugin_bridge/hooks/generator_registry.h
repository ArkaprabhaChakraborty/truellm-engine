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

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace truellm
