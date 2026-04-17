#pragma once
// ---------------------------------------------------------------------------
// tool_registry.h — Thread-safe tool provider cache + dispatch
// ---------------------------------------------------------------------------

#include <truellm/plugin.h>
#include "plugin_bridge/plugin_context.h"

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>

namespace truellm {

// ---------------------------------------------------------------------------
// ToolCallResult — returned from dispatch()
// ---------------------------------------------------------------------------
struct ToolCallResult {
    std::string     payload;
    truellm_error_t error     = TRUELLM_OK;
    std::string     error_msg;
};

// ---------------------------------------------------------------------------
// ToolRegistry
// ---------------------------------------------------------------------------
class ToolRegistry {
public:
    ToolRegistry();

    // Called by PluginBridge after native_loader successfully initialises a plugin.
    void register_provider(truellm_tool_provider_t* provider);

    // Called by PluginBridge on plugin unload/disable (native).
    void unregister_plugin(const std::string& plugin_id);

    // ── Remote (IPC-backed) tools ─────────────────────────────────────────────
    // Called by PluginBridge::on_sidecar_announce for each tool in the announce.
    // dispatch_fn is called on every tool invocation; it wraps IpcClient::call_tool.
    using DispatchFn = std::function<ToolCallResult(const std::string& args_json,
                                                     truellm_context_t* ctx)>;
    void register_remote(const std::string& plugin_id,
                         const std::string& tool_name,
                         const std::string& schema_json,
                         DispatchFn         dispatch_fn);

    // Remove all remote entries belonging to a sidecar plugin (e.g. on restart).
    void unregister_remote_plugin(const std::string& plugin_id);

    // Returns an OpenAI-format JSON array string: [{type:"function",...}, ...]
    // Returns "[]" if no tools are registered.
    // Shared-lock read — concurrent reads don't block each other.
    std::string get_schemas_json() const;

    // Returns total registered tool count.
    std::size_t tool_count() const;

    // Execute a named tool call.
    // qualified_name: "plugin_id.tool_name"
    ToolCallResult dispatch(const std::string& qualified_name,
                            const std::string& args_json,
                            truellm_context_t* ctx);

private:
    struct Entry {
        std::string              plugin_id;
        std::string              tool_name;       // unqualified
        std::string              qualified_name;  // "plugin_id.tool_name"
        std::string              schema_json;     // pre-serialized OpenAI tool object
        truellm_tool_provider_t* provider;        // non-owning
    };

    // ── Remote entry (IPC-backed) ─────────────────────────────────────────────
    struct RemoteEntry {
        std::string plugin_id;
        std::string tool_name;
        std::string qualified_name;
        std::string schema_json;
        DispatchFn  dispatch_fn;
    };

    mutable std::shared_mutex    mutex_;
    std::vector<Entry>           entries_;         // native (C ABI) tools
    std::vector<RemoteEntry>     remote_entries_;  // IPC-backed sidecar tools
    std::string                  schemas_cache_;   // JSON array of all schemas

    void rebuild_cache_locked();  // must hold unique lock
};

} // namespace truellm
