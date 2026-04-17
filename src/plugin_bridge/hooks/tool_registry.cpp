// ---------------------------------------------------------------------------
// tool_registry.cpp
// ---------------------------------------------------------------------------

#include "tool_registry.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>

namespace truellm {

using json = nlohmann::json;

ToolRegistry::ToolRegistry()
{
    schemas_cache_ = "[]";
}

// ---------------------------------------------------------------------------
// register_provider
// ---------------------------------------------------------------------------
void ToolRegistry::register_provider(truellm_tool_provider_t* provider)
{
    if (!provider || !provider->plugin_id) return;

    const truellm_tool_def_t* const* tools = provider->get_tools
        ? provider->get_tools(provider->state)
        : nullptr;
    if (!tools) return;

    std::string pid     = provider->plugin_id;
    std::string version = provider->plugin_version ? provider->plugin_version : "";

    std::unique_lock<std::shared_mutex> lk(mutex_);

    for (int i = 0; tools[i] != nullptr; ++i) {
        const auto* td = tools[i];
        if (!td->name) continue;

        Entry e;
        e.plugin_id      = pid;
        e.tool_name      = td->name;
        e.qualified_name = pid + "." + td->name;
        e.provider       = provider;

        // Build pre-serialised OpenAI tool object
        json params;
        if (td->parameters_json) {
            params = json::parse(td->parameters_json, nullptr, /*exceptions=*/false);
            if (params.is_discarded()) params = json::object();
        } else {
            params = json::object();
        }

        json obj = {
            {"type", "function"},
            {"function", {
                {"name",        e.qualified_name},
                {"description", td->description ? td->description : ""},
                {"parameters",  params}
            }}
        };
        e.schema_json = obj.dump();

        entries_.push_back(std::move(e));
        spdlog::info("[ToolRegistry] registered tool '{}'  plugin={}",
                     entries_.back().qualified_name, pid);
    }

    rebuild_cache_locked();
}

// ---------------------------------------------------------------------------
// register_remote
// ---------------------------------------------------------------------------
void ToolRegistry::register_remote(const std::string& plugin_id,
                                    const std::string& tool_name,
                                    const std::string& schema_json,
                                    DispatchFn         dispatch_fn)
{
    if (plugin_id.empty() || tool_name.empty() || !dispatch_fn) return;

    RemoteEntry e;
    e.plugin_id      = plugin_id;
    e.tool_name      = tool_name;
    e.qualified_name = plugin_id + "." + tool_name;
    e.dispatch_fn    = std::move(dispatch_fn);

    // schema_json is the full pre-serialised OpenAI tool object from the sidecar.
    // Accept it as-is; validate it is non-empty.
    e.schema_json = schema_json.empty() ? "{}" : schema_json;

    std::unique_lock<std::shared_mutex> lk(mutex_);
    remote_entries_.push_back(std::move(e));
    rebuild_cache_locked();
    spdlog::info("[ToolRegistry] registered remote tool '{}'  plugin={}",
                 remote_entries_.back().qualified_name, plugin_id);
}

// ---------------------------------------------------------------------------
// unregister_remote_plugin
// ---------------------------------------------------------------------------
void ToolRegistry::unregister_remote_plugin(const std::string& plugin_id)
{
    std::unique_lock<std::shared_mutex> lk(mutex_);
    auto it = std::remove_if(remote_entries_.begin(), remote_entries_.end(),
        [&](const RemoteEntry& e) { return e.plugin_id == plugin_id; });
    remote_entries_.erase(it, remote_entries_.end());
    rebuild_cache_locked();
    spdlog::info("[ToolRegistry] unregistered remote tools for plugin '{}'", plugin_id);
}

// ---------------------------------------------------------------------------
// unregister_plugin
// ---------------------------------------------------------------------------
void ToolRegistry::unregister_plugin(const std::string& plugin_id)
{
    std::unique_lock<std::shared_mutex> lk(mutex_);

    auto it = std::remove_if(entries_.begin(), entries_.end(),
        [&](const Entry& e) { return e.plugin_id == plugin_id; });
    entries_.erase(it, entries_.end());

    rebuild_cache_locked();
    spdlog::info("[ToolRegistry] unregistered all tools for plugin '{}'", plugin_id);
}

// ---------------------------------------------------------------------------
// get_schemas_json
// ---------------------------------------------------------------------------
std::string ToolRegistry::get_schemas_json() const
{
    std::shared_lock<std::shared_mutex> lk(mutex_);
    return schemas_cache_;
}

// ---------------------------------------------------------------------------
// tool_count
// ---------------------------------------------------------------------------
std::size_t ToolRegistry::tool_count() const
{
    std::shared_lock<std::shared_mutex> lk(mutex_);
    return entries_.size() + remote_entries_.size();
}

// ---------------------------------------------------------------------------
// dispatch
// ---------------------------------------------------------------------------
ToolCallResult ToolRegistry::dispatch(const std::string& qualified_name,
                                      const std::string& args_json,
                                      truellm_context_t* ctx)
{
    // Take a shared lock just long enough to snapshot what we need.
    truellm_tool_provider_t* provider   = nullptr;
    std::string              tool_name;
    DispatchFn               remote_fn;
    {
        std::shared_lock<std::shared_mutex> lk(mutex_);
        // Check native entries first.
        for (const auto& e : entries_) {
            if (e.qualified_name == qualified_name) {
                provider  = e.provider;
                tool_name = e.tool_name;
                break;
            }
        }
        // If not native, check remote (IPC-backed) entries.
        if (!provider) {
            for (const auto& e : remote_entries_) {
                if (e.qualified_name == qualified_name) {
                    remote_fn = e.dispatch_fn;
                    break;
                }
            }
        }

        // Short-name fallback: the caller may have passed just "fetch_url"
        // instead of the qualified "tool_web_fetch.fetch_url".
        // Try matching by tool_name alone so models using short names still work.
        if (!provider && !remote_fn) {
            for (const auto& e : entries_) {
                if (e.tool_name == qualified_name) {
                    provider  = e.provider;
                    tool_name = e.tool_name;
                    break;
                }
            }
            if (!provider) {
                for (const auto& e : remote_entries_) {
                    if (e.tool_name == qualified_name) {
                        remote_fn = e.dispatch_fn;
                        break;
                    }
                }
            }
        }
    }

    if (!provider && !remote_fn) {
        spdlog::warn("[ToolRegistry] dispatch: tool not found: '{}'", qualified_name);
        return {"", TRUELLM_ERR_NOT_FOUND,
                "Tool not found: " + qualified_name};
    }

    spdlog::debug("[ToolRegistry] dispatch '{}' args_len={}", qualified_name, args_json.size());

    // ── Remote (IPC) dispatch ─────────────────────────────────────────────────
    if (remote_fn) {
        return remote_fn(args_json, ctx);
    }

    // ── Native dispatch ───────────────────────────────────────────────────────
    truellm_tool_result_t raw = provider->call_tool(
        provider->state, ctx, tool_name.c_str(), args_json.c_str());

    // Copy payload out before freeing plugin memory
    std::string payload = raw.payload ? raw.payload : "";
    truellm_error_t err = raw.error;

    if (raw.free_fn) raw.free_fn(raw.free_arg);

    if (err != TRUELLM_OK) {
        spdlog::warn("[ToolRegistry] tool '{}' returned error {}", qualified_name,
                     static_cast<int>(err));
        return {std::move(payload), err, "Tool call failed (err=" + std::to_string(err) + ")"};
    }

    return {std::move(payload), TRUELLM_OK, ""};
}

// ---------------------------------------------------------------------------
// rebuild_cache_locked  (caller holds unique_lock)
// ---------------------------------------------------------------------------
void ToolRegistry::rebuild_cache_locked()
{
    if (entries_.empty() && remote_entries_.empty()) {
        schemas_cache_ = "[]";
        return;
    }
    std::string out = "[";
    bool first = true;
    for (const auto& e : entries_) {
        if (!first) out += ',';
        out += e.schema_json;
        first = false;
    }
    for (const auto& e : remote_entries_) {
        if (!first) out += ',';
        out += e.schema_json;
        first = false;
    }
    out += ']';
    schemas_cache_ = std::move(out);
}

} // namespace truellm
