#pragma once
// ---------------------------------------------------------------------------
// native_router.h — /truellm/v1/* TrueLLM-specific endpoints
// ---------------------------------------------------------------------------

#include <truellm/config_schema.h>
#include <httplib.h>

#include <functional>

namespace truellm {

class EngineInterface;

using AuthCheck = std::function<bool(const httplib::Request&, httplib::Response&)>;

class PluginBridge;

class NativeRouter {
public:
    NativeRouter(const TrueLLMConfig& cfg, EngineInterface* engine);

    void register_routes(httplib::Server& svr, AuthCheck auth);

    void set_plugin_bridge(PluginBridge* pb) { plugin_bridge_ = pb; }

private:
    const TrueLLMConfig& cfg_;
    EngineInterface*     engine_;
    PluginBridge*        plugin_bridge_ = nullptr;

    std::chrono::steady_clock::time_point start_time_ =
        std::chrono::steady_clock::now();

    void handle_status       (const httplib::Request&, httplib::Response&);
    void handle_plugins      (const httplib::Request&, httplib::Response&);
    void handle_rlm_status   (const httplib::Request&, httplib::Response&);
    void handle_compression_status(const httplib::Request&, httplib::Response&);
    void handle_chat_status  (const httplib::Request&, httplib::Response&);
    void handle_research_status(const httplib::Request&, httplib::Response&);
    void handle_tools_dispatch(const httplib::Request&, httplib::Response&);
};

} // namespace truellm
