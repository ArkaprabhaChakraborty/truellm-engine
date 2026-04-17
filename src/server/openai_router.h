#pragma once
// ---------------------------------------------------------------------------
// openai_router.h — /v1/* OpenAI-compatible endpoints
// ---------------------------------------------------------------------------

#include <truellm/config_schema.h>
#include <httplib.h>

#include <functional>

namespace truellm {

class EngineInterface;
class PluginBridge;

using AuthCheck = std::function<bool(const httplib::Request&, httplib::Response&)>;

class OpenAIRouter {
public:
    OpenAIRouter(const TrueLLMConfig& cfg, EngineInterface* engine);

    void register_routes(httplib::Server& svr, AuthCheck auth);

    // Set after plugin bridge is initialised (optional — nullptr = no plugins).
    void set_plugin_bridge(PluginBridge* pb) { plugin_bridge_ = pb; }

private:
    const TrueLLMConfig& cfg_;
    EngineInterface*     engine_;
    PluginBridge*        plugin_bridge_ = nullptr;

    void handle_models         (const httplib::Request&, httplib::Response&);
    void handle_completions    (const httplib::Request&, httplib::Response&);
    void handle_chat_completions(const httplib::Request&, httplib::Response&);

    // Context splitter: summarise each chunk of an oversized tool result and
    // return the concatenated per-chunk summaries.  Called when a tool response
    // would exceed the model's context window.
    std::string summarize_chunks(const std::vector<std::string>& chunks,
                                 const std::string& user_question,
                                 const std::string& tmpl,
                                 int                summary_tokens,
                                 float              temperature) const;
};

} // namespace truellm
