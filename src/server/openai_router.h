#pragma once
// ---------------------------------------------------------------------------
// openai_router.h — /v1/* OpenAI-compatible endpoints
// ---------------------------------------------------------------------------

#include <truellm/config_schema.h>
#include "chat_format.h"
#include <httplib.h>

#include <functional>
#include <string>

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

    // §12 — run the full /v1/chat/completions pipeline on an
    // already-built request/response pair.  AnthropicRouter calls this
    // to reuse the entire preprocessor + generate + tool pipeline behind
    // the /v1/messages translation layer instead of duplicating it.
    void run_chat_completions(const httplib::Request& req,
                              httplib::Response&      res) {
        handle_chat_completions(req, res);
    }

private:
    const TrueLLMConfig& cfg_;
    EngineInterface*     engine_;
    PluginBridge*        plugin_bridge_ = nullptr;

    // common_chat-backed templating + reasoning parsing (src/server/chat_format).
    // Lazily built from the loaded model's embedded template on first use; falls
    // back to the legacy chat_template.h path when unavailable/invalid.
    ChatFormatter        chat_fmt_;
    bool                 chat_fmt_tried_ = false;
    void                 ensure_chat_formatter();
    bool                 reasoning_active() const {
        return chat_fmt_.valid() && cfg_.inference.reasoning.format != "none";
    }

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
