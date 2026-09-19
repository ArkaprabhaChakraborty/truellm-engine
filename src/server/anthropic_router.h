#pragma once
// ---------------------------------------------------------------------------
// anthropic_router.h — native Anthropic Messages API (`/v1/messages`)
//
// Hosts the Anthropic-shaped API inside the engine so the `claude` CLI can
// point ANTHROPIC_BASE_URL straight at a TrueLLM server with no sidecar
// shim.  The router is a pure translation layer: it converts an Anthropic
// request to the OpenAI chat-completions shape, runs it through the
// existing OpenAIRouter pipeline (preprocessors + generate + tool loop),
// and converts the result back to an Anthropic `Message`.
//
// See Code_Capability_design.md §12.
// ---------------------------------------------------------------------------

#include "openai_router.h"     // TrueLLMConfig, AuthCheck, OpenAIRouter

#include <httplib.h>

namespace truellm {

class EngineInterface;

class AnthropicRouter {
public:
    AnthropicRouter(const TrueLLMConfig& cfg, EngineInterface* engine,
                    OpenAIRouter* openai);

    void register_routes(httplib::Server& svr, AuthCheck auth);

private:
    const TrueLLMConfig& cfg_;
    EngineInterface*     engine_;
    OpenAIRouter*        openai_;   // reused for the full inference pipeline

    void handle_messages     (const httplib::Request&, httplib::Response&);
    void handle_count_tokens (const httplib::Request&, httplib::Response&);
};

} // namespace truellm
