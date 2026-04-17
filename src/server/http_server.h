#pragma once
// ---------------------------------------------------------------------------
// http_server.h — cpp-httplib server setup + lifecycle
// ---------------------------------------------------------------------------

#include <truellm/config_schema.h>
#include "openai_router.h"
#include "native_router.h"
#include "middleware/auth_middleware.h"
#include "middleware/logging_middleware.h"

#define CPPHTTPLIB_THREAD_POOL_COUNT 8
#include <httplib.h>

#include <memory>

namespace truellm {

class EngineInterface;
class PluginBridge;
namespace policy { class RateLimiter; }

class HttpServer {
public:
    HttpServer(const TrueLLMConfig& cfg, EngineInterface* engine);
    ~HttpServer();

    // Call before setup_routes() to enable plugin endpoints and tool dispatch.
    void set_plugin_bridge(PluginBridge* pb);

    void setup_routes();
    void listen();  // blocks until stop()
    void stop();

private:
    const TrueLLMConfig& cfg_;
    EngineInterface*     engine_;
    PluginBridge*        plugin_bridge_ = nullptr;
    httplib::Server      svr_;

    // Routers must outlive the route lambdas they register — keep them as members.
    OpenAIRouter openai_router_;
    NativeRouter native_router_;

    // Per-IP rate limiter; constructed from hardware.guardrails.preset.
    std::unique_ptr<policy::RateLimiter> rate_limiter_;

    // Returns false (and writes 401) when auth fails.
    bool check_auth(const httplib::Request& req, httplib::Response& res) const;

    // Returns the Access-Control-Allow-Origin value for this request.
    // If cors_origins is empty → "*".
    // If the request Origin matches an entry in cors_origins → reflect it back.
    // Otherwise → the first configured origin (browsers will block the request).
    std::string resolve_cors_origin(const httplib::Request& req) const;
};

} // namespace truellm
