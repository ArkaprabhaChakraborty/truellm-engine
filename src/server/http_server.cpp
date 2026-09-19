// ---------------------------------------------------------------------------
// http_server.cpp
// ---------------------------------------------------------------------------

#include "http_server.h"

#include "../inference/engine_interface.h"
#include "../plugin_bridge/plugin_bridge.h"
#include "../policy/rate_limiter.h"
#include "../policy/guardrail_policy.h"

#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>

namespace truellm {

HttpServer::HttpServer(const TrueLLMConfig& cfg, EngineInterface* engine)
    : cfg_(cfg)
    , engine_(engine)
    , openai_router_(cfg, engine)
    , native_router_(cfg, engine)
    , anthropic_router_(cfg, engine, &openai_router_)
{
    // Build a rate limiter config from the guardrail preset so that rate
    // limiting tightens and loosens together with memory guardrails.
    auto rl_cfg = policy::config_for_guardrail_preset(
        cfg_.hardware.guardrails.preset);
    rate_limiter_ = std::make_unique<policy::RateLimiter>(rl_cfg);
}

HttpServer::~HttpServer() = default;

void HttpServer::set_plugin_bridge(PluginBridge* pb)
{
    plugin_bridge_ = pb;
    openai_router_.set_plugin_bridge(pb);
    native_router_.set_plugin_bridge(pb);
}

// ---------------------------------------------------------------------------
// setup_routes
// ---------------------------------------------------------------------------
void HttpServer::setup_routes()
{
    // ---- HTTP request/response logger (via logging middleware) ----
    svr_.set_logger(middleware::make_request_logger());

    // ---- Timestamp injection + rate limiting (pre-routing) ----
    auto timestamp_fn = middleware::make_timestamp_injector();
    svr_.set_pre_routing_handler(
        [this, timestamp_fn](const httplib::Request& req, httplib::Response& res)
            -> httplib::Server::HandlerResponse
        {
            // 1. Stamp start time for elapsed-ms calculation in logger.
            timestamp_fn(req, res);

            // 2. Rate limiting — skip for health probe to avoid false 429s.
            if (req.path != "/health" && rate_limiter_->config().enabled) {
                const std::string key = req.remote_addr.empty() ? "unknown" : req.remote_addr;
                if (!rate_limiter_->try_acquire(key)) {
                    nlohmann::json err = {
                        {"error", {
                            {"message", "Too Many Requests — rate limit exceeded"},
                            {"type",    "rate_limit_error"}
                        }}
                    };
                    res.status = 429;
                    res.set_header("Retry-After", "1");
                    res.set_content(err.dump(), "application/json");
                    return httplib::Server::HandlerResponse::Handled;
                }
            }

            return httplib::Server::HandlerResponse::Unhandled;
        });

    // ---- CORS pre-flight ----
    svr_.Options(".*", [&](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin",  resolve_cors_origin(req));
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers",
                       "Content-Type, Authorization");
        res.status = 204;
    });

    // ---- Liveness probe ----
    svr_.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(R"({"status":"ok"})", "application/json");
    });

    // ---- OpenAI-compatible routes ----
    openai_router_.register_routes(svr_, [this](const httplib::Request& req,
                                                 httplib::Response& res) {
        return check_auth(req, res);
    });

    // ---- TrueLLM native routes ----
    native_router_.register_routes(svr_, [this](const httplib::Request& req,
                                                 httplib::Response& res) {
        return check_auth(req, res);
    });

    // ---- Anthropic Messages API (/v1/messages) — Code_Capability_design.md §12 ----
    anthropic_router_.register_routes(svr_, [this](const httplib::Request& req,
                                                    httplib::Response& res) {
        return check_auth(req, res);
    });

    // ---- CORS response headers on all routes ----
    svr_.set_post_routing_handler([&](const httplib::Request& req,
                                      httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", resolve_cors_origin(req));
    });

    // ---- Request timeout ----
    svr_.set_read_timeout(cfg_.server.request_timeout_ms / 1000,
                          (cfg_.server.request_timeout_ms % 1000) * 1000);
}

// ---------------------------------------------------------------------------
// listen / stop
// ---------------------------------------------------------------------------
void HttpServer::listen()
{
    spdlog::info("[HttpServer] Listening on {}:{}", cfg_.server.host, cfg_.server.port);
    svr_.listen(cfg_.server.host.c_str(), cfg_.server.port);
}

void HttpServer::stop()
{
    svr_.stop();
}

// ---------------------------------------------------------------------------
// check_auth — delegates to auth middleware
// ---------------------------------------------------------------------------
bool HttpServer::check_auth(const httplib::Request& req, httplib::Response& res) const
{
    return middleware::check_api_key(req, res, cfg_.server.api_key);
}

// ---------------------------------------------------------------------------
// resolve_cors_origin — per-request CORS origin selection
//
// The CORS spec does not allow a comma-separated list in Allow-Origin; it
// accepts only "*" or a single specific origin.  For multi-origin configs we
// inspect the request Origin header and reflect it back when it matches the
// allowlist.  This satisfies both browser pre-flight and credentialed requests.
// ---------------------------------------------------------------------------
std::string HttpServer::resolve_cors_origin(const httplib::Request& req) const
{
    const auto& origins = cfg_.server.cors_origins;
    if (origins.empty()) return "*";

    const std::string request_origin = req.get_header_value("Origin");
    if (!request_origin.empty()) {
        auto it = std::find(origins.begin(), origins.end(), request_origin);
        if (it != origins.end()) return request_origin;
    }

    // Origin not in allowlist — return first configured origin so the browser
    // blocks the request (as intended) rather than silently allowing it.
    return origins[0];
}

} // namespace truellm
