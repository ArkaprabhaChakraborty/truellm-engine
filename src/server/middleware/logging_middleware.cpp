// ---------------------------------------------------------------------------
// logging_middleware.cpp
// ---------------------------------------------------------------------------

#include "logging_middleware.h"

#include <spdlog/spdlog.h>

#include <chrono>
#include <string>

namespace truellm::middleware {

// ---------------------------------------------------------------------------
// make_request_logger
// ---------------------------------------------------------------------------
RequestLogger make_request_logger()
{
    return [](const httplib::Request& req, const httplib::Response& res) {
        if (!spdlog::should_log(spdlog::level::debug)) return;

        int64_t elapsed_ms = 0;
        if (req.has_header("X-Request-Start-Us")) {
            try {
                int64_t start_us = std::stoll(req.get_header_value("X-Request-Start-Us"));
                int64_t now_us   = static_cast<int64_t>(
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()).count());
                elapsed_ms = (now_us - start_us) / 1000;
            } catch (...) {}
        }

        spdlog::debug("[HTTP] {} {} {} -> {} (+{}ms)",
                      req.method,
                      req.path,
                      req.remote_addr.empty() ? "-" : req.remote_addr,
                      res.status,
                      elapsed_ms);

        if (spdlog::should_log(spdlog::level::trace)) {
            if (!req.body.empty()) {
                std::string snippet = req.body.substr(0, 512);
                if (req.body.size() > 512) snippet += "...";
                spdlog::trace("[HTTP] >>> body: {}", snippet);
            }
            if (!res.body.empty()) {
                std::string snippet = res.body.substr(0, 512);
                if (res.body.size() > 512) snippet += "...";
                spdlog::trace("[HTTP] <<< body: {}", snippet);
            }
        }
    };
}

// ---------------------------------------------------------------------------
// make_timestamp_injector
// ---------------------------------------------------------------------------
PreRoutingHandler make_timestamp_injector()
{
    return [](const httplib::Request& req, httplib::Response& /*res*/)
        -> httplib::Server::HandlerResponse
    {
        int64_t now_us = static_cast<int64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        const_cast<httplib::Request&>(req).set_header(
            "X-Request-Start-Us", std::to_string(now_us));
        return httplib::Server::HandlerResponse::Unhandled;
    };
}

} // namespace truellm::middleware
