#pragma once
// ---------------------------------------------------------------------------
// logging_middleware.h — HTTP request/response logging middleware
//
// make_request_logger()
//   Returns an httplib::Logger compatible callable.
//   At debug level: logs method + path + remote addr + status + elapsed ms.
//   At trace level: also logs truncated request/response body snippets.
//
// make_timestamp_injector()
//   Returns a pre-routing handler that stamps X-Request-Start-Us on each
//   inbound request so that the logger can compute elapsed time.
//
// Wire-up in HttpServer::setup_routes():
//   svr_.set_logger(middleware::make_request_logger());
//   svr_.set_pre_routing_handler(middleware::make_timestamp_injector());
// ---------------------------------------------------------------------------

#ifndef CPPHTTPLIB_THREAD_POOL_COUNT
#define CPPHTTPLIB_THREAD_POOL_COUNT 8
#endif
#include <httplib.h>

#include <functional>

namespace truellm::middleware {

using RequestLogger = std::function<void(const httplib::Request&,
                                         const httplib::Response&)>;

using PreRoutingHandler =
    std::function<httplib::Server::HandlerResponse(const httplib::Request&,
                                                    httplib::Response&)>;

// Returns a callable suitable for httplib::Server::set_logger().
RequestLogger make_request_logger();

// Returns a callable suitable for httplib::Server::set_pre_routing_handler().
// Injects an X-Request-Start-Us header with the current steady clock
// timestamp so that make_request_logger() can compute request duration.
PreRoutingHandler make_timestamp_injector();

} // namespace truellm::middleware
