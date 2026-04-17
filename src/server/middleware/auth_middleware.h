#pragma once
// ---------------------------------------------------------------------------
// auth_middleware.h — API key validation middleware
//
// Provides check_api_key(), a stateless helper used by HttpServer::check_auth()
// and any route that needs independent authentication.
//
// Wire-up in HttpServer::check_auth():
//   return middleware::check_api_key(req, res, cfg_.server.api_key);
// ---------------------------------------------------------------------------

#ifndef CPPHTTPLIB_THREAD_POOL_COUNT
#define CPPHTTPLIB_THREAD_POOL_COUNT 8
#endif
#include <httplib.h>

#include <string>

namespace truellm::middleware {

// Validates the Authorization: Bearer <key> header against expected_key.
//
// - If expected_key is empty, authentication is disabled and the function
//   always returns true without writing to res.
// - If the header is missing or does not match, writes a JSON 401 response
//   and returns false.
// - On success returns true without modifying res.
bool check_api_key(const httplib::Request& req,
                   httplib::Response&      res,
                   const std::string&      expected_key);

} // namespace truellm::middleware
