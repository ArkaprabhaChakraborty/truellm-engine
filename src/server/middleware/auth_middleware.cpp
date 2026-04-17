// ---------------------------------------------------------------------------
// auth_middleware.cpp
// ---------------------------------------------------------------------------

#include "auth_middleware.h"

#include <nlohmann/json.hpp>

namespace truellm::middleware {

bool check_api_key(const httplib::Request& req,
                   httplib::Response&      res,
                   const std::string&      expected_key)
{
    if (expected_key.empty()) return true;  // no auth configured

    const auto auth = req.get_header_value("Authorization");
    if (auth == "Bearer " + expected_key) return true;

    nlohmann::json err = {
        {"error", {
            {"message", "Unauthorized: missing or invalid API key"},
            {"type",    "authentication_error"}
        }}
    };
    res.status = 401;
    res.set_content(err.dump(), "application/json");
    return false;
}

} // namespace truellm::middleware
