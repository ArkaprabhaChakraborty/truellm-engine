// ---------------------------------------------------------------------------
// guardrail_policy.cpp
// ---------------------------------------------------------------------------

#include "guardrail_policy.h"

#include <algorithm>
#include <sstream>

namespace truellm::policy {

// ---------------------------------------------------------------------------
// constraints_for_preset
// ---------------------------------------------------------------------------
RequestConstraints constraints_for_preset(const std::string& preset)
{
    if (preset == "strict") {
        return {/*max_tokens=*/2048, /*max_ctx=*/4096, /*allow_overcommit=*/false};
    }
    if (preset == "balanced") {
        return {/*max_tokens=*/4096, /*max_ctx=*/16384, /*allow_overcommit=*/false};
    }
    if (preset == "relaxed") {
        return {/*max_tokens=*/16384, /*max_ctx=*/65536, /*allow_overcommit=*/true};
    }
    if (preset == "off") {
        return {/*max_tokens=*/0, /*max_ctx=*/0, /*allow_overcommit=*/true};
    }
    // "custom" or unknown — use balanced as a safe default
    return {/*max_tokens=*/4096, /*max_ctx=*/16384, /*allow_overcommit=*/false};
}

// ---------------------------------------------------------------------------
// check_request
// ---------------------------------------------------------------------------
bool check_request(const RequestConstraints& c,
                   int     requested_max_tokens,
                   int64_t requested_ctx_tokens,
                   std::string& out_violation)
{
    if (c.max_tokens > 0 && requested_max_tokens > c.max_tokens) {
        std::ostringstream oss;
        oss << "max_tokens " << requested_max_tokens
            << " exceeds guardrail limit " << c.max_tokens
            << " for this server configuration";
        out_violation = oss.str();
        return false;
    }

    if (c.max_ctx_tokens > 0 && requested_ctx_tokens > c.max_ctx_tokens) {
        std::ostringstream oss;
        oss << "context length " << requested_ctx_tokens
            << " exceeds guardrail limit " << c.max_ctx_tokens
            << " for this server configuration";
        out_violation = oss.str();
        return false;
    }

    return true;
}

// ---------------------------------------------------------------------------
// clamp_max_tokens
// ---------------------------------------------------------------------------
int clamp_max_tokens(const RequestConstraints& c, int requested_max_tokens)
{
    if (c.max_tokens <= 0) return requested_max_tokens;
    return std::min(requested_max_tokens, c.max_tokens);
}

} // namespace truellm::policy
