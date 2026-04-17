#pragma once
// ---------------------------------------------------------------------------
// guardrail_policy.h — Request-level constraints derived from hardware.guardrails.preset
//
// The guardrail preset limits how many tokens a request can generate and
// how much of the context window a single request may consume.  These limits
// protect the host from runaway generation or memory overcommit.
//
// Preset table:
//   strict   — conservative limits, suitable for shared production services
//   balanced — moderate limits; the default for single-user or workstation use
//   relaxed  — advisory only; limits are very high and allow_overcommit is true
//   off      — no limits enforced
//   custom   — limits from GuardrailsConfig.max_model_size_gb (not handled here)
// ---------------------------------------------------------------------------

#include <cstdint>
#include <string>

namespace truellm::policy {

struct RequestConstraints {
    int     max_tokens       = 0;   // 0 = no limit
    int64_t max_ctx_tokens   = 0;   // 0 = no limit
    bool    allow_overcommit = true;
};

// Returns the RequestConstraints for the named preset.
RequestConstraints constraints_for_preset(const std::string& preset);

// Validates a request against the given constraints.
// Returns true if the request is within limits.
// On false, `out_violation` is populated with a human-readable description.
bool check_request(const RequestConstraints& c,
                   int     requested_max_tokens,
                   int64_t requested_ctx_tokens,
                   std::string& out_violation);

// Clamps `requested_max_tokens` to the constraint limit.
// Returns the clamped value (unchanged if no limit or within limit).
int clamp_max_tokens(const RequestConstraints& c, int requested_max_tokens);

} // namespace truellm::policy
