#pragma once

// ---------------------------------------------------------------------------
// secrets.h — Environment variable substitution for config string values
//
// Replaces ${ENV:VAR_NAME} tokens in any string with the value of the named
// environment variable.  Missing variables resolve to an empty string; a
// warning is logged.
// ---------------------------------------------------------------------------

#include <string>

namespace truellm::config {

// Scan `value` for ${ENV:VAR_NAME} tokens and replace each one with the
// corresponding environment variable value.
// Returns the resulting string (may equal the input if no tokens are found).
std::string resolve_secrets(const std::string& value);

} // namespace truellm::config
