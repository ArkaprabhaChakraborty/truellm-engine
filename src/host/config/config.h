#pragma once

// ---------------------------------------------------------------------------
// config.h — TOML configuration loader
//
// Entry point:  truellm::config::load(path, profile_override)
//
// Load order:
//   1. Start with compiled-in defaults  (struct field initialisers)
//   2. Apply profile preset if requested (--profile or [server.toml] key)
//   3. Parse and overlay server.toml
//   4. Resolve ${ENV:VAR} tokens in string fields
// ---------------------------------------------------------------------------

#include <truellm/config_schema.h>
#include <string>

namespace truellm::config {

// Load the server.toml at `path` and return a fully-populated TrueLLMConfig.
//
// `profile_override` — if non-empty, apply this preset name BEFORE parsing
// the TOML file (so TOML values still win over the preset).  Pass an empty
// string to let the file's own "profile" key (or no key) decide.
//
// Throws std::runtime_error on TOML parse errors or validation failures.
TrueLLMConfig load(const std::string& path,
                   const std::string& profile_override = "");

// Validate a populated TrueLLMConfig.  Returns an empty string on success;
// returns a human-readable error message if any field is out of range.
std::string validate(const TrueLLMConfig& cfg);

} // namespace truellm::config
