#pragma once

// ---------------------------------------------------------------------------
// profile.h — Named profile presets
//
// Presets provide reasonable defaults tuned for a target environment.
// CLI flag --profile <name> (or profile = "name" in server.toml) applies
// the preset before any other overrides, so TOML values still win.
//
// Available presets:
//   local-cpu   — single-user laptop, conservative memory usage
//   workstation — power-user desktop with a discrete GPU
//   server      — headless production daemon
// ---------------------------------------------------------------------------

#include <truellm/config_schema.h>
#include <string>

namespace truellm::config {

// Apply the named preset to `cfg`.  Fields the preset touches are
// overwritten; fields not mentioned by the preset are left unchanged.
// Unknown preset names are silently ignored (the caller should warn).
void apply_profile(TrueLLMConfig& cfg, const std::string& preset_name);

// Returns true if `name` is a known preset.
bool is_known_profile(const std::string& name);

} // namespace truellm::config
