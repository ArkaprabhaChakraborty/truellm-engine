// ---------------------------------------------------------------------------
// secrets.cpp — ${ENV:VAR_NAME} substitution for config string values
// ---------------------------------------------------------------------------

#include "secrets.h"

#include <spdlog/spdlog.h>

#include <cstdlib>
#include <string>

namespace truellm::config {

std::string resolve_secrets(const std::string& value)
{
    static constexpr std::string_view PREFIX = "${ENV:";
    static constexpr char              SUFFIX = '}';

    std::string result;
    result.reserve(value.size());

    std::size_t pos = 0;
    while (pos < value.size()) {
        auto start = value.find(PREFIX, pos);
        if (start == std::string::npos) {
            result.append(value, pos, std::string::npos);
            break;
        }

        // Append the literal text before the token
        result.append(value, pos, start - pos);

        auto name_start = start + PREFIX.size();
        auto end        = value.find(SUFFIX, name_start);
        if (end == std::string::npos) {
            // Malformed token — emit as-is and stop substituting
            spdlog::warn("[config::secrets] Malformed token (missing closing '}}') in: {}", value);
            result.append(value, start, std::string::npos);
            break;
        }

        std::string var_name = value.substr(name_start, end - name_start);
#ifdef _MSC_VER
        char*       env_buf  = nullptr;
        std::size_t env_len  = 0;
        _dupenv_s(&env_buf, &env_len, var_name.c_str());
        const char* env_val  = env_buf;
#else
        const char* env_val  = std::getenv(var_name.c_str());
#endif
        if (env_val == nullptr) {
            spdlog::warn("[config::secrets] Environment variable '{}' is not set", var_name);
        } else {
            result.append(env_val);
        }

#ifdef _MSC_VER
        std::free(env_buf);
#endif

        pos = end + 1; // skip past the '}'
    }

    return result;
}

} // namespace truellm::config
