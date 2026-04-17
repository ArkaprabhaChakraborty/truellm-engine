# ---------------------------------------------------------------------------
# truellm_policy — Request policy enforcement (rate limiting, guardrails)
# ---------------------------------------------------------------------------

add_library(truellm_policy STATIC
    rate_limiter.cpp
    guardrail_policy.cpp
)

# Expose src/ so consumers can #include "policy/rate_limiter.h"
target_include_directories(truellm_policy
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(truellm_policy
    PUBLIC  truellm_headers
    PRIVATE truellm_warnings
            spdlog::spdlog
)
