# ---------------------------------------------------------------------------
# truellm_server — HTTP server + OpenAI/native router + middleware
# ---------------------------------------------------------------------------

add_library(truellm_server STATIC
    http_server.cpp
    openai_router.cpp
    native_router.cpp
    middleware/auth_middleware.cpp
    middleware/logging_middleware.cpp
)

target_include_directories(truellm_server
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(truellm_server
    PUBLIC  truellm_headers
            truellm_inference
            truellm_plugin_bridge
            truellm_policy
    PRIVATE truellm_warnings
            httplib::httplib
            nlohmann_json::nlohmann_json
            fmt::fmt
            spdlog::spdlog
)
