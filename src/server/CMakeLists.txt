# ---------------------------------------------------------------------------
# truellm_server — HTTP server + OpenAI/native router + middleware
# ---------------------------------------------------------------------------

add_library(truellm_server STATIC
    http_server.cpp
    openai_router.cpp
    native_router.cpp
    anthropic_router.cpp
    chat_format.cpp
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

# chat_format.cpp is the only TU that bridges to llama.cpp's common_chat_*
# layer.  It links `common` (chat templates / jinja / reasoning + tool parsers
# / json-schema-to-grammar).  `common` is only defined when LLAMA_BUILD_COMMON
# is ON (set in the root CMakeLists); guard so a no-llama build still configures.
# common's headers pull their own vendored nlohmann/json — that include dir is
# carried transitively via the `common` target, and both jsons are 3.12.0 so
# there is no ODR conflict with the vcpkg nlohmann used by the other TUs.
if(TARGET common)
    target_link_libraries(truellm_server PRIVATE common)
    set_source_files_properties(chat_format.cpp PROPERTIES
        COMPILE_DEFINITIONS TRUELLM_HAVE_COMMON_CHAT=1)
else()
    message(WARNING "[truellm] common (llama.cpp) target not found — "
                    "chat_format will compile as a no-op fallback")
endif()
