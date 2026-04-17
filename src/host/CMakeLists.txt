# ---------------------------------------------------------------------------
# src/host/ — Config library + Application orchestrator + main executable
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# truellm_config — TOML config parser, profile presets, secrets resolver
# ---------------------------------------------------------------------------
add_library(truellm_config STATIC
    config/config.cpp
    config/profile.cpp
    config/secrets.cpp
)

target_include_directories(truellm_config
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}   # exposes config/config.h etc. as "config/..."
)

target_link_libraries(truellm_config
    PUBLIC  truellm_headers
    PRIVATE truellm_warnings
            fmt::fmt
            spdlog::spdlog
            toml11::toml11
)

# ---------------------------------------------------------------------------
# truellm_app — Application orchestrator
# ---------------------------------------------------------------------------
add_library(truellm_app STATIC
    application.cpp
)

target_include_directories(truellm_app
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(truellm_app
    PUBLIC  truellm_headers
            truellm_inference
            truellm_server
    PRIVATE truellm_warnings
            fmt::fmt
            spdlog::spdlog
)

# ---------------------------------------------------------------------------
# truellm-server — main executable
# ---------------------------------------------------------------------------
add_executable(truellm-server
    main.cpp
)

target_link_libraries(truellm-server
    PRIVATE truellm_headers
            truellm_warnings
            truellm_config
            truellm_app
            truellm_inference
            truellm_server
            fmt::fmt
            spdlog::spdlog
            CLI11::CLI11
)

install(TARGETS truellm-server
    RUNTIME DESTINATION bin
)

truellm_codesign_target(truellm-server)
