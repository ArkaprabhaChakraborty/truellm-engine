# ---------------------------------------------------------------------------
# truellm_plugin_bridge — plugin loading, tool registry, and router integration
# ---------------------------------------------------------------------------

add_library(truellm_plugin_bridge STATIC
    plugin_bridge.cpp
    hooks/tool_registry.cpp
    hooks/preprocessor_chain.cpp
    hooks/generator_registry.cpp
    loader/native_loader.cpp
    ipc/ipc_transport.cpp
    ipc/ipc_client.cpp
    sidecar_launcher.cpp
    health_monitor.cpp
)

# Expose src/plugin_bridge/ so consumers can #include "plugin_bridge/plugin_bridge.h" etc.
# Also expose the project-root include/ so plugin.h / host_api.h are found.
target_include_directories(truellm_plugin_bridge
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/..  # src/ → allows "plugin_bridge/plugin_bridge.h"
    PUBLIC  ${CMAKE_SOURCE_DIR}/include     # include/ → allows <truellm/plugin.h>
)

target_link_libraries(truellm_plugin_bridge
    PUBLIC  truellm_headers
    PRIVATE truellm_warnings
            truellm_inference
            spdlog::spdlog
            nlohmann_json::nlohmann_json
            fmt::fmt
)

# dlopen / dlsym on POSIX; nothing extra needed on Windows (kernel32 is implicit).
if(UNIX)
    target_link_libraries(truellm_plugin_bridge PRIVATE dl)
endif()

# IPC transport: sockets are built into the kernel on both platforms.
# On Windows, Winsock2 is already linked transitively via cpp-httplib.
# No extra link target needed.
