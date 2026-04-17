# ---------------------------------------------------------------------------
# truellm_backend_cpu — llama.cpp wrapper (CPU + GPU via n_gpu_layers)
# ---------------------------------------------------------------------------

add_library(truellm_backend_cpu STATIC
    ggml_model.cpp
    ggml_context.cpp
    ggml_engine.cpp
)

target_include_directories(truellm_backend_cpu
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../   # engine_interface.h + gpu/gpu_device.h
)

target_link_libraries(truellm_backend_cpu
    PUBLIC  truellm_headers
            llama
    PRIVATE truellm_warnings
            truellm_gpu
            truellm_scheduler
            spdlog::spdlog
            fmt::fmt
)
