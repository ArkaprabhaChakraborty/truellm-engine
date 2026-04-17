# ---------------------------------------------------------------------------
# src/inference/ — Engine interface + factory + backends
# ---------------------------------------------------------------------------

# GPU device wrapper (stubs to zeros in CPU builds)
add_subdirectory(gpu)

# Continuous batching scheduler (vLLM-style multi-sequence decode)
add_subdirectory(scheduler)

# CPU backend is always built (handles GPU offload too via n_gpu_layers)
add_subdirectory(backends/cpu)

# CUDA backend — only when both TRUELLM_HAS_CUDA and TRUELLM_BUILD_CUDA_ENGINE are ON
if(TRUELLM_HAS_CUDA AND TRUELLM_BUILD_CUDA_ENGINE)
    add_subdirectory(backends/cuda)
    message(STATUS "[truellm] CudaEngine: ENABLED")
else()
    message(STATUS "[truellm] CudaEngine: DISABLED (set TRUELLM_BUILD_CUDA_ENGINE=ON to enable)")
endif()

add_library(truellm_inference STATIC
    engine_factory.cpp
)

target_include_directories(truellm_inference
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}          # engine_interface.h
)

target_link_libraries(truellm_inference
    PUBLIC  truellm_headers
            truellm_backend_cpu
            truellm_gpu
            truellm_scheduler
    PRIVATE truellm_warnings
            fmt::fmt
            spdlog::spdlog
)

if(TRUELLM_HAS_CUDA AND TRUELLM_BUILD_CUDA_ENGINE)
    target_link_libraries(truellm_inference PUBLIC truellm_backend_cuda)
endif()

# Pass CUDA flag as a compile definition so engine_factory.cpp
# can conditionally reference CUDA backends
if(TRUELLM_HAS_CUDA)
    target_compile_definitions(truellm_inference PRIVATE TRUELLM_HAS_CUDA=1)
endif()
