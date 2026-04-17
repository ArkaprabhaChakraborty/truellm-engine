# ---------------------------------------------------------------------------
# TrueLLMOptions.cmake — Global options and feature flags
# ---------------------------------------------------------------------------

# Backend selection: CPU (default) or CUDA
set(TRUELLM_BACKEND "CPU" CACHE STRING "Inference backend: CPU or CUDA")
set_property(CACHE TRUELLM_BACKEND PROPERTY STRINGS "CPU" "CUDA")

if(TRUELLM_BACKEND STREQUAL "CUDA")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(TRUELLM_HAS_CUDA ON)
    message(STATUS "[truellm] Backend: CUDA (${CUDAToolkit_VERSION})")
else()
    set(TRUELLM_HAS_CUDA OFF)
    message(STATUS "[truellm] Backend: CPU")
endif()

# Build toggles
option(TRUELLM_BUILD_TESTS      "Build unit + integration tests"                ON)
option(TRUELLM_BUILD_BENCHMARKS "Build benchmark targets"                        OFF)

# Plugin bridge (Part 2 — stubbed for now)
option(TRUELLM_PLUGIN_BRIDGE    "Build plugin bridge (Part 2)"                  OFF)

# CudaEngine — paged attention + vllm kernels (Part 2 GPU)
option(TRUELLM_BUILD_CUDA_ENGINE "Build CudaEngine with paged attention"         OFF)

# Auto-detect FP8 capability from CUDA arch list
if(TRUELLM_HAS_CUDA)
    set(_TRUELLM_HAS_SM89 OFF)
    foreach(_arch ${CMAKE_CUDA_ARCHITECTURES})
        if(_arch GREATER_EQUAL 89)
            set(_TRUELLM_HAS_SM89 ON)
        endif()
    endforeach()
    option(TRUELLM_CUDA_SM89_PLUS "Enable FP8 kernel paths (Ada/Hopper sm_89+)"
           ${_TRUELLM_HAS_SM89})
    if(TRUELLM_CUDA_SM89_PLUS)
        message(STATUS "[truellm] FP8 KV cache: ENABLED (sm_89+)")
    endif()
endif()

message(STATUS "[truellm] Version:    ${PROJECT_VERSION}")
message(STATUS "[truellm] Tests:      ${TRUELLM_BUILD_TESTS}")
message(STATUS "[truellm] Benchmarks: ${TRUELLM_BUILD_BENCHMARKS}")
