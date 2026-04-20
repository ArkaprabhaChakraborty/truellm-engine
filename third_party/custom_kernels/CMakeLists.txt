cmake_minimum_required(VERSION 3.21)

# custom_kernels — megakernel and related cooperative-launch device code.
# Only built when TRUELLM_HAS_CUDA is ON.

if(NOT TRUELLM_HAS_CUDA)
    return()
endif()

enable_language(CUDA)

# ---------------------------------------------------------------------------
# Build target: truellm_custom_kernels (STATIC)
# ---------------------------------------------------------------------------
add_library(truellm_custom_kernels STATIC
    megakernel.cu
)

target_include_directories(truellm_custom_kernels PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../vllm_kernels/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
)

target_link_libraries(truellm_custom_kernels PUBLIC
    truellm_headers
)

# Cooperative kernels need separate compilation and linking.
set_target_properties(truellm_custom_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES          "70;80;86;89;90"
    POSITION_INDEPENDENT_CODE   ON
)

# Maximum register usage guard: prevent register spills that cause silent
# wrong output in cooperative kernels.
target_compile_options(truellm_custom_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        --generate-line-info
        -Xptxas=-v
        --maxrregcount=128
    >
)
