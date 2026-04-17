# ---------------------------------------------------------------------------
# third_party/vllm_kernels — vendored vLLM CUDA kernels (PyTorch-free)
#
# Only built when TRUELLM_HAS_CUDA is ON.
# Produces: truellm_vllm_kernels (static library)
# ---------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.21)

# Determine if we are building with sm_89+ (Ada / Hopper) in the arch list.
# If so, enable FP8 paths in the kernels.
set(_HAS_SM89 OFF)
foreach(_arch ${CMAKE_CUDA_ARCHITECTURES})
    if(_arch GREATER_EQUAL 89)
        set(_HAS_SM89 ON)
    endif()
endforeach()

if(_HAS_SM89)
    message(STATUS "[vllm_kernels] FP8 paths enabled (sm_89+ detected)")
else()
    message(STATUS "[vllm_kernels] FP8 paths disabled (no sm_89+ arch)")
endif()

add_library(truellm_vllm_kernels STATIC
    activation_kernels.cu
    layernorm_kernels.cu
    cache_kernels.cu
    paged_attention_v1.cu
    paged_attention_v2.cu
    fp8_gemm.cu
)

set_target_properties(truellm_vllm_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
    POSITION_INDEPENDENT_CODE   ON
)

target_include_directories(truellm_vllm_kernels
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/include          # <truellm/types.h>
)

target_compile_definitions(truellm_vllm_kernels
    PUBLIC TRUELLM_HAS_CUDA=1
)

if(_HAS_SM89)
    target_compile_definitions(truellm_vllm_kernels
        PUBLIC TRUELLM_CUDA_SM89_PLUS=1
    )
endif()

# NVCC flags: fast math, line info for profiling
target_compile_options(truellm_vllm_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -lineinfo
        -Xcompiler=-fPIC
    >
)

# Link CUDA runtime (cudart)
target_link_libraries(truellm_vllm_kernels
    PUBLIC  CUDA::cudart
)
