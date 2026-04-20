# ---------------------------------------------------------------------------
# truellm_backend_cuda — CudaEngine with paged attention + vllm kernels
#
# Only built when TRUELLM_HAS_CUDA=ON and TRUELLM_BUILD_CUDA_ENGINE=ON.
# ---------------------------------------------------------------------------

# MSVC + NVCC 13.x: Platform/Windows-NVIDIA-CUDA.cmake hardcodes
#   -Xcompiler=-Fd<TARGET_COMPILE_PDB>,-FS
# into CMAKE_CUDA_COMPILE_OBJECT (and the device-link templates) unconditionally.
# CMake's Ninja backend emits this as a separate response-file token; NVCC
# receives bare -FS and mis-parses it as its own -F S (force-include "S"),
# then rejects the .cu file as a second input:
#   "A single input file is required for a non-link phase when an output file
#    is specified"
# MSVC_DEBUG_INFORMATION_FORMAT=Embedded adds -Xcompiler=-Z7 but does NOT
# remove the -Fd,-FS suffix — hence why that property alone is not enough.
# Fix: strip the suffix from the rule templates before add_library.
if(MSVC)
    foreach(_tpl
            CMAKE_CUDA_COMPILE_OBJECT
            CMAKE_CUDA_DEVICE_LINK_LIBRARY
            CMAKE_CUDA_DEVICE_LINK_EXECUTABLE)
        string(REPLACE " -Xcompiler=-Fd<TARGET_COMPILE_PDB>,-FS" ""
               ${_tpl} "${${_tpl}}")
    endforeach()
endif()

add_library(truellm_backend_cuda STATIC
    cuda_engine.cpp
    cuda_model.cpp
    cuda_allocator.cpp
    cuda_stream.cpp
    multi_gpu.cpp
    # Compression passes — formerly plugin kv_compression_kernel + compression_preprocessor.
    # Compiled as engine-internal: direct device pointer access, no vtable overhead.
    compression/kv_compression_pass.cpp
    compression/compression_scoring_pass.cpp
    compression/vq_cache.cu
    compression/kivi_cache.cu
    compression/codebook_init.cu
)

set_source_files_properties(
    compression/vq_cache.cu
    compression/kivi_cache.cu
    compression/codebook_init.cu
    PROPERTIES LANGUAGE CUDA
)

target_include_directories(truellm_backend_cuda
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../   # engine_interface.h
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../gpu  # gpu_device.h
    PRIVATE ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/src  # llama-model.h internal API
            ${CMAKE_CURRENT_SOURCE_DIR}/compression
)

target_link_libraries(truellm_backend_cuda
    PUBLIC  truellm_headers
            llama
            truellm_vllm_kernels
            truellm_gpu
            CUDA::cudart
            CUDA::cublas
    PRIVATE truellm_warnings
            spdlog::spdlog
            fmt::fmt
            # Megakernel cooperative CUDA kernel (Phase M4).
            # truellm_custom_kernels is STATIC with CUDA_SEPARABLE_COMPILATION ON,
            # so its device symbols are available for cooperative launch.
            $<TARGET_NAME_IF_EXISTS:truellm_custom_kernels>
)

# Compression CUDA sources always require separable compilation.
set_target_properties(truellm_backend_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# MSVC + NVCC 13.x bug: CMake injects -Xcompiler=-Fd<pdb>,-FS for shared PDB
# tracking.  After splitting on the comma, NVCC interprets -FS as its own
# "-F S" (preinclude file "S"), then rejects the .cu file as a second input:
#   "A single input file is required for a non-link phase when an output file
#    is specified"
# Fix: CMP0141 NEW + Embedded switches /Zi → /Z7 (debug info embedded in .obj),
# which removes -Fd/-FS from compiler flags entirely.
if(MSVC)
    set_target_properties(truellm_backend_cuda PROPERTIES
        MSVC_DEBUG_INFORMATION_FORMAT Embedded)
endif()

# Custom kernels require CUDA separable linking when present.
if(TARGET truellm_custom_kernels)
    set_target_properties(truellm_backend_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_include_directories(truellm_backend_cuda PRIVATE
        ${CMAKE_SOURCE_DIR}/third_party/custom_kernels/include
    )
endif()

target_compile_definitions(truellm_backend_cuda
    PUBLIC TRUELLM_HAS_CUDA=1
)
