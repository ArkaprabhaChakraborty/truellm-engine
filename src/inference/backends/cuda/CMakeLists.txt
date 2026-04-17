# ---------------------------------------------------------------------------
# truellm_backend_cuda — CudaEngine with paged attention + vllm kernels
#
# Only built when TRUELLM_HAS_CUDA=ON and TRUELLM_BUILD_CUDA_ENGINE=ON.
# ---------------------------------------------------------------------------

add_library(truellm_backend_cuda STATIC
    cuda_engine.cpp
    cuda_model.cpp
    cuda_allocator.cpp
    cuda_stream.cpp
    multi_gpu.cpp
)

target_include_directories(truellm_backend_cuda
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../   # engine_interface.h
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../gpu  # gpu_device.h
    PRIVATE ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/src  # llama-model.h internal API
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
)

target_compile_definitions(truellm_backend_cuda
    PUBLIC TRUELLM_HAS_CUDA=1
)
