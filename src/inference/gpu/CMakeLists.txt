# ---------------------------------------------------------------------------
# src/inference/gpu/ — GPU device query wrapper
#
# In CPU builds: compiles to a stub library that returns 0/empty from all
# methods.  In CUDA builds: links ggml-cuda and exposes live device info.
# ---------------------------------------------------------------------------

add_library(truellm_gpu STATIC gpu_device.cpp)

target_include_directories(truellm_gpu
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}   # gpu_device.h
    PRIVATE ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml/include
)

target_link_libraries(truellm_gpu
    PUBLIC  truellm_headers
    PRIVATE truellm_warnings spdlog::spdlog
)

if(TRUELLM_HAS_CUDA)
    target_compile_definitions(truellm_gpu PUBLIC TRUELLM_HAS_CUDA=1)
    # ggml-cuda is built by the llama.cpp submodule when GGML_CUDA=ON
    target_link_libraries(truellm_gpu PRIVATE ggml-cuda)
endif()
