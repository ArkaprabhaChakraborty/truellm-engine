cmake_minimum_required(VERSION 3.21)

# CMP0141: MSVC_DEBUG_INFORMATION_FORMAT target property support.
# Must be set before project() so CUDA language init picks up /Z7 (embedded
# debug info) instead of /Zi+/Fd, which NVCC mis-parses via -Xcompiler=-Fd,-FS.
if(POLICY CMP0141)
    cmake_policy(SET CMP0141 NEW)
endif()

# ---------------------------------------------------------------------------
# Version stamping (from git tag or fallback)
# ---------------------------------------------------------------------------
include(cmake/TrueLLMVersion.cmake)

project(truellm
    VERSION ${TRUELLM_VERSION_MAJOR}.${TRUELLM_VERSION_MINOR}.${TRUELLM_VERSION_PATCH}
    DESCRIPTION "TrueLLM — C++ LLM inference server"
    LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Output binaries into build/bin for easy discovery
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ---------------------------------------------------------------------------
# CMake modules
# ---------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(TrueLLMOptions)
include(CompilerWarnings)
include(Sanitizers)
include(StaticAnalysis)
include(CodeSigning)

# ---------------------------------------------------------------------------
# MSVC global definitions
# Suppress min/max macro pollution and trim the Windows header set.
# ---------------------------------------------------------------------------
if(MSVC)
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# ---------------------------------------------------------------------------
# vcpkg dependencies
# ---------------------------------------------------------------------------
find_package(fmt CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(toml11 CONFIG REQUIRED)
find_package(CLI11 CONFIG REQUIRED)
find_package(httplib CONFIG REQUIRED)

# ---------------------------------------------------------------------------
# Third-party: llama.cpp (git submodule)
# ---------------------------------------------------------------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/CMakeLists.txt")
    message(STATUS "[truellm] Building llama.cpp from submodule")

    # Suppress llama.cpp's own examples/tests/tools — we only want the library
    set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
    # common holds the chat-template (jinja) engine + per-model chat/tool/reasoning
    # parsers (common_chat_*) and json-schema-to-grammar.  TrueLLM wraps these in
    # src/server/chat_format.{h,cpp} for faithful prompt templating, reasoning
    # (thinking) streaming, and constrained output.  Light deps: links only
    # llama + cpp-httplib (already used) + Threads; llguidance stays OFF.
    set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
    set(LLAMA_LLGUIDANCE     OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)

    # Forward our CUDA option to llama.cpp's GGML
    if(TRUELLM_BACKEND STREQUAL "CUDA")
        set(GGML_CUDA ON CACHE BOOL "" FORCE)
    endif()

    add_subdirectory(third_party/llama.cpp)
else()
    message(STATUS "[truellm] llama.cpp submodule not found — building without inference backend")
    message(STATUS "[truellm]   Run: git submodule update --init --recursive")
endif()

# ---------------------------------------------------------------------------
# Third-party: vllm_kernels (PyTorch-free vendored CUDA kernels)
# Only built when TRUELLM_HAS_CUDA is ON.
# ---------------------------------------------------------------------------
if(TRUELLM_HAS_CUDA AND TRUELLM_BUILD_CUDA_ENGINE)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/vllm_kernels/CMakeLists.txt")
        add_subdirectory(third_party/vllm_kernels)
    else()
        message(WARNING "[truellm] third_party/vllm_kernels not found — CudaEngine disabled")
        set(TRUELLM_BUILD_CUDA_ENGINE OFF CACHE BOOL "" FORCE)
    endif()
endif()

# ---------------------------------------------------------------------------
# TrueLLM public headers (include/truellm/)
# ---------------------------------------------------------------------------
add_library(truellm_headers INTERFACE)
target_include_directories(truellm_headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# ---------------------------------------------------------------------------
# Source subdirectories
# ---------------------------------------------------------------------------
add_subdirectory(src/policy)
add_subdirectory(src/inference)
add_subdirectory(src/plugin_bridge)
add_subdirectory(src/server)
add_subdirectory(src/host)

# ---------------------------------------------------------------------------
# Native plugins are built from a separate repository: truellm-native-plugins.
# Point [plugins] plugin_dirs in your server config at the plugins build output.
# See: https://github.com/ArkaprabhaChakraborty/truellm-native-plugins
# ---------------------------------------------------------------------------
# Tests (GoogleTest)
# ---------------------------------------------------------------------------
if(TRUELLM_BUILD_TESTS)
    find_package(GTest CONFIG REQUIRED)
    include(CTest)
    include(GoogleTest)
    add_subdirectory(tests)
endif()

# ---------------------------------------------------------------------------
# Benchmarks (Google Benchmark, optional)
# ---------------------------------------------------------------------------
option(TRUELLM_BUILD_BENCHMARKS "Build Google Benchmark throughput benchmarks" OFF)
if(TRUELLM_BUILD_BENCHMARKS)
    find_package(benchmark CONFIG REQUIRED)
endif()
