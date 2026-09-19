# ---------------------------------------------------------------------------
# tests/ — GoogleTest unit tests + Google Benchmark suites
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------
add_executable(truellm_unit_tests
    unit/host/test_version.cpp
    unit/host/test_config.cpp
    unit/inference/test_engine_factory.cpp
    unit/inference/test_ggml_engine.cpp
    unit/inference/test_gpu_device.cpp
    unit/plugin_bridge/test_tool_registry.cpp
    unit/policy/test_rate_limiter.cpp
    unit/server/test_chat_format.cpp
)

target_include_directories(truellm_unit_tests
    PRIVATE ${CMAKE_SOURCE_DIR}/src/inference   # backends/cpu/ggml_engine.h
            ${CMAKE_SOURCE_DIR}/src             # policy/, plugin_bridge/, server/
)

target_link_libraries(truellm_unit_tests
    PRIVATE truellm_headers
            truellm_config
            truellm_inference
            truellm_backend_cpu
            truellm_gpu
            truellm_plugin_bridge
            truellm_policy
            truellm_server          # chat_format.* (common_chat wrapper)
            GTest::gtest
            GTest::gtest_main
            fmt::fmt
            spdlog::spdlog
)

# Mirror the per-source define from src/server so the chat_format test selects
# the common-backed assertions vs the fallback assertions.  Also hand the test
# a real reasoning template (DeepSeek-R1) so it can prove reasoning extraction
# rather than just exercising the chatml fallback.
if(TARGET common)
    set(_r1_tmpl "${CMAKE_SOURCE_DIR}/third_party/llama.cpp/models/templates/llama-cpp-deepseek-r1.jinja")
    set_source_files_properties(unit/server/test_chat_format.cpp PROPERTIES
        COMPILE_DEFINITIONS "TRUELLM_HAVE_COMMON_CHAT=1;TRUELLM_R1_TEMPLATE_PATH=\"${_r1_tmpl}\"")
endif()

# ---------------------------------------------------------------------------
# CUDA-only unit tests for the compression passes.  These exercise host-side
# logic (bounds checks, cache-mismatch guards, constructor clamps) without
# launching any CUDA kernels, but the headers they reference include
# <cuda_fp16.h> / <cuda_runtime.h>, so the test sources and the backend
# library can only be linked when the CUDA backend is being built.
# ---------------------------------------------------------------------------
if(TRUELLM_HAS_CUDA AND TARGET truellm_backend_cuda)
    target_sources(truellm_unit_tests PRIVATE
        unit/inference/cuda/test_compression_scoring_pass.cpp
        unit/inference/cuda/test_kivi_bounds.cpp
        unit/inference/cuda/test_kivi_roundtrip.cpp
        unit/inference/cuda/test_kv_compression_pass.cpp
        unit/inference/cuda/test_vllm_kernels.cpp
        unit/inference/cuda/test_vq_kernels.cpp
        unit/inference/cuda/test_codebook_init.cpp
    )
    target_link_libraries(truellm_unit_tests PRIVATE
        truellm_backend_cuda
        truellm_vllm_kernels
        CUDA::cudart
    )
endif()

# Auto-discover TEST() macros so CTest shows individual test names.
gtest_discover_tests(truellm_unit_tests)

# ---------------------------------------------------------------------------
# Benchmarks (only when parent enabled TRUELLM_BUILD_BENCHMARKS)
# ---------------------------------------------------------------------------
if(TRUELLM_BUILD_BENCHMARKS AND TARGET benchmark::benchmark)
    add_executable(truellm_benchmarks
        benchmarks/bench_throughput.cpp
    )
    target_include_directories(truellm_benchmarks
        PRIVATE ${CMAKE_SOURCE_DIR}/src/inference
    )
    target_link_libraries(truellm_benchmarks
        PRIVATE truellm_headers
                truellm_inference
                truellm_backend_cpu
                benchmark::benchmark
                benchmark::benchmark_main
                spdlog::spdlog
    )

    # ---------------------------------------------------------------------------
    # CUDA-specific benchmarks (megakernel vs multi-launch, sync overhead, etc.)
    # Only built when TRUELLM_HAS_CUDA is ON.
    # bench_smem_occupancy.cpp includes device-side headers so must be compiled
    # as CUDA (rename to .cu handled by set_source_files_properties).
    # ---------------------------------------------------------------------------
    if(TRUELLM_HAS_CUDA)
        set_source_files_properties(
            benchmarks/bench_smem_occupancy.cpp
            PROPERTIES LANGUAGE CUDA
        )
        add_executable(truellm_cuda_benchmarks
            benchmarks/bench_megakernel_vs_multilaunch.cpp
            benchmarks/bench_sync_overhead.cpp
            benchmarks/bench_smem_occupancy.cpp
            benchmarks/bench_power_efficiency.cpp
        )
        target_include_directories(truellm_cuda_benchmarks
            PRIVATE ${CMAKE_SOURCE_DIR}/src/inference
                    ${CMAKE_SOURCE_DIR}/third_party/custom_kernels/include
        )
        target_link_libraries(truellm_cuda_benchmarks
            PRIVATE truellm_headers
                    truellm_inference
                    truellm_backend_cuda
                    truellm_custom_kernels
                    benchmark::benchmark
                    benchmark::benchmark_main
                    spdlog::spdlog
                    CUDA::cudart
        )
        target_compile_definitions(truellm_cuda_benchmarks
            PRIVATE TRUELLM_HAS_CUDA=1
        )
        set_target_properties(truellm_cuda_benchmarks PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
        )
        if(TRUELLM_BENCH_NVML)
            target_compile_definitions(truellm_cuda_benchmarks
                PRIVATE TRUELLM_BENCH_NVML=1)
            target_link_libraries(truellm_cuda_benchmarks PRIVATE nvidia-ml)
        endif()
    endif()
endif()
