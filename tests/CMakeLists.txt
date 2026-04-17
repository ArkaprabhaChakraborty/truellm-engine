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
)

target_include_directories(truellm_unit_tests
    PRIVATE ${CMAKE_SOURCE_DIR}/src/inference   # backends/cpu/ggml_engine.h
            ${CMAKE_SOURCE_DIR}/src             # policy/, plugin_bridge/
)

target_link_libraries(truellm_unit_tests
    PRIVATE truellm_headers
            truellm_config
            truellm_inference
            truellm_backend_cpu
            truellm_gpu
            truellm_plugin_bridge
            truellm_policy
            GTest::gtest
            GTest::gtest_main
            fmt::fmt
            spdlog::spdlog
)

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
endif()
