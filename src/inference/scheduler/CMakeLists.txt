add_library(truellm_scheduler STATIC
    batch_scheduler.cpp
)

target_include_directories(truellm_scheduler
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/..   # src/inference/ — engine_interface.h, etc.
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}      # src/inference/scheduler/ — batch_scheduler.h
)

target_link_libraries(truellm_scheduler
    PUBLIC
        truellm_headers
    PRIVATE
        llama
        spdlog::spdlog
)

target_compile_features(truellm_scheduler PUBLIC cxx_std_17)
