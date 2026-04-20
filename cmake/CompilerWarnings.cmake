# ---------------------------------------------------------------------------
# CompilerWarnings.cmake — Unified warning flags for MSVC / GCC / Clang
# ---------------------------------------------------------------------------

add_library(truellm_warnings INTERFACE)

if(MSVC)
    # NVCC does not accept bare MSVC /flag-style options; they must be wrapped
    # with -Xcompiler= when compiling CUDA translation units.
    foreach(_w /W4 /wd4100 /wd4201 /wd4251 /wd4275 /permissive- /utf-8 /Zc:preprocessor)
        target_compile_options(truellm_warnings INTERFACE
            $<$<COMPILE_LANGUAGE:CXX,C>:${_w}>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${_w}>
        )
    endforeach()
else()
    target_compile_options(truellm_warnings INTERFACE
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wnon-virtual-dtor
        -Wold-style-cast
        -Wcast-align
        -Wunused
        -Woverloaded-virtual
        -Wconversion
        -Wsign-conversion
        -Wnull-dereference
        -Wdouble-promotion
        -Wformat=2
        -Wimplicit-fallthrough
    )

    # GCC-specific
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(truellm_warnings INTERFACE
            -Wmisleading-indentation
            -Wduplicated-cond
            -Wduplicated-branches
            -Wlogical-op
            -Wuseless-cast
        )
    endif()
endif()
