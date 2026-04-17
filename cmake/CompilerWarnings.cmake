# ---------------------------------------------------------------------------
# CompilerWarnings.cmake — Unified warning flags for MSVC / GCC / Clang
# ---------------------------------------------------------------------------

add_library(truellm_warnings INTERFACE)

if(MSVC)
    target_compile_options(truellm_warnings INTERFACE
        /W4
        /wd4100  # unreferenced formal parameter
        /wd4201  # nameless struct/union
        /wd4251  # dll-interface warnings on STL members
        /wd4275  # non-dll-interface base class
        /permissive-
        /utf-8   # source and execution charset UTF-8
        /Zc:preprocessor  # conformant preprocessor (required for __VA_OPT__ etc.)
    )
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
