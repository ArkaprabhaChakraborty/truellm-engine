# ---------------------------------------------------------------------------
# StaticAnalysis.cmake — Optional clang-tidy and cppcheck integration
#
# CMake options:
#   -DTRUELLM_ENABLE_CLANG_TIDY=ON   -- run clang-tidy on every compiled TU
#   -DTRUELLM_ENABLE_CPPCHECK=ON     -- run cppcheck on every compiled TU
#
# Both tools are skipped silently when the executable is not on PATH, so
# developer machines without the tools can still build normally.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# clang-tidy
# ---------------------------------------------------------------------------
option(TRUELLM_ENABLE_CLANG_TIDY "Run clang-tidy on all compiled sources" OFF)

if(TRUELLM_ENABLE_CLANG_TIDY)
    find_program(CLANG_TIDY_EXE
        NAMES clang-tidy clang-tidy-18 clang-tidy-17 clang-tidy-16
        DOC   "Path to clang-tidy executable"
    )
    if(CLANG_TIDY_EXE)
        # Checks: enable a useful subset; suppress noisy ones from vcpkg headers.
        # -header-filter limits diagnostics to project headers only.
        set(_CT_CHECKS
            "bugprone-*"
            "clang-analyzer-*"
            "cppcoreguidelines-avoid-goto"
            "cppcoreguidelines-init-variables"
            "cppcoreguidelines-no-malloc"
            "modernize-use-nullptr"
            "modernize-use-override"
            "modernize-use-emplace"
            "performance-*"
            "portability-*"
            "readability-braces-around-statements"
            "readability-const-return-type"
            "readability-misleading-indentation"
            "-bugprone-easily-swappable-parameters"
        )
        string(REPLACE ";" "," _CT_CHECKS_STR "${_CT_CHECKS}")

        set(CMAKE_CXX_CLANG_TIDY
            "${CLANG_TIDY_EXE}"
            "--checks=${_CT_CHECKS_STR}"
            "--header-filter=${CMAKE_SOURCE_DIR}/src/.*|${CMAKE_SOURCE_DIR}/include/.*|${CMAKE_SOURCE_DIR}/plugins/.*"
            "--warnings-as-errors="    # warn only, do not fail the build
            "--extra-arg=-Wno-unknown-warning-option"
        )
        message(STATUS "[truellm] clang-tidy: ${CLANG_TIDY_EXE}")
    else()
        message(WARNING "[truellm] TRUELLM_ENABLE_CLANG_TIDY=ON but clang-tidy not found")
    endif()
endif()

# ---------------------------------------------------------------------------
# cppcheck
# ---------------------------------------------------------------------------
option(TRUELLM_ENABLE_CPPCHECK "Run cppcheck on all compiled sources" OFF)

if(TRUELLM_ENABLE_CPPCHECK)
    find_program(CPPCHECK_EXE
        NAMES cppcheck
        DOC   "Path to cppcheck executable"
    )
    if(CPPCHECK_EXE)
        # --suppress=missingIncludeSystem   ignore system / vcpkg headers
        # --inline-suppr                    honour // cppcheck-suppress comments
        set(CMAKE_CXX_CPPCHECK
            "${CPPCHECK_EXE}"
            "--std=c++17"
            "--enable=warning,performance,portability,style"
            "--suppress=missingIncludeSystem"
            "--suppress=unmatchedSuppression"
            "--inline-suppr"
            "--quiet"
            "--error-exitcode=1"
        )
        message(STATUS "[truellm] cppcheck: ${CPPCHECK_EXE}")
    else()
        message(WARNING "[truellm] TRUELLM_ENABLE_CPPCHECK=ON but cppcheck not found")
    endif()
endif()
