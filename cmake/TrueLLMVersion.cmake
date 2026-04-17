# ---------------------------------------------------------------------------
# TrueLLMVersion.cmake — Git tag → version stamping
#
# Reads version from git describe, falls back to 0.1.0 if not in a git repo.
# Sets: TRUELLM_VERSION_MAJOR, TRUELLM_VERSION_MINOR, TRUELLM_VERSION_PATCH,
#       TRUELLM_VERSION_STRING, TRUELLM_GIT_HASH
# ---------------------------------------------------------------------------

find_package(Git QUIET)

set(TRUELLM_VERSION_MAJOR 0)
set(TRUELLM_VERSION_MINOR 1)
set(TRUELLM_VERSION_PATCH 0)
set(TRUELLM_GIT_HASH "unknown")

if(GIT_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags --always --dirty
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_DESCRIBE
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE GIT_RESULT
    )

    if(GIT_RESULT EQUAL 0)
        # Try to parse vMAJOR.MINOR.PATCH from the tag
        if(GIT_DESCRIBE MATCHES "^v?([0-9]+)\\.([0-9]+)\\.([0-9]+)")
            set(TRUELLM_VERSION_MAJOR ${CMAKE_MATCH_1})
            set(TRUELLM_VERSION_MINOR ${CMAKE_MATCH_2})
            set(TRUELLM_VERSION_PATCH ${CMAKE_MATCH_3})
        endif()
    endif()

    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE TRUELLM_GIT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
endif()

set(TRUELLM_VERSION_STRING
    "${TRUELLM_VERSION_MAJOR}.${TRUELLM_VERSION_MINOR}.${TRUELLM_VERSION_PATCH}")

message(STATUS "[truellm] Version: ${TRUELLM_VERSION_STRING} (${TRUELLM_GIT_HASH})")
