# ---------------------------------------------------------------------------
# Sanitizers.cmake — ASan / TSan / UBSan toggles
#
# Usage: -DTRUELLM_SANITIZERS="address;undefined"
# ---------------------------------------------------------------------------

set(TRUELLM_SANITIZERS "" CACHE STRING
    "Semicolon-separated list of sanitizers: address, undefined, thread, memory")

if(TRUELLM_SANITIZERS)
    if(MSVC)
        message(WARNING "[truellm] Sanitizers not fully supported on MSVC — skipping")
        return()
    endif()

    # Build the -fsanitize= flag string
    string(REPLACE ";" "," SANITIZER_FLAG "${TRUELLM_SANITIZERS}")

    add_compile_options(-fsanitize=${SANITIZER_FLAG} -fno-omit-frame-pointer)
    add_link_options(-fsanitize=${SANITIZER_FLAG})

    message(STATUS "[truellm] Sanitizers enabled: ${TRUELLM_SANITIZERS}")
endif()
