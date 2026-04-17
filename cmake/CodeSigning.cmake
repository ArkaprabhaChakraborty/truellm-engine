# ---------------------------------------------------------------------------
# CodeSigning.cmake -- Optional Authenticode signing via signtool.exe
#
# Activate with:
#   cmake .. -DTRUELLM_CODESIGN=ON
#            -DTRUELLM_CODESIGN_CERT_SHA1=<thumbprint>   # preferred
#         or -DTRUELLM_CODESIGN_CERT_NAME="YourDevCert"  # fallback
#
# The SHA1 thumbprint is shown by:
#   Get-ChildItem Cert:\CurrentUser\My | Where Subject -like "*YourDevCert*"
#     | Select Thumbprint
#
# Only has effect on WIN32 builds.  On other platforms the option is parsed
# but all functions are no-ops so cross-platform CMakeLists stay unchanged.
# ---------------------------------------------------------------------------

option(TRUELLM_CODESIGN
    "Sign built binaries with signtool.exe (Windows only; requires a code-signing cert)"
    OFF
)

set(TRUELLM_CODESIGN_CERT_SHA1 ""
    CACHE STRING
    "SHA1 thumbprint of the code-signing certificate (preferred over CERT_NAME)"
)
set(TRUELLM_CODESIGN_CERT_NAME ""
    CACHE STRING
    "Certificate subject CN for signtool /n (used when CERT_SHA1 is not set)"
)
set(TRUELLM_CODESIGN_TIMESTAMP_URL "http://timestamp.digicert.com"
    CACHE STRING
    "RFC 3161 timestamp server URL added to every signature"
)

# ---------------------------------------------------------------------------
# Internal: locate signtool.exe
# Search glob over all installed Windows SDK versions before falling back to
# plain PATH lookup.  Globbing handles SDK version churn automatically.
# ---------------------------------------------------------------------------
if(WIN32 AND TRUELLM_CODESIGN)
    # Collect candidate directories from both Program Files locations
    file(GLOB _SIGNTOOL_CANDIDATES
        "C:/Program Files (x86)/Windows Kits/10/bin/*/x64/signtool.exe"
        "C:/Program Files/Windows Kits/10/bin/*/x64/signtool.exe"
        "C:/Program Files (x86)/Windows Kits/10/bin/x64/signtool.exe"
        "C:/Program Files/Windows Kits/10/bin/x64/signtool.exe"
    )

    # Sort descending so the newest SDK version wins
    list(SORT _SIGNTOOL_CANDIDATES ORDER DESCENDING)
    list(GET  _SIGNTOOL_CANDIDATES 0 _SIGNTOOL_FIRST)

    find_program(SIGNTOOL_EXE
        NAMES signtool signtool.exe
        HINTS "${_SIGNTOOL_FIRST}"   # newest SDK candidate first
        DOC   "Full path to signtool.exe"
    )

    if(SIGNTOOL_EXE)
        message(STATUS "[truellm] Code signing: ${SIGNTOOL_EXE}")
    else()
        message(WARNING
            "[truellm] TRUELLM_CODESIGN=ON but signtool.exe not found.\n"
            "  Install the Windows 10/11 SDK (included with Visual Studio)\n"
            "  or add the signtool directory to PATH.")
    endif()

    # Validate cert identity was supplied
    if(NOT TRUELLM_CODESIGN_CERT_SHA1 AND NOT TRUELLM_CODESIGN_CERT_NAME)
        message(WARNING
            "[truellm] TRUELLM_CODESIGN=ON but no certificate was specified.\n"
            "  Set TRUELLM_CODESIGN_CERT_SHA1=<thumbprint>  (preferred)\n"
            "  or TRUELLM_CODESIGN_CERT_NAME=<CN subject>.\n"
            "  Run scripts/windows/create-dev-cert.ps1 to generate a local cert.")
    endif()
endif()

# ---------------------------------------------------------------------------
# truellm_codesign_target(<target>)
#
# Attaches a POST_BUILD command that runs signtool on the output of <target>.
# Safe to call unconditionally -- returns immediately on non-Windows or when
# TRUELLM_CODESIGN is OFF.
# ---------------------------------------------------------------------------
function(truellm_codesign_target target)
    if(NOT WIN32 OR NOT TRUELLM_CODESIGN)
        return()
    endif()
    if(NOT SIGNTOOL_EXE)
        return()
    endif()
    if(NOT TRUELLM_CODESIGN_CERT_SHA1 AND NOT TRUELLM_CODESIGN_CERT_NAME)
        return()
    endif()

    # Build the signtool argument list.
    # /fd SHA256    -- file digest algorithm
    # /tr <url>    -- RFC 3161 timestamp authority
    # /td SHA256    -- timestamp digest algorithm
    set(_args
        sign
        /fd SHA256
        /tr "${TRUELLM_CODESIGN_TIMESTAMP_URL}"
        /td SHA256
    )

    if(TRUELLM_CODESIGN_CERT_SHA1)
        # Thumbprint match is unambiguous; preferred over name-based lookup.
        list(APPEND _args /sha1 "${TRUELLM_CODESIGN_CERT_SHA1}")
    else()
        list(APPEND _args /n "${TRUELLM_CODESIGN_CERT_NAME}")
    endif()

    add_custom_command(TARGET "${target}" POST_BUILD
        COMMAND "${SIGNTOOL_EXE}" ${_args} "$<TARGET_FILE:${target}>"
        COMMENT "[truellm] Signing $<TARGET_FILE_NAME:${target}>..."
        VERBATIM
    )
endfunction()
