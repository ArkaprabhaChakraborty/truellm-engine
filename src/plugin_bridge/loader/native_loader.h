#pragma once
// ---------------------------------------------------------------------------
// native_loader.h — Platform-portable shared library loader for native plugins
// ---------------------------------------------------------------------------

#include <truellm/plugin.h>
#include <truellm/host_api.h>
#include <truellm/types.h>

#include <string>
#include <unordered_map>

namespace truellm {

// ---------------------------------------------------------------------------
// NativeLoader
//
// Wraps dlopen/LoadLibrary, resolves truellm_plugin_init, performs ABI checks,
// and keeps handles alive until unload() is called.
// ---------------------------------------------------------------------------
class NativeLoader {
public:
    NativeLoader()  = default;
    ~NativeLoader();

    // Non-copyable / non-movable (handles are raw pointers).
    NativeLoader(const NativeLoader&)            = delete;
    NativeLoader& operator=(const NativeLoader&) = delete;

    // Load a shared library, call truellm_plugin_init, and return the providers.
    // Any of out_tool / out_prep / out_gen may be nullptr if the plugin does not
    // implement that hook type.
    // Returns ErrorCode::Ok on success; logs a diagnostic and returns the
    // appropriate error code on failure.
    ErrorCode load(const std::string&        lib_path,
                   const truellm_host_api_t* host,
                   truellm_tool_provider_t** out_tool,
                   truellm_preprocessor_t**  out_prep,
                   truellm_generator_t**     out_gen);

    // Unload a single library.  Caller must have called destroy() on all
    // providers obtained from this library before calling unload().
    void unload(const std::string& lib_path);

    // Unload all libraries (called from destructor).
    void unload_all();

private:
    struct LoadedLib {
        void*       handle;  // HMODULE (Windows) or dlopen handle (POSIX); stored as void*
        std::string path;
    };

    std::unordered_map<std::string, LoadedLib> libs_;
};

} // namespace truellm
