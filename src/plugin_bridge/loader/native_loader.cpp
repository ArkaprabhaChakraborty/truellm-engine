// ---------------------------------------------------------------------------
// native_loader.cpp — Platform-portable shared library loader
// ---------------------------------------------------------------------------

#include "native_loader.h"

#include <spdlog/spdlog.h>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

namespace truellm {

// Signature of the plugin entry point.
using PluginInitFn = truellm_error_t (*)(
    const truellm_host_api_t*,
    truellm_tool_provider_t**,
    truellm_preprocessor_t**,
    truellm_generator_t**);

// ---------------------------------------------------------------------------
// Platform helpers
// ---------------------------------------------------------------------------
static void* platform_open(const std::string& path)
{
#ifdef _WIN32
    // LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR requires a fully-qualified (absolute)
    // path — passing a relative path causes ERROR_INVALID_PARAMETER.
    // Resolve to absolute first, then load with the flag so that sibling DLLs
    // in the same directory (libssl, libcrypto, brotli…) are found when
    // resolving the loaded DLL's own imports.
    char abs[MAX_PATH] = {};
    if (GetFullPathNameA(path.c_str(), MAX_PATH, abs, nullptr) == 0)
        return nullptr;  // path resolution failed

    return static_cast<void*>(
        LoadLibraryExA(abs, nullptr,
                       LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                       LOAD_LIBRARY_SEARCH_DEFAULT_DIRS));
#else
    return dlopen(path.c_str(), RTLD_LOCAL | RTLD_NOW);
#endif
}

static void* platform_sym(void* handle, const char* name)
{
#ifdef _WIN32
    return reinterpret_cast<void*>(
        GetProcAddress(static_cast<HMODULE>(handle), name));
#else
    return dlsym(handle, name);
#endif
}

static void platform_close(void* handle)
{
#ifdef _WIN32
    FreeLibrary(static_cast<HMODULE>(handle));
#else
    dlclose(handle);
#endif
}

static std::string platform_error()
{
#ifdef _WIN32
    DWORD err = GetLastError();
    char buf[512] = {};
    FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                   nullptr, err, 0, buf, sizeof(buf) - 1, nullptr);
    return buf;
#else
    const char* e = dlerror();
    return e ? e : "(unknown)";
#endif
}

// ---------------------------------------------------------------------------
// NativeLoader::~NativeLoader
// ---------------------------------------------------------------------------
NativeLoader::~NativeLoader()
{
    unload_all();
}

// ---------------------------------------------------------------------------
// NativeLoader::load
// ---------------------------------------------------------------------------
ErrorCode NativeLoader::load(const std::string&        lib_path,
                              const truellm_host_api_t* host,
                              truellm_tool_provider_t** out_tool,
                              truellm_preprocessor_t**  out_prep,
                              truellm_generator_t**     out_gen)
{
    *out_tool = nullptr;
    *out_prep = nullptr;
    *out_gen  = nullptr;

    if (libs_.count(lib_path)) {
        spdlog::warn("[NativeLoader] '{}' is already loaded", lib_path);
        return ErrorCode::InvalidArgument;
    }

    // ── Step 1: open ──────────────────────────────────────────────────────────
    void* handle = platform_open(lib_path);
    if (!handle) {
        spdlog::error("[NativeLoader] Failed to open '{}': {}", lib_path, platform_error());
        return ErrorCode::IoError;
    }

    // ── Step 2: resolve symbol ────────────────────────────────────────────────
    void* raw_fn = platform_sym(handle, "truellm_plugin_init");
    if (!raw_fn) {
        // Not a TrueLLM plugin -- could be a dependency DLL (e.g. brotli, zlib)
        // that landed in the same directory.  Log at debug so the output is not
        // alarming; this is expected when httplib or other libraries copy their
        // runtime DLLs alongside the plugin binary.
        spdlog::debug("[NativeLoader] '{}' has no truellm_plugin_init -- skipping "
                      "(not a TrueLLM plugin)", lib_path);
        platform_close(handle);
        return ErrorCode::NotFound;
    }
    auto fn = reinterpret_cast<PluginInitFn>(raw_fn);

    // ── Step 3: call entry point ──────────────────────────────────────────────
    truellm_tool_provider_t* tool_tmp = nullptr;
    truellm_preprocessor_t*  prep_tmp = nullptr;
    truellm_generator_t*     gen_tmp  = nullptr;

    truellm_error_t rc = fn(host, &tool_tmp, &prep_tmp, &gen_tmp);
    if (rc != TRUELLM_OK) {
        spdlog::error("[NativeLoader] truellm_plugin_init returned error {} for '{}'",
                      static_cast<int>(rc), lib_path);
        platform_close(handle);
        return ErrorCode::PluginError;
    }

    // ── Step 4: ABI version check ─────────────────────────────────────────────
    auto check_abi = [&](uint32_t got, const char* which) -> bool {
        if (got != TRUELLM_ABI_VERSION) {
            spdlog::error("[NativeLoader] ABI mismatch in '{}' {} provider: "
                          "expected v{} got v{}", lib_path, which,
                          TRUELLM_ABI_VERSION, got);
            return false;
        }
        return true;
    };

    auto destroy_all = [&]() {
        if (tool_tmp && tool_tmp->destroy) tool_tmp->destroy(tool_tmp->state);
        if (prep_tmp && prep_tmp->destroy) prep_tmp->destroy(prep_tmp->state);
        if (gen_tmp  && gen_tmp->destroy)  gen_tmp->destroy(gen_tmp->state);
        platform_close(handle);
    };

    if (tool_tmp && !check_abi(tool_tmp->abi_version, "tool"))
        { destroy_all(); return ErrorCode::InvalidArgument; }
    if (prep_tmp && !check_abi(prep_tmp->abi_version, "preprocessor"))
        { destroy_all(); return ErrorCode::InvalidArgument; }
    if (gen_tmp  && !check_abi(gen_tmp->abi_version,  "generator"))
        { destroy_all(); return ErrorCode::InvalidArgument; }

    // ── Step 5: struct size guard (forward-compat) ────────────────────────────
    if (tool_tmp) {
        if (tool_tmp->struct_size > sizeof(truellm_tool_provider_t))
            spdlog::debug("[NativeLoader] '{}' built against newer SDK; extra fields ignored",
                          lib_path);
        else if (tool_tmp->struct_size < sizeof(truellm_tool_provider_t))
            spdlog::warn("[NativeLoader] '{}' built against older SDK minor ABI", lib_path);
    }

    // ── Step 6: register ──────────────────────────────────────────────────────
    libs_.emplace(lib_path, LoadedLib{handle, lib_path});

    *out_tool = tool_tmp;
    *out_prep = prep_tmp;
    *out_gen  = gen_tmp;

    spdlog::info("[NativeLoader] loaded '{}' (tool={} prep={} gen={})",
                 lib_path, tool_tmp != nullptr, prep_tmp != nullptr, gen_tmp != nullptr);
    return ErrorCode::Ok;
}

// ---------------------------------------------------------------------------
// NativeLoader::unload
// ---------------------------------------------------------------------------
void NativeLoader::unload(const std::string& lib_path)
{
    auto it = libs_.find(lib_path);
    if (it == libs_.end()) return;

    platform_close(it->second.handle);
    libs_.erase(it);
    spdlog::debug("[NativeLoader] unloaded '{}'", lib_path);
}

// ---------------------------------------------------------------------------
// NativeLoader::unload_all
// ---------------------------------------------------------------------------
void NativeLoader::unload_all()
{
    for (auto& [path, lib] : libs_) {
        platform_close(lib.handle);
    }
    libs_.clear();
}

} // namespace truellm
