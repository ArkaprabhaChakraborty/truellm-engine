#pragma once
// ---------------------------------------------------------------------------
// plugin_bridge.h — Facade hiding all plugin subsystems from the router.
// ---------------------------------------------------------------------------

#include <truellm/config_schema.h>
#include <truellm/host_api.h>
#include <truellm/types.h>

#include "hooks/tool_registry.h"
#include "hooks/preprocessor_chain.h"
#include "hooks/generator_registry.h"
#include "loader/native_loader.h"
#include "ipc/ipc_transport.h"
#include "ipc/ipc_client.h"
#include "sidecar_launcher.h"
#include "health_monitor.h"

#include <nlohmann/json.hpp>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace truellm {

class EngineInterface;

// ---------------------------------------------------------------------------
// PluginBridge
// ---------------------------------------------------------------------------
class PluginBridge {
public:
    explicit PluginBridge(const TrueLLMConfig& cfg);
    ~PluginBridge();

    PluginBridge(const PluginBridge&)            = delete;
    PluginBridge& operator=(const PluginBridge&) = delete;

    // Called once after engine->load_model().  Non-fatal: returns Ok even if
    // no plugins are found.
    ErrorCode init(EngineInterface* engine);

    // ── Router API ────────────────────────────────────────────────────────────
    bool        has_tools()         const { return tool_registry_.tool_count() > 0; }
    bool        has_preprocessors() const { return prep_chain_.has_preprocessors(); }
    bool        has_generators()    const { return gen_registry_.has_generators(); }

    std::string get_tool_schemas_json() const { return tool_registry_.get_schemas_json(); }

    ToolCallResult dispatch_tool(const std::string& qualified_name,
                                 const std::string& args_json,
                                 truellm_context_t* ctx);

    PreprocessResult run_preprocessors(const std::string& messages_json,
                                       truellm_context_t* ctx);

    bool generator_claims(const std::string& messages_json,
                          truellm_context_t* ctx);

    // ── Status endpoint ───────────────────────────────────────────────────────
    nlohmann::json get_status() const;

    // ── Vtable trampoline helpers ─────────────────────────────────────────────
    const char* vtable_model_id()   const;
    const char* vtable_model_arch() const;
    int64_t     vtable_ctx_len()    const;
    int64_t     vtable_vocab_size() const;
    const char* vtable_get_config(const std::string& plugin_id,
                                  const std::string& key) const;

    // api_minor >= 1 helpers
    const int32_t* vtable_tokenize(const char* text, int32_t* out_count);
    const char*    vtable_detokenize(const int32_t* tokens, int32_t count);
    const char*    vtable_get_engine_config(const char* section, const char* key);
    const void*    vtable_get_attention_weights(int32_t layer_idx,
                                                int64_t* out_size_bytes);
    const void*    vtable_get_hidden_states(int32_t layer_idx,
                                            int64_t* out_size_bytes);
    const void*    vtable_get_kv_cache_tensor(int32_t layer_idx, int32_t type,
                                              int64_t out_shape[4]);
    truellm_error_t vtable_call_engine(
        const truellm_context_t* ctx,
        const char* messages_json,
        const char* gen_params_json,
        void (*on_token)(void*, const char*, int),
        void* cb_data,
        char** result_json_out);
    truellm_error_t vtable_inject_tokens(
        truellm_context_t* ctx, int64_t pos, const char* tokens_json);
    void        vtable_set_context_metadata(truellm_context_t* ctx,
                                            const char* key, const char* val);
    const char* vtable_get_context_metadata(const truellm_context_t* ctx,
                                            const char* key);
    const char* vtable_get_server_capability(const truellm_context_t* ctx,
                                             const char* cap_key);

private:
    // ── Subsystems ────────────────────────────────────────────────────────────
    ToolRegistry      tool_registry_;
    PreprocessorChain prep_chain_;
    GeneratorRegistry gen_registry_;
    NativeLoader      native_loader_;

    // ── IPC / sidecar subsystems ──────────────────────────────────────────────
    struct SidecarSession {
        SidecarRuntime             runtime;
        std::string                plugin_dir;
        std::unique_ptr<IpcTransport>   transport;
        std::unique_ptr<IpcClient>      client;
        std::unique_ptr<SidecarLauncher> launcher;
        std::unique_ptr<HealthMonitor>  health;
        bool    ready          = false;
        int     restart_count  = 0;   // consecutive crash restarts
        bool    shutting_down  = false; // set during graceful teardown
    };
    std::vector<SidecarSession> sidecar_sessions_;
    mutable std::mutex          sidecar_sessions_mu_; // guards sidecar_sessions_

    // ── Engine reference ──────────────────────────────────────────────────────
    const TrueLLMConfig& cfg_;
    EngineInterface*     engine_ = nullptr;

    // ── Cached model info ─────────────────────────────────────────────────────
    struct ModelCache {
        std::string model_id;
        std::string architecture;
        int64_t     context_length = 0;
        int64_t     vocab_size     = 0;
    };
    ModelCache model_cache_;

    // ── Config value string cache ─────────────────────────────────────────────
    // vtable_get_config() returns const char* into this map.  Stable for process
    // lifetime because we never remove entries once inserted.
    mutable std::mutex                                    config_cache_mu_;
    mutable std::unordered_map<std::string, std::string> config_cache_;

    // ── api_minor >= 1 per-call scratch buffers ────────────────────────────────
    // Reused across calls; not thread-safe — plugins must serialise these calls
    // or accept that concurrent use may clobber the previous result.
    mutable std::mutex           tokenize_mu_;
    mutable std::vector<int32_t> tokenize_buf_;
    mutable std::mutex           detokenize_mu_;
    mutable std::string          detokenize_buf_;
    mutable std::mutex           cap_buf_mu_;
    mutable std::unordered_map<std::string, std::string> cap_cache_;

    // ── C vtable passed to native plugins ─────────────────────────────────────
    truellm_host_api_t host_vtable_{};

    // ── Loaded plugin records ─────────────────────────────────────────────────
    struct LoadedPlugin {
        std::string              path;
        std::string              plugin_id;
        std::string              plugin_version;
        std::string              runtime   = "cpp";
        std::string              state     = "SERVING";
        std::vector<std::string> tool_names;
        truellm_tool_provider_t* tool_provider = nullptr;
        truellm_preprocessor_t*  prep_provider = nullptr;
        truellm_generator_t*     gen_provider  = nullptr;
    };
    mutable std::mutex        plugins_mu_;   // guards loaded_plugins_ across threads
    std::vector<LoadedPlugin> loaded_plugins_;

    // ── Internals ─────────────────────────────────────────────────────────────
    void      build_host_vtable();
    ErrorCode load_plugin_dir(const std::string& dir);
    ErrorCode load_one_plugin(const std::string& lib_path);

    // Sidecar path
    ErrorCode start_sidecar(const std::string& plugin_dir, SidecarRuntime runtime);
    void      on_sidecar_announce(SidecarSession& sess, const nlohmann::json& announce);
    void      on_sidecar_exit(SidecarSession* sess, int exit_code);
    void      restart_sidecar(SidecarSession* sess);
};

} // namespace truellm
