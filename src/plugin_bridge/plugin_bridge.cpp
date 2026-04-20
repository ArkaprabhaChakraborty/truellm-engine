// ---------------------------------------------------------------------------
// plugin_bridge.cpp
// ---------------------------------------------------------------------------

#include "plugin_bridge.h"
#include "../inference/engine_interface.h"

#include <truellm/version.h>

#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <thread>

namespace fs = std::filesystem;

namespace truellm {

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// Global singleton pointer — used by captureless vtable trampoline functions.
// There is exactly one PluginBridge per server process.
//
// Stored as an atomic so that trampoline functions (called from plugin threads)
// and the destructor (called from the main thread) do not race.  Trampolines
// load with acquire; constructor stores with release; destructor CAS with acq_rel.
// ---------------------------------------------------------------------------
namespace {
std::atomic<PluginBridge*> g_bridge{nullptr};

// ── Vtable trampolines ────────────────────────────────────────────────────────
static void tramp_log_debug(const char* id, const char* msg)
    { spdlog::debug("[Plugin:{}] {}", id ? id : "?", msg ? msg : ""); }

static void tramp_log_info(const char* id, const char* msg)
    { spdlog::info ("[Plugin:{}] {}", id ? id : "?", msg ? msg : ""); }

static void tramp_log_warn(const char* id, const char* msg)
    { spdlog::warn ("[Plugin:{}] {}", id ? id : "?", msg ? msg : ""); }

static void tramp_log_error(const char* id, const char* msg)
    { spdlog::error("[Plugin:{}] {}", id ? id : "?", msg ? msg : ""); }

// get_config(plugin_id, key):
//   1. Tries "plugin_id.key" in [plugins.settings] (per-plugin override).
//   2. Falls back to bare "key".
//   Returns a pointer into the global string cache (stable for process lifetime).
static const char* tramp_get_config(const char* plugin_id, const char* key)
{
    auto* b = g_bridge.load(std::memory_order_acquire);
    if (!b || !key) return nullptr;
    return b->vtable_get_config(plugin_id ? plugin_id : "", key);
}

static const char* tramp_model_id()
{
    auto* b = g_bridge.load(std::memory_order_acquire);
    return b ? b->vtable_model_id()   : "";
}

static const char* tramp_model_arch()
{
    auto* b = g_bridge.load(std::memory_order_acquire);
    return b ? b->vtable_model_arch() : "";
}

static int64_t tramp_ctx_len()
{
    auto* b = g_bridge.load(std::memory_order_acquire);
    return b ? b->vtable_ctx_len()    : 0;
}

static int64_t tramp_vocab_size()
{
    auto* b = g_bridge.load(std::memory_order_acquire);
    return b ? b->vtable_vocab_size() : 0;
}

static const char* tramp_request_id(const truellm_context_t* ctx)
    { return ctx ? ctx->request_id.c_str() : ""; }

static int64_t tramp_token_budget(const truellm_context_t* ctx)
    { return ctx ? ctx->token_budget : 0; }

static int32_t tramp_max_tokens(const truellm_context_t* ctx)
    { return ctx ? ctx->max_tokens : 0; }

static float tramp_temperature(const truellm_context_t* ctx)
    { return ctx ? ctx->temperature : 0.7f; }

static void tramp_report_status(const truellm_context_t* ctx, const char* msg)
{
    if (ctx && ctx->status_sink && msg)
        ctx->status_sink(msg);
}

static void* tramp_alloc(size_t n)   { return ::operator new(n, std::nothrow); }
static void  tramp_dealloc(void* p)  { ::operator delete(p); }

// ── api_minor >= 1 trampolines ─────────────────────────────────────────────

static const int32_t* tramp_tokenize(const char* text, int32_t* out_count)
{
    auto* b = g_bridge.load(std::memory_order_acquire);
    if (!b) { if (out_count) *out_count = 0; return nullptr; }
    return b->vtable_tokenize(text, out_count);
}

static const char* tramp_detokenize(const int32_t* tokens, int32_t count)
{
    auto* b = g_bridge.load(std::memory_order_acquire);
    return b ? b->vtable_detokenize(tokens, count) : "";
}

static const char* tramp_get_engine_config(const char* section, const char* key)
{
    auto* b = g_bridge.load(std::memory_order_acquire);
    return b ? b->vtable_get_engine_config(section, key) : nullptr;
}

static const void* tramp_get_attention_weights(int32_t layer_idx,
                                               int64_t* out_size_bytes)
{
    auto* b = g_bridge.load(std::memory_order_acquire);
    if (!b) { if (out_size_bytes) *out_size_bytes = 0; return nullptr; }
    return b->vtable_get_attention_weights(layer_idx, out_size_bytes);
}

static const void* tramp_get_hidden_states(int32_t layer_idx,
                                           int64_t* out_size_bytes)
{
    auto* b = g_bridge.load(std::memory_order_acquire);
    if (!b) { if (out_size_bytes) *out_size_bytes = 0; return nullptr; }
    return b->vtable_get_hidden_states(layer_idx, out_size_bytes);
}

static const void* tramp_get_kv_cache_tensor(int32_t layer_idx, int32_t type,
                                             int64_t out_shape[4])
{
    auto* b = g_bridge.load(std::memory_order_acquire);
    if (!b) {
        if (out_shape) { out_shape[0]=out_shape[1]=out_shape[2]=out_shape[3]=0; }
        return nullptr;
    }
    return b->vtable_get_kv_cache_tensor(layer_idx, type, out_shape);
}

static truellm_error_t tramp_call_engine(
    const truellm_context_t* ctx,
    const char* messages_json,
    const char* gen_params_json,
    void (*on_token)(void*, const char*, int),
    void* cb_data,
    char** result_json_out)
{
    auto* b = g_bridge.load(std::memory_order_acquire);
    if (!b) return TRUELLM_ERR_INTERNAL;
    return b->vtable_call_engine(ctx, messages_json, gen_params_json,
                                 on_token, cb_data, result_json_out);
}

static truellm_error_t tramp_inject_tokens(
    truellm_context_t* ctx, int64_t pos, const char* tokens_json)
{
    auto* b = g_bridge.load(std::memory_order_acquire);
    return b ? b->vtable_inject_tokens(ctx, pos, tokens_json)
             : TRUELLM_ERR_INTERNAL;
}

static void tramp_set_context_metadata(truellm_context_t* ctx,
                                       const char* key, const char* val)
{
    auto* b = g_bridge.load(std::memory_order_acquire);
    if (b) b->vtable_set_context_metadata(ctx, key, val);
}

static const char* tramp_get_context_metadata(const truellm_context_t* ctx,
                                               const char* key)
{
    auto* b = g_bridge.load(std::memory_order_acquire);
    return b ? b->vtable_get_context_metadata(ctx, key) : nullptr;
}

static const char* tramp_get_server_capability(const truellm_context_t* ctx,
                                                const char* cap_key)
{
    auto* b = g_bridge.load(std::memory_order_acquire);
    return b ? b->vtable_get_server_capability(ctx, cap_key) : nullptr;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------
PluginBridge::PluginBridge(const TrueLLMConfig& cfg)
    : cfg_(cfg)
{
    g_bridge.store(this, std::memory_order_release);
    build_host_vtable();
}

PluginBridge::~PluginBridge()
{
    // Stop all sidecars first.  Mark shutting_down before stopping so exit
    // callbacks do not trigger restart logic.
    for (auto& sess : sidecar_sessions_) {
        sess.shutting_down = true;
        if (sess.health)   sess.health->stop();
        if (sess.launcher) sess.launcher->stop();
    }
    sidecar_sessions_.clear();

    // Destroy native providers in reverse load order, then unload libraries.
    std::lock_guard<std::mutex> lk(plugins_mu_);
    for (auto it = loaded_plugins_.rbegin(); it != loaded_plugins_.rend(); ++it) {
        auto& lp = *it;
        if (lp.tool_provider) {
            tool_registry_.unregister_plugin(lp.plugin_id);
            if (lp.tool_provider->destroy)
                lp.tool_provider->destroy(lp.tool_provider->state);
        } else if (!lp.tool_names.empty()) {
            // Remote plugin — clean up registry entries.
            tool_registry_.unregister_remote_plugin(lp.plugin_id);
        }
        if (lp.prep_provider && lp.prep_provider->destroy)
            lp.prep_provider->destroy(lp.prep_provider->state);
        if (lp.gen_provider  && lp.gen_provider->destroy)
            lp.gen_provider->destroy(lp.gen_provider->state);
        if (!lp.path.empty()) native_loader_.unload(lp.path);
    }
    loaded_plugins_.clear();

    PluginBridge* expected = this;
    g_bridge.compare_exchange_strong(expected, nullptr, std::memory_order_acq_rel);
}

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------
ErrorCode PluginBridge::init(EngineInterface* engine)
{
    engine_ = engine;

    // Cache model info strings so vtable trampolines can return stable c_str()s.
    if (engine_ && engine_->is_model_loaded()) {
        auto info         = engine_->get_model_info();
        model_cache_.model_id       = info.model_id;
        model_cache_.architecture   = info.architecture;
        model_cache_.context_length = info.context_length;
        model_cache_.vocab_size     = info.vocab_size;
    }

    if (!cfg_.plugins.enabled) {
        spdlog::info("[PluginBridge] plugins.enabled=false — no plugins loaded");
        return ErrorCode::Ok;
    }

    if (cfg_.plugins.plugin_dirs.empty()) {
        spdlog::info("[PluginBridge] no plugin_dirs configured");
        return ErrorCode::Ok;
    }

    for (const auto& dir : cfg_.plugins.plugin_dirs) {
        // Check if this dir is marked as Python or JS (by a sidecar hint file).
        // Fallback: scan for .py → Python sidecar; .ts / .js → JS sidecar.
        // Native .dll / .so → native_loader as before.
        bool has_py  = fs::exists(fs::path(dir) / "manifest.toml") &&
                       fs::exists(fs::path(dir) / "main.py");
        bool has_js  = fs::exists(fs::path(dir) / "manifest.toml") &&
                       (fs::exists(fs::path(dir) / "main.ts") ||
                        fs::exists(fs::path(dir) / "main.js"));

        if (has_py) {
            auto ec = start_sidecar(dir, SidecarRuntime::Python);
            if (ec != ErrorCode::Ok)
                spdlog::warn("[PluginBridge] Python sidecar failed for '{}'", dir);
        } else if (has_js) {
            auto ec = start_sidecar(dir, SidecarRuntime::JavaScript);
            if (ec != ErrorCode::Ok)
                spdlog::warn("[PluginBridge] JS sidecar failed for '{}'", dir);
        } else {
            auto ec = load_plugin_dir(dir);
            if (ec != ErrorCode::Ok)
                spdlog::warn("[PluginBridge] errors in plugin dir '{}' (continuing)", dir);
        }
    }

    spdlog::info("[PluginBridge] init done — {} native plugin(s) loaded, {} tool(s) registered "
                 "(sidecar tools arrive asynchronously via PluginAnnounce)",
                 [this]{ std::lock_guard<std::mutex> l(plugins_mu_); return loaded_plugins_.size(); }(),
                 tool_registry_.tool_count());
    return ErrorCode::Ok;
}

// ---------------------------------------------------------------------------
// Router API
// ---------------------------------------------------------------------------
ToolCallResult PluginBridge::dispatch_tool(const std::string& qualified_name,
                                            const std::string& args_json,
                                            truellm_context_t* ctx)
{
    return tool_registry_.dispatch(qualified_name, args_json, ctx);
}

PreprocessResult PluginBridge::run_preprocessors(const std::string& messages_json,
                                                  truellm_context_t* ctx)
{
    return prep_chain_.run(messages_json, ctx);
}

bool PluginBridge::generator_claims(const std::string& messages_json,
                                     truellm_context_t* ctx)
{
    return gen_registry_.claims(messages_json, ctx);
}

// ---------------------------------------------------------------------------
// get_status
// ---------------------------------------------------------------------------
nlohmann::json PluginBridge::get_status() const
{
    // Snapshot loaded_plugins_ under the mutex; release before building JSON.
    std::vector<LoadedPlugin> snapshot;
    {
        std::lock_guard<std::mutex> lk(plugins_mu_);
        snapshot = loaded_plugins_;
    }

    json plugins_arr = json::array();
    for (const auto& lp : snapshot) {
        json tools_arr = json::array();
        for (const auto& tname : lp.tool_names) {
            tools_arr.push_back({
                {"name",           tname},
                {"qualified_name", lp.plugin_id + "." + tname}
            });
        }

        json hooks_arr = json::array();
        // Native providers are tracked via pointers; remote plugins have tool_names
        // populated but null pointers — detect both paths.
        if (lp.tool_provider || !lp.tool_names.empty()) hooks_arr.push_back("tool_provider");
        if (lp.prep_provider)                            hooks_arr.push_back("preprocessor");
        if (lp.gen_provider)                             hooks_arr.push_back("generator");

        plugins_arr.push_back({
            {"id",      lp.plugin_id},
            {"version", lp.plugin_version},
            {"state",   lp.state},
            {"runtime", lp.runtime},
            {"hooks",   hooks_arr},
            {"tools",   tools_arr}
        });
    }

    return {
        {"plugin_count", static_cast<int>(snapshot.size())},
        {"tool_count",   static_cast<int>(tool_registry_.tool_count())},
        {"plugins",      plugins_arr}
    };
}

// ---------------------------------------------------------------------------
// Vtable trampoline getters
// ---------------------------------------------------------------------------
const char* PluginBridge::vtable_model_id()   const { return model_cache_.model_id.c_str(); }
const char* PluginBridge::vtable_model_arch() const { return model_cache_.architecture.c_str(); }
int64_t     PluginBridge::vtable_ctx_len()    const { return model_cache_.context_length; }
int64_t     PluginBridge::vtable_vocab_size() const { return model_cache_.vocab_size; }

const char* PluginBridge::vtable_get_config(const std::string& plugin_id,
                                             const std::string& key) const
{
    if (key.empty()) return nullptr;

    const auto& settings = cfg_.plugins.settings;

    // 1. Per-plugin key: "plugin_id.key"
    auto qualified = plugin_id.empty() ? key : (plugin_id + "." + key);
    auto it = settings.find(qualified);
    if (it == settings.end() && !plugin_id.empty()) {
        // 2. Global key fallback: bare "key"
        it = settings.find(key);
    }
    if (it == settings.end()) return nullptr;

    // Cache the c_str() so the pointer is stable across calls.
    std::lock_guard<std::mutex> lk(config_cache_mu_);
    auto [cache_it, _inserted] = config_cache_.emplace(qualified, it->second);
    return cache_it->second.c_str();
}

// ---------------------------------------------------------------------------
// api_minor >= 1 vtable helpers
// ---------------------------------------------------------------------------
const int32_t* PluginBridge::vtable_tokenize(const char* text, int32_t* out_count)
{
    if (!engine_ || !text) { if (out_count) *out_count = 0; return nullptr; }
    std::lock_guard<std::mutex> lk(tokenize_mu_);
    tokenize_buf_ = engine_->tokenize(text);
    if (out_count) *out_count = static_cast<int32_t>(tokenize_buf_.size());
    return tokenize_buf_.empty() ? nullptr : tokenize_buf_.data();
}

const char* PluginBridge::vtable_detokenize(const int32_t* tokens, int32_t count)
{
    if (!engine_ || !tokens || count <= 0) return "";
    std::lock_guard<std::mutex> lk(detokenize_mu_);
    std::vector<int32_t> tok_vec(tokens, tokens + count);
    detokenize_buf_ = engine_->detokenize(tok_vec);
    return detokenize_buf_.c_str();
}

const char* PluginBridge::vtable_get_engine_config(const char* section,
                                                    const char* key)
{
    // Exposes engine config values by dotted section.key path.
    // Uses the same stable-pointer config_cache_ as vtable_get_config().
    // Currently wired to the plugins.settings flat map via a "section.key" lookup;
    // Phase C5 will add a dedicated engine-config lookup table.
    if (!section || !key) return nullptr;
    std::string composite = std::string(section) + "." + key;
    std::lock_guard<std::mutex> lk(config_cache_mu_);
    auto it = config_cache_.find(composite);
    if (it != config_cache_.end()) return it->second.c_str();
    return nullptr;
}

const void* PluginBridge::vtable_get_attention_weights(int32_t layer_idx,
                                                        int64_t* out_size_bytes)
{
    if (!engine_) { if (out_size_bytes) *out_size_bytes = 0; return nullptr; }
    return engine_->get_attention_weights(layer_idx, out_size_bytes);
}

const void* PluginBridge::vtable_get_hidden_states(int32_t layer_idx,
                                                    int64_t* out_size_bytes)
{
    if (!engine_) { if (out_size_bytes) *out_size_bytes = 0; return nullptr; }
    return engine_->get_hidden_states(layer_idx, out_size_bytes);
}

const void* PluginBridge::vtable_get_kv_cache_tensor(int32_t layer_idx,
                                                      int32_t type,
                                                      int64_t out_shape[4])
{
    if (!engine_) {
        if (out_shape) { out_shape[0]=out_shape[1]=out_shape[2]=out_shape[3]=0; }
        return nullptr;
    }
    return engine_->get_kv_cache_tensor(layer_idx, type, out_shape);
}

truellm_error_t PluginBridge::vtable_call_engine(
    const truellm_context_t* ctx,
    const char* messages_json,
    const char* gen_params_json,
    void (*on_token)(void*, const char*, int),
    void* cb_data,
    char** result_json_out)
{
    if (!engine_ || !ctx) return TRUELLM_ERR_INTERNAL;
    if (!messages_json)   return TRUELLM_ERR_INVALID;

    // Enforce recursion depth limit.
    const int max_depth = cfg_.inference.rlm.max_hierarchy_depth;
    if (ctx->recursion_depth >= max_depth) {
        spdlog::warn("[PluginBridge] call_engine depth {} >= limit {}; aborting sub-call",
                     ctx->recursion_depth, max_depth);
        return TRUELLM_ERR_ABORT;
    }

    // Parse optional gen_params (temperature, max_tokens).
    float temperature = ctx->temperature;
    int32_t max_tokens = ctx->max_tokens;
    if (gen_params_json) {
        try {
            auto jp = json::parse(gen_params_json);
            if (jp.contains("temperature")) temperature = jp["temperature"].get<float>();
            if (jp.contains("max_tokens"))  max_tokens  = jp["max_tokens"].get<int32_t>();
        } catch (...) {}
    }

    // Build request — tokenise messages_json as a pre-formatted prompt.
    GenerateRequest req;
    req.temperature = temperature;
    req.max_tokens  = max_tokens;
    req.tokens      = engine_->tokenize(messages_json);
    if (on_token) {
        req.on_token = [on_token, cb_data](int32_t, const std::string& text) {
            on_token(cb_data, text.c_str(), 0);
        };
    }

    auto result = engine_->generate(req);
    if (result.error != ErrorCode::Ok)
        return TRUELLM_ERR_INTERNAL;

    if (on_token) on_token(cb_data, "", 1);  // signal completion

    if (result_json_out) {
        // Allocate engine-owned JSON result string.
        std::string out = "{\"text\":\"" + result.text + "\"}";
        char* buf = static_cast<char*>(tramp_alloc(out.size() + 1));
        if (buf) {
            std::copy(out.begin(), out.end(), buf);
            buf[out.size()] = '\0';
        }
        *result_json_out = buf;
    }
    return TRUELLM_OK;
}

truellm_error_t PluginBridge::vtable_inject_tokens(
    truellm_context_t* ctx, int64_t pos, const char* tokens_json)
{
    // Token injection modifies the engine's KV cache; stub for Phase C3.
    (void)ctx; (void)pos; (void)tokens_json;
    spdlog::debug("[PluginBridge] inject_tokens called (stub — Phase C3)");
    return TRUELLM_OK;
}

void PluginBridge::vtable_set_context_metadata(truellm_context_t* ctx,
                                                const char* key, const char* val)
{
    if (!ctx || !key) return;
    std::lock_guard<std::mutex> lk(ctx->metadata_mu);
    if (val) ctx->metadata[key] = val;
    else     ctx->metadata.erase(key);
}

const char* PluginBridge::vtable_get_context_metadata(
    const truellm_context_t* ctx, const char* key)
{
    if (!ctx || !key) return nullptr;
    std::lock_guard<std::mutex> lk(ctx->metadata_mu);
    auto it = ctx->metadata.find(key);
    if (it == ctx->metadata.end()) return nullptr;
    // Return pointer into the metadata map — stable for the request's lifetime.
    return it->second.c_str();
}

const char* PluginBridge::vtable_get_server_capability(
    const truellm_context_t* ctx, const char* cap_key)
{
    if (!cap_key) return nullptr;
    (void)ctx;
    std::lock_guard<std::mutex> lk(cap_buf_mu_);
    auto it = cap_cache_.find(cap_key);
    if (it != cap_cache_.end()) return it->second.c_str();

    // Compute capability value once and cache it.
    std::string val;
    const auto& inf = cfg_.inference;
    std::string key(cap_key);

    if      (key == TRUELLM_CAP_COMPRESSION)  val = inf.compression.enabled ? "1" : "0";
    else if (key == TRUELLM_CAP_RLM)          val = inf.rlm.enabled ? "1" : "0";
    else if (key == TRUELLM_CAP_RLM_MAX_DEPTH)val = std::to_string(inf.rlm.max_hierarchy_depth);
    else if (key == TRUELLM_CAP_MEGAKERNEL)   val = "0"; // Phase M4 sets this
    else if (key == TRUELLM_CAP_MEGAKERNEL_TILE) val = "8";
    else if (key == TRUELLM_CAP_ATTN_WEIGHTS) val = "0"; // Phase M4 sets this
    else if (key == TRUELLM_CAP_HIDDEN_STATES)val = "0"; // Phase M4 sets this
    else return nullptr;

    auto [ins_it, _ok] = cap_cache_.emplace(key, std::move(val));
    return ins_it->second.c_str();
}

// ---------------------------------------------------------------------------
// build_host_vtable
// ---------------------------------------------------------------------------
void PluginBridge::build_host_vtable()
{
    host_vtable_.api_version = TRUELLM_ABI_VERSION;
    host_vtable_.api_minor   = TRUELLM_ABI_MINOR;

    host_vtable_.log_debug  = tramp_log_debug;
    host_vtable_.log_info   = tramp_log_info;
    host_vtable_.log_warn   = tramp_log_warn;
    host_vtable_.log_error  = tramp_log_error;

    host_vtable_.get_config = tramp_get_config;

    host_vtable_.model_id          = tramp_model_id;
    host_vtable_.model_arch        = tramp_model_arch;
    host_vtable_.model_context_len = tramp_ctx_len;
    host_vtable_.model_vocab_size  = tramp_vocab_size;

    host_vtable_.request_id          = tramp_request_id;
    host_vtable_.token_budget        = tramp_token_budget;
    host_vtable_.request_max_tokens  = tramp_max_tokens;
    host_vtable_.request_temperature = tramp_temperature;

    host_vtable_.report_status = tramp_report_status;

    host_vtable_.alloc   = tramp_alloc;
    host_vtable_.dealloc = tramp_dealloc;

    // ── api_minor >= 1 ────────────────────────────────────────────────────────
    host_vtable_.tokenize               = tramp_tokenize;
    host_vtable_.detokenize             = tramp_detokenize;
    host_vtable_.get_engine_config      = tramp_get_engine_config;
    host_vtable_.get_attention_weights  = tramp_get_attention_weights;
    host_vtable_.get_hidden_states      = tramp_get_hidden_states;
    host_vtable_.get_kv_cache_tensor    = tramp_get_kv_cache_tensor;
    host_vtable_.call_engine            = tramp_call_engine;
    host_vtable_.inject_tokens          = tramp_inject_tokens;
    host_vtable_.set_context_metadata   = tramp_set_context_metadata;
    host_vtable_.get_context_metadata   = tramp_get_context_metadata;
    host_vtable_.get_server_capability  = tramp_get_server_capability;
}

// ---------------------------------------------------------------------------
// load_plugin_dir — scan a directory for .dll / .so files
// ---------------------------------------------------------------------------
ErrorCode PluginBridge::load_plugin_dir(const std::string& dir)
{
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) {
        spdlog::warn("[PluginBridge] plugin_dir '{}' is not a directory ({})",
                     dir, ec.message());
        return ErrorCode::IoError;
    }

#ifdef _WIN32
    constexpr const char* kExt = ".dll";
#else
    constexpr const char* kExt = ".so";
#endif

    bool any_error = false;
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        if (ec) break;
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != kExt) continue;

        auto rc = load_one_plugin(entry.path().string());
        // NotFound means the DLL has no truellm_plugin_init — it's a dependency
        // DLL (e.g. brotli, zlib) that happens to sit alongside the plugin binary.
        // This is expected and not an error.
        if (rc != ErrorCode::Ok && rc != ErrorCode::NotFound) any_error = true;
    }

    return any_error ? ErrorCode::PluginError : ErrorCode::Ok;
}

// ---------------------------------------------------------------------------
// load_one_plugin — load a single shared library
// ---------------------------------------------------------------------------
ErrorCode PluginBridge::load_one_plugin(const std::string& lib_path)
{
    truellm_tool_provider_t* tool = nullptr;
    truellm_preprocessor_t*  prep = nullptr;
    truellm_generator_t*     gen  = nullptr;

    auto rc = native_loader_.load(lib_path, &host_vtable_, &tool, &prep, &gen);
    if (rc != ErrorCode::Ok) return rc;

    LoadedPlugin lp;
    lp.path         = lib_path;
    lp.tool_provider = tool;
    lp.prep_provider = prep;
    lp.gen_provider  = gen;

    // Derive plugin identity from providers (prefer tool > prep > gen).
    const char* pid = nullptr;
    const char* pver = nullptr;
    if (tool) { pid = tool->plugin_id; pver = tool->plugin_version; }
    else if (prep) { pid = prep->plugin_id; pver = prep->plugin_version; }
    else if (gen)  { pid = gen->plugin_id;  pver = gen->plugin_version; }

    lp.plugin_id      = pid  ? pid  : fs::path(lib_path).stem().string();
    lp.plugin_version = pver ? pver : "unknown";

    // Register with subsystems and record tool names.
    if (tool) {
        tool_registry_.register_provider(tool);
        if (tool->get_tools) {
            const truellm_tool_def_t* const* tools = tool->get_tools(tool->state);
            for (int i = 0; tools && tools[i]; ++i) {
                if (tools[i]->name) lp.tool_names.push_back(tools[i]->name);
            }
        }
    }
    if (prep) prep_chain_.register_preprocessor(prep);
    if (gen)  gen_registry_.register_generator(gen);

    std::string pid_log, pver_log;
    {
        std::lock_guard<std::mutex> lk(plugins_mu_);
        loaded_plugins_.push_back(std::move(lp));
        pid_log  = loaded_plugins_.back().plugin_id;
        pver_log = loaded_plugins_.back().plugin_version;
    }
    spdlog::info("[PluginBridge] plugin '{}' v{} loaded from '{}'",
                 pid_log, pver_log, lib_path);
    return ErrorCode::Ok;
}

// ---------------------------------------------------------------------------
// start_sidecar — spawn Python or JS sidecar for a plugin directory
// ---------------------------------------------------------------------------
ErrorCode PluginBridge::start_sidecar(const std::string& plugin_dir,
                                       SidecarRuntime runtime)
{
    const char* rt_name = (runtime == SidecarRuntime::Python) ? "py" : "js";
    spdlog::info("[PluginBridge] starting {} sidecar for '{}'", rt_name, plugin_dir);

    SidecarSession sess;
    sess.runtime    = runtime;
    sess.plugin_dir = plugin_dir;

    // Create transport — the socket/pipe path includes the PID for uniqueness.
    sess.transport = std::make_unique<IpcTransport>(rt_name);

    // The sidecar will connect after we spawn it.  We start listening first.
    // Use a background thread so the main init thread isn't blocked waiting.
    std::string socket_path = sess.transport->socket_path();

    // Spawn sidecar.
    SidecarConfig sc_cfg;
    sc_cfg.runtime    = runtime;
    sc_cfg.socket_path = socket_path;
    sc_cfg.config_path = "server.toml";  // sidecar uses this path; engine passes it at startup
    sc_cfg.plugin_dir  = plugin_dir;

    if (runtime == SidecarRuntime::Python) {
        sc_cfg.module = "-m truellm_engine";
    } else {
        sc_cfg.module = "truellm-js-runtime/src/index.ts";
    }

    sess.launcher = std::make_unique<SidecarLauncher>();
    sess.client   = std::make_unique<IpcClient>(sess.transport.get());

    auto* sess_ptr = &sidecar_sessions_.emplace_back(std::move(sess));

    bool launched = sess_ptr->launcher->launch(sc_cfg,
        [this, sess_ptr](int code) { on_sidecar_exit(sess_ptr, code); });

    if (!launched) {
        sidecar_sessions_.pop_back();
        return ErrorCode::PluginError;
    }

    // Accept connection (blocks until sidecar connects or fails).
    if (!sess_ptr->transport->listen_and_accept()) {
        sess_ptr->launcher->stop();
        sidecar_sessions_.pop_back();
        return ErrorCode::PluginError;
    }

    // Wire announce callback before starting recv loop so no message is missed.
    sess_ptr->client->set_announce_callback(
        [this, sess_ptr](nlohmann::json announce) {
            on_sidecar_announce(*sess_ptr, announce);
        });

    // Start recv loop — dispatches incoming messages on a background thread.
    sess_ptr->client->start();

    // Start health monitor.
    sess_ptr->health = std::make_unique<HealthMonitor>(sess_ptr->client.get());
    sess_ptr->health->set_unhealthy_callback([this, sess_ptr]() {
        const char* rt = (sess_ptr->runtime == SidecarRuntime::Python) ? "py" : "js";
        spdlog::error("[PluginBridge] sidecar ({}) for '{}' is unhealthy — forcing restart",
                      rt, sess_ptr->plugin_dir);
        // Kill the sidecar so the exit callback fires and triggers restart with backoff.
        if (sess_ptr->launcher) sess_ptr->launcher->stop(/*timeout_ms=*/1000);
    });
    sess_ptr->health->start();
    sess_ptr->ready = true;

    spdlog::info("[PluginBridge] {} sidecar ready for '{}'", rt_name, plugin_dir);
    return ErrorCode::Ok;
}

// ---------------------------------------------------------------------------
// on_sidecar_announce — handle PluginAnnounce message from sidecar
// Called from the IpcClient recv thread — must be thread-safe.
// ---------------------------------------------------------------------------
void PluginBridge::on_sidecar_announce(SidecarSession& sess, const json& announce)
{
    if (!announce.contains("plugins") || !announce["plugins"].is_array()) {
        spdlog::warn("[PluginBridge] PluginAnnounce missing 'plugins' array");
        return;
    }

    const auto& plugins = announce["plugins"];
    spdlog::info("[PluginBridge] PluginAnnounce: {} plugin(s)", plugins.size());

    for (const auto& p : plugins) {
        std::string plugin_id      = p.value("plugin_id",      "");
        std::string plugin_version = p.value("plugin_version", "unknown");
        std::string runtime_str    = p.value("runtime",        "unknown");

        if (plugin_id.empty()) {
            spdlog::warn("[PluginBridge] PluginAnnounce: skipping plugin with empty id");
            continue;
        }

        LoadedPlugin lp;
        lp.plugin_id      = plugin_id;
        lp.plugin_version = plugin_version;
        lp.runtime        = runtime_str;
        lp.state          = "SERVING";
        // No C ABI provider pointers — dispatch goes through IPC.

        IpcClient* client_ptr = sess.client.get();

        const auto& tool_defs = p.contains("tool_defs") ? p["tool_defs"] : json::array();
        for (const auto& td : tool_defs) {
            std::string qname = td.value("qualified_name", "");
            if (qname.empty()) qname = td.value("name", "");
            if (qname.empty()) continue;

            // Derive bare tool name from "plugin_id.tool_name".
            std::string tool_name = qname;
            if (auto dot = qname.rfind('.'); dot != std::string::npos)
                tool_name = qname.substr(dot + 1);

            // Sidecars send structured fields, not a pre-serialised schema_json.
            // If schema_json is present use it directly; otherwise build from fields.
            std::string schema_str = td.value("schema_json", "");
            if (schema_str.empty()) {
                json params = json::object();
                if (td.contains("parameters")) {
                    if (td["parameters"].is_string()) {
                        // Some sidecars send parameters as a JSON string.
                        params = json::parse(td["parameters"].get<std::string>(),
                                             nullptr, /*exceptions=*/false);
                        if (params.is_discarded()) params = json::object();
                    } else {
                        params = td["parameters"];
                    }
                }
                json schema_obj = {
                    {"type", "function"},
                    {"function", {
                        {"name",        qname},
                        {"description", td.value("description", "")},
                        {"parameters",  params}
                    }}
                };
                schema_str = schema_obj.dump();
            }

            lp.tool_names.push_back(tool_name);

            tool_registry_.register_remote(
                plugin_id, tool_name, schema_str,
                [client_ptr, qname](const std::string& args_json,
                                    truellm_context_t* ctx) -> ToolCallResult {
                    IpcClient::ToolCallRequest req;
                    req.qualified_name = qname;
                    req.args_json      = args_json;
                    if (ctx) {
                        req.request_id   = ctx->request_id;
                        req.token_budget = ctx->token_budget;
                        req.max_tokens   = ctx->max_tokens;
                        req.temperature  = ctx->temperature;
                    }
                    auto resp = client_ptr->call_tool(req);
                    if (!resp.ok)
                        return {"", TRUELLM_ERR_INTERNAL, resp.error_msg};
                    return {resp.payload, TRUELLM_OK, ""};
                });
        }

        {
            std::lock_guard<std::mutex> lk(plugins_mu_);
            loaded_plugins_.push_back(std::move(lp));
        }
        spdlog::info("[PluginBridge] remote plugin '{}' v{} ({}) — {} tool(s)",
                     plugin_id, plugin_version, runtime_str,
                     loaded_plugins_.back().tool_names.size());
    }
}

// ---------------------------------------------------------------------------
// on_sidecar_exit — called on the sidecar launcher's monitor thread when the
//                   sidecar process exits.
// ---------------------------------------------------------------------------
void PluginBridge::on_sidecar_exit(SidecarSession* sess, int exit_code)
{
    const char* rt = (sess->runtime == SidecarRuntime::Python) ? "py" : "js";

    if (sess->shutting_down || exit_code == 0) {
        spdlog::info("[PluginBridge] sidecar ({}) for '{}' exited cleanly ({})",
                     rt, sess->plugin_dir, exit_code);
        return;
    }

    spdlog::warn("[PluginBridge] sidecar ({}) for '{}' crashed (exit={})",
                 rt, sess->plugin_dir, exit_code);

    if (!cfg_.plugins.restart_on_crash) return;

    if (sess->restart_count >= cfg_.plugins.max_restart_attempts) {
        spdlog::error("[PluginBridge] sidecar ({}) for '{}' exceeded max restarts ({}), "
                      "giving up",
                      rt, sess->plugin_dir, cfg_.plugins.max_restart_attempts);
        sess->ready = false;
        return;
    }

    // Exponential backoff: 1s, 2s, 4s, 8s ... capped at 30s.
    int delay_ms = (std::min)(1000 * (1 << sess->restart_count), 30000);
    ++sess->restart_count;

    spdlog::warn("[PluginBridge] sidecar ({}) for '{}' restarting in {}ms "
                 "(attempt {}/{})",
                 rt, sess->plugin_dir, delay_ms,
                 sess->restart_count, cfg_.plugins.max_restart_attempts);

    // Restart asynchronously so the exit callback thread is not blocked.
    std::thread([this, sess, delay_ms]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        restart_sidecar(sess);
    }).detach();
}

// ---------------------------------------------------------------------------
// restart_sidecar — tear down and re-launch a crashed sidecar session.
// Called from a detached restart thread.
// ---------------------------------------------------------------------------
void PluginBridge::restart_sidecar(SidecarSession* sess)
{
    const char* rt = (sess->runtime == SidecarRuntime::Python) ? "py" : "js";
    spdlog::info("[PluginBridge] restarting sidecar ({}) for '{}'",
                 rt, sess->plugin_dir);

    // 1. Stop health monitor and clean up old IPC objects.
    if (sess->health)   { sess->health->stop();   sess->health.reset(); }
    if (sess->client)   { sess->client.reset(); }
    if (sess->launcher) { sess->launcher->stop(); sess->launcher.reset(); }
    if (sess->transport){ sess->transport.reset(); }
    sess->ready = false;

    // 2. Remove old tools that were announced by this session.
    //    We identify them by scanning loaded_plugins_ for entries whose runtime
    //    matches and whose plugin_dir was set by this session.  Because
    //    plugin_dir is stored per session (not per LoadedPlugin), we track it
    //    via the session pointer.  For now we unregister all remote plugins
    //    from this runtime and let the re-announce re-register them.
    {
        std::lock_guard<std::mutex> lk(plugins_mu_);
        auto it = loaded_plugins_.begin();
        while (it != loaded_plugins_.end()) {
            if (it->runtime == (sess->runtime == SidecarRuntime::Python ? "python" : "js")
                && it->tool_provider == nullptr) {
                tool_registry_.unregister_remote_plugin(it->plugin_id);
                it = loaded_plugins_.erase(it);
            } else {
                ++it;
            }
        }
    }

    // 3. Re-run the full sidecar startup sequence.
    auto ec = start_sidecar(sess->plugin_dir, sess->runtime);
    if (ec != ErrorCode::Ok) {
        spdlog::error("[PluginBridge] sidecar ({}) restart failed for '{}'",
                      rt, sess->plugin_dir);
    } else {
        // Reset restart counter on successful startup.
        sess->restart_count = 0;
        spdlog::info("[PluginBridge] sidecar ({}) restarted successfully for '{}'",
                     rt, sess->plugin_dir);
    }
}

} // namespace truellm
