// ---------------------------------------------------------------------------
// plugin_bridge.cpp
// ---------------------------------------------------------------------------

#include "plugin_bridge.h"
#include "../inference/engine_interface.h"
#include "../server/chat_template.h"

#include <truellm/version.h>

#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

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

// ── api_minor >= 2: sandboxed per-plugin blob filesystem ─────────────────────
static const char* tramp_plugin_data_dir(const char* plugin_id)
{
    auto* b = g_bridge.load(std::memory_order_acquire);
    return b ? b->vtable_plugin_data_dir(plugin_id) : nullptr;
}

static truellm_error_t tramp_plugin_fs_write(const char* plugin_id,
                                              const char* relpath,
                                              const void* bytes, size_t n)
{
    auto* b = g_bridge.load(std::memory_order_acquire);
    return b ? b->vtable_plugin_fs_write(plugin_id, relpath, bytes, n)
             : TRUELLM_ERR_INTERNAL;
}

static truellm_error_t tramp_plugin_fs_read(const char* plugin_id,
                                             const char* relpath,
                                             void** out_bytes, size_t* out_n)
{
    auto* b = g_bridge.load(std::memory_order_acquire);
    return b ? b->vtable_plugin_fs_read(plugin_id, relpath, out_bytes, out_n)
             : TRUELLM_ERR_INTERNAL;
}

static truellm_error_t tramp_plugin_fs_unlink(const char* plugin_id,
                                               const char* relpath)
{
    auto* b = g_bridge.load(std::memory_order_acquire);
    return b ? b->vtable_plugin_fs_unlink(plugin_id, relpath)
             : TRUELLM_ERR_INTERNAL;
}

static truellm_error_t tramp_plugin_fs_listdir(const char* plugin_id,
                                                const char* relpath,
                                                char*** out_names,
                                                size_t* out_count)
{
    auto* b = g_bridge.load(std::memory_order_acquire);
    return b ? b->vtable_plugin_fs_listdir(plugin_id, relpath,
                                            out_names, out_count)
             : TRUELLM_ERR_INTERNAL;
}

static truellm_error_t tramp_call_tool(const truellm_context_t* ctx,
                                        const char* qualified_name,
                                        const char* args_json,
                                        char** result_json_out)
{
    auto* b = g_bridge.load(std::memory_order_acquire);
    return b ? b->vtable_call_tool(ctx, qualified_name, args_json,
                                    result_json_out)
             : TRUELLM_ERR_INTERNAL;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------
PluginBridge::PluginBridge(const TrueLLMConfig& cfg)
    : cfg_(cfg)
{
    // Resolve the per-plugin data-dir sandbox root.  Empty string disables
    // the host fs API (vtable returns NULL / TRUELLM_ERR_PERMISSION).
    if (!cfg_.plugins.data_dir.empty()) {
        try {
            plugin_data_root_ =
                fs::weakly_canonical(fs::path(cfg_.plugins.data_dir)).string();
        } catch (...) {
            plugin_data_root_ = cfg_.plugins.data_dir;
        }
    } else {
        try {
            plugin_data_root_ =
                fs::weakly_canonical(fs::current_path() / "build" / "plugin-data")
                    .string();
        } catch (...) {
            plugin_data_root_ = (fs::current_path() / "build" / "plugin-data").string();
        }
    }
    spdlog::info("[PluginBridge] plugin data dir: {}", plugin_data_root_);

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

GeneratorDispatchOutcome PluginBridge::try_dispatch_generator(
    const std::string&   messages_json,
    truellm_context_t*   ctx,
    void (*on_token)(void*, const char*, int),
    void*                cb_data)
{
    {
        std::lock_guard<std::mutex> lk(counters_mu_);
        ++counters_.generator_dispatches_attempted;
    }
    auto outcome = gen_registry_.dispatch(messages_json, ctx, on_token, cb_data);
    {
        std::lock_guard<std::mutex> lk(counters_mu_);
        if (outcome.claimed) {
            ++counters_.generator_dispatches_claimed;
            if (!outcome.plugin_id.empty())
                ++counters_.claims_per_plugin[outcome.plugin_id];
        }
        if (outcome.claimed && !outcome.ok)
            ++counters_.generator_dispatches_failed;
    }
    return outcome;
}

PluginBridge::RuntimeCounters PluginBridge::get_runtime_counters() const
{
    std::lock_guard<std::mutex> lk(counters_mu_);
    return counters_;
}

// ---------------------------------------------------------------------------
// record_chat_outcome / get_chat_outcomes  (§4.4 /truellm/v1/chat/status)
// ---------------------------------------------------------------------------
//
// Called by openai_router exactly once per top-level /v1/chat/completions
// request, after the response is built but before ctx is destroyed.  We
// read a handful of well-known keys from ctx.metadata; everything else
// stays in the per-request bag.
void PluginBridge::record_chat_outcome(const truellm_context_t* ctx)
{
    if (!ctx) return;

    // Snapshot the keys we care about under ctx's own metadata mutex.
    std::string phase, strategy, sid, classification, conscutive_fail;
    {
        std::lock_guard<std::mutex> lk(ctx->metadata_mu);
        auto pull = [&](const char* k) -> std::string {
            auto it = ctx->metadata.find(k);
            return (it == ctx->metadata.end()) ? std::string{} : it->second;
        };
        phase          = pull("chat.compact_phase");
        strategy       = pull("chat.compact_strategy");
        sid            = pull("session.id");
        classification = pull("safety.classification");
        conscutive_fail= pull("chat.consecutive_compact_failures");
    }

    std::lock_guard<std::mutex> lk(chat_outcomes_mu_);
    ++chat_outcomes_.total_requests;
    if (!sid.empty()) ++chat_outcomes_.with_session_id;

    if (strategy.empty()) {
        ++chat_outcomes_.auto_compactor_skipped;
    } else {
        ++chat_outcomes_.strategy_counts[strategy];
    }
    if (!phase.empty())          ++chat_outcomes_.phase_counts[phase];
    if (!classification.empty()) ++chat_outcomes_.classification_counts[classification];

    if (!conscutive_fail.empty()) {
        try {
            uint64_t n = static_cast<uint64_t>(std::stoull(conscutive_fail));
            if (n > chat_outcomes_.consecutive_compact_failures_max)
                chat_outcomes_.consecutive_compact_failures_max = n;
        } catch (...) { /* ignore malformed value */ }
    }
}

PluginBridge::ChatOutcomes PluginBridge::get_chat_outcomes() const
{
    std::lock_guard<std::mutex> lk(chat_outcomes_mu_);
    return chat_outcomes_;
}

// ---------------------------------------------------------------------------
// record_research_outcome / get_research_outcomes  (§9.8)
// ---------------------------------------------------------------------------
//
// Called by openai_router at request completion when the request was a
// research-mode request (ctx.metadata["mode"] == "research").  Reads the
// research_orchestrator's terminal-reason field plus the sub-call usage
// counters that vtable_call_engine accumulates and aggregates them so
// operators can answer "how many research turns did effort=high cost?".
void PluginBridge::record_research_outcome(const truellm_context_t* ctx)
{
    if (!ctx) return;

    std::string mode, effort, terminal;
    int64_t sub_prompt = 0, sub_completion = 0, sub_count = 0;
    {
        std::lock_guard<std::mutex> lk(ctx->metadata_mu);
        auto pull = [&](const char* k) -> std::string {
            auto it = ctx->metadata.find(k);
            return (it == ctx->metadata.end()) ? std::string{} : it->second;
        };
        mode     = pull("mode");
        effort   = pull("effort");
        terminal = pull("research.terminal_reason");
        auto pulli = [&](const char* k) -> int64_t {
            auto it = ctx->metadata.find(k);
            if (it == ctx->metadata.end()) return 0;
            try { return std::stoll(it->second); } catch (...) { return 0; }
        };
        sub_prompt     = pulli("research.subcall_prompt_tokens");
        sub_completion = pulli("research.subcall_completion_tokens");
        sub_count      = pulli("research.subcall_count");
    }
    if (mode != "research") return;

    std::lock_guard<std::mutex> lk(research_outcomes_mu_);
    ++research_outcomes_.total_requests;
    research_outcomes_.total_subcalls          += static_cast<uint64_t>(sub_count);
    research_outcomes_.total_prompt_tokens     += static_cast<uint64_t>(sub_prompt);
    research_outcomes_.total_completion_tokens += static_cast<uint64_t>(sub_completion);
    if (!effort.empty())   ++research_outcomes_.effort_counts[effort];
    if (!terminal.empty()) ++research_outcomes_.terminal_reason_counts[terminal];
}

PluginBridge::ResearchOutcomes PluginBridge::get_research_outcomes() const
{
    std::lock_guard<std::mutex> lk(research_outcomes_mu_);
    return research_outcomes_;
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

    // ── Enforce recursion depth limit ─────────────────────────────────────────
    // Architectural invariant #13 / Section 19.8: the would-be sub-call depth
    // is parent_depth + 1.  We check that value against the configured
    // hierarchy limit so a genuinely recursive sub-call (a sub-call that
    // itself invokes call_engine) fails fast even if parent_depth is 0.
    const int max_depth = cfg_.inference.rlm.max_hierarchy_depth;
    const int sub_depth = ctx->recursion_depth + 1;
    {
        std::lock_guard<std::mutex> lk(counters_mu_);
        ++counters_.call_engine_subcalls;
        if (static_cast<uint64_t>(sub_depth) > counters_.max_recursion_depth_reached)
            counters_.max_recursion_depth_reached = static_cast<uint64_t>(sub_depth);
    }
    if (sub_depth > max_depth) {
        {
            std::lock_guard<std::mutex> lk(counters_mu_);
            ++counters_.call_engine_depth_aborts;
        }
        spdlog::warn("[PluginBridge] call_engine sub_depth={} > limit={}; "
                     "aborting sub-call (parent_depth={})",
                     sub_depth, max_depth, ctx->recursion_depth);
        return TRUELLM_ERR_ABORT;
    }

    // Parse optional gen_params (temperature, max_tokens, run_preprocessors,
    // session_id); fall back to the parent context's values when a field is
    // absent or the JSON is malformed.
    float       temperature       = ctx->temperature;
    int32_t     max_tokens        = ctx->max_tokens;
    bool        run_preprocessors = false;
    std::string sub_session_id;            // §4.2 KV-reuse hint (plumbing only)
    if (gen_params_json) {
        try {
            auto jp = json::parse(gen_params_json);
            if (jp.contains("temperature"))       temperature       = jp["temperature"].get<float>();
            if (jp.contains("max_tokens"))        max_tokens        = jp["max_tokens"].get<int32_t>();
            if (jp.contains("run_preprocessors")) run_preprocessors = jp["run_preprocessors"].get<bool>();
            if (jp.contains("session_id")
                && jp["session_id"].is_string())  sub_session_id    = jp["session_id"].get<std::string>();
        } catch (...) {}
    }

    // ── Optional preprocessor re-entry (§3.3.2 + §11.2.1 follow-up) ──────────
    // Plugins like fork_agent_pool flip agent.kind on the ctx before
    // sub-calling; if they pass run_preprocessors=true, replay the chain
    // here so plan_mode_guard re-applies the allow-list, the snip_compactor
    // re-trims, etc.  Preprocessors that should NOT re-fire on sub-calls
    // gate on ctx->recursion_depth themselves (auto_compactor already does
    // this).  PREPROC_RESPOND short-circuits the sub-call; PREPROC_ABORT
    // surfaces as TRUELLM_ERR_INVALID.
    std::string effective_messages = messages_json;
    if (run_preprocessors && prep_chain_.has_preprocessors()) {
        auto pr = this->run_preprocessors(effective_messages,
                                           const_cast<truellm_context_t*>(ctx));
        if (pr.decision == TRUELLM_PREPROC_ABORT) {
            spdlog::warn("[PluginBridge] call_engine sub-call aborted by "
                         "preprocessor chain");
            return TRUELLM_ERR_INVALID;
        }
        if (pr.decision == TRUELLM_PREPROC_RESPOND) {
            // The preprocessor produced a synthetic response — short-circuit.
            if (on_token) {
                on_token(cb_data, pr.response_text.c_str(), 0);
                on_token(cb_data, "", 1);
            }
            if (result_json_out) {
                const std::string out = json{
                    {"text",          pr.response_text},
                    {"finish_reason", "preprocessor_respond"}
                }.dump();
                char* buf = static_cast<char*>(tramp_alloc(out.size() + 1));
                if (buf) {
                    std::copy(out.begin(), out.end(), buf);
                    buf[out.size()] = '\0';
                }
                *result_json_out = buf;
            }
            return TRUELLM_OK;
        }
        if (!pr.modified_messages_json.empty()) {
            effective_messages = pr.modified_messages_json;
        }
    }

    // §4.2 — `session_id` is currently a forward-compatible hint.  The CUDA
    // scheduler does not yet key its prefix-reuse path on it, so we just log
    // and write it into ctx.metadata under "call_engine.sub_session_id" so
    // a future scheduler change can pick it up without further ABI work.
    if (!sub_session_id.empty()) {
        std::lock_guard<std::mutex> lk(ctx->metadata_mu);
        const_cast<truellm_context_t*>(ctx)->metadata
            ["call_engine.sub_session_id"] = sub_session_id;
    }

    // Build request.  messages_json comes in as a JSON array of chat
    // messages.  We MUST run it through the chat-template formatter before
    // tokenising; passing the raw JSON literal to the tokenizer makes the
    // model see characters like `{"role":"user","content":"..."}` instead
    // of a real prompt and the resulting completion is garbage.  Falling
    // back to the raw bytes only happens when the JSON does not parse as
    // an array (e.g. a caller that passes a pre-formatted prompt string).
    const std::string tmpl = resolve_template(
        cfg_.model.chat_template,
        engine_ ? engine_->get_model_info().architecture : "");
    std::string formatted = apply_chat_template_str(effective_messages, tmpl);
    if (formatted.empty()) {
        formatted = effective_messages;   // fallback: caller pre-formatted
    }

    GenerateRequest req;
    req.temperature = temperature;
    req.max_tokens  = max_tokens;
    req.tokens      = engine_->tokenize(formatted);
    // §4.2 — propagate session_id to the scheduler so KV-prefix reuse
    // can fire on this sub-call.  Empty string = no reuse (scheduler
    // falls back to normal cold prefill).
    req.session_id  = sub_session_id;
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
        // Allocate engine-owned JSON result string using nlohmann::json so
        // that embedded quotes, backslashes, and control characters in
        // result.text are properly escaped. Raw concatenation produces
        // malformed JSON when the model output contains `"`, `\n`, etc.
        //
        // §9.5 — surface finish_reason ("stop" / "length" / "error") so
        // plugins (research_synthesizer) can detect length-truncation and
        // resume.  prompt_tokens / generated_tokens are added under the
        // same key so §9.6 sub-call usage accumulation can read them.
        const std::string out = json{
            {"text",             result.text},
            {"finish_reason",    result.finish_reason},
            {"prompt_tokens",    result.prompt_tokens},
            {"completion_tokens",result.generated_tokens}
        }.dump();
        char* buf = static_cast<char*>(tramp_alloc(out.size() + 1));
        if (buf) {
            std::copy(out.begin(), out.end(), buf);
            buf[out.size()] = '\0';
        }
        *result_json_out = buf;
    }

    // §9.6 — accumulate sub-call token usage on the parent ctx so the
    // outer /v1/chat/completions response can surface usage.research_*.
    {
        std::lock_guard<std::mutex> lk(ctx->metadata_mu);
        auto& m = const_cast<truellm_context_t*>(ctx)->metadata;
        auto bump = [&](const char* k, int64_t delta) {
            int64_t cur = 0;
            auto it = m.find(k);
            if (it != m.end()) {
                try { cur = std::stoll(it->second); } catch (...) { cur = 0; }
            }
            m[k] = std::to_string(cur + delta);
        };
        bump("research.subcall_prompt_tokens",     result.prompt_tokens);
        bump("research.subcall_completion_tokens", result.generated_tokens);
        bump("research.subcall_count",             1);
    }

    return TRUELLM_OK;
}

truellm_error_t PluginBridge::vtable_inject_tokens(
    truellm_context_t* ctx, int64_t pos, const char* tokens_json)
{
    {
        std::lock_guard<std::mutex> lk(counters_mu_);
        ++counters_.inject_tokens_calls;
    }
    if (!ctx || !tokens_json) return TRUELLM_ERR_INVALID;
    if (!engine_)             return TRUELLM_ERR_INTERNAL;

    std::vector<int32_t> ids;
    try {
        auto arr = json::parse(tokens_json);
        if (!arr.is_array()) return TRUELLM_ERR_INVALID;
        ids.reserve(arr.size());
        for (const auto& v : arr) {
            if (v.is_number_integer())       ids.push_back(v.get<int32_t>());
            else if (v.is_number_unsigned()) ids.push_back(
                                                 static_cast<int32_t>(v.get<uint32_t>()));
            else return TRUELLM_ERR_INVALID;
        }
    } catch (...) {
        return TRUELLM_ERR_INVALID;
    }
    if (ids.empty()) return TRUELLM_ERR_INVALID;

    ErrorCode ec = engine_->inject_tokens(ctx->request_id, pos, ids);
    switch (ec) {
        case ErrorCode::Ok:                 return TRUELLM_OK;
        case ErrorCode::InvalidArgument:    return TRUELLM_ERR_INVALID;
        case ErrorCode::NotImplemented: {
            {
                std::lock_guard<std::mutex> lk(counters_mu_);
                ++counters_.inject_tokens_not_impl;
            }
            static std::once_flag warned;
            std::call_once(warned, []{
                spdlog::warn("[PluginBridge] inject_tokens: backend returned "
                             "NotImplemented; plugins should re-tokenise the "
                             "updated messages as a fallback.");
            });
            return TRUELLM_ERR_NOT_FOUND;
        }
        default:                             return TRUELLM_ERR_INTERNAL;
    }
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
    // api_minor 2 is required for plugin_data_dir/plugin_fs_* storage APIs.
    // Keep this explicit so storage plugins fail only when the vtable is
    // actually missing the functions, not because a stale macro advertised
    // the older api_minor 1 surface.
    host_vtable_.api_minor   = 3u;

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

    // api_minor >= 2: sandboxed per-plugin blob filesystem.
    host_vtable_.plugin_data_dir   = tramp_plugin_data_dir;
    host_vtable_.plugin_fs_write   = tramp_plugin_fs_write;
    host_vtable_.plugin_fs_read    = tramp_plugin_fs_read;
    host_vtable_.plugin_fs_unlink  = tramp_plugin_fs_unlink;
    host_vtable_.plugin_fs_listdir = tramp_plugin_fs_listdir;

    // api_minor >= 3: cross-plugin tool dispatch (§9.3.4).
    host_vtable_.call_tool         = tramp_call_tool;
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

// ---------------------------------------------------------------------------
// api_minor >= 2 — sandboxed per-plugin blob filesystem
//
// Path policy (enforced inside resolve_plugin_path_):
//   1. plugin_id must match [a-z0-9_]+ — no separators, no dots.
//   2. relpath must NOT start with /, \, or a Windows drive letter.
//   3. Each path component (split on / or \) must NOT be "" or ".." or ".".
//   4. NUL bytes anywhere → reject.
// Anything else lands inside ${plugin_data_root_}/<plugin_id>/, never above.
// ---------------------------------------------------------------------------
namespace {
bool plugin_id_ok(const std::string& id)
{
    if (id.empty() || id.size() > 64) return false;
    for (char c : id) {
        const bool ok = (c >= 'a' && c <= 'z')
                     || (c >= 'A' && c <= 'Z')
                     || (c >= '0' && c <= '9')
                     || c == '_';
        if (!ok) return false;
    }
    return true;
}

bool split_relpath_ok(const std::string& rel, std::vector<std::string>& parts)
{
    parts.clear();
    if (rel.empty() || rel.size() > 4096) return false;
    for (char c : rel) if (c == '\0') return false;
    // Reject absolute paths and Windows drive letters.
    if (rel[0] == '/' || rel[0] == '\\') return false;
    if (rel.size() >= 2 && std::isalpha(static_cast<unsigned char>(rel[0]))
        && rel[1] == ':') return false;

    std::string cur;
    for (char c : rel) {
        if (c == '/' || c == '\\') {
            if (cur.empty()) return false;       // empty component
            if (cur == "." || cur == "..") return false;
            parts.push_back(std::move(cur));
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    if (cur.empty()) return false;               // trailing slash
    if (cur == "." || cur == "..") return false;
    parts.push_back(std::move(cur));
    return !parts.empty();
}
} // anonymous namespace

const char* PluginBridge::vtable_plugin_data_dir(const char* plugin_id)
{
    if (!plugin_id || plugin_data_root_.empty()) return nullptr;
    std::string id = plugin_id;
    if (!plugin_id_ok(id)) return nullptr;

    std::lock_guard<std::mutex> lk(plugin_dir_mu_);
    auto it = per_plugin_dir_.find(id);
    if (it != per_plugin_dir_.end()) return it->second.c_str();

    fs::path dir = fs::path(plugin_data_root_) / id;
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) {
        spdlog::warn("[PluginBridge] could not create plugin data dir {}: {}",
                     dir.string(), ec.message());
        return nullptr;
    }
    auto inserted = per_plugin_dir_.emplace(id, dir.string());
    return inserted.first->second.c_str();
}

namespace {
// Returns absolute path under <root>/<plugin_id>/<relpath> or empty on
// validation failure.  Does not perform I/O.
std::string resolve_plugin_path(const std::string& root,
                                 const std::string& plugin_id,
                                 const std::string& relpath)
{
    if (root.empty() || !plugin_id_ok(plugin_id)) return {};
    std::vector<std::string> parts;
    if (!split_relpath_ok(relpath, parts)) return {};
    fs::path p = fs::path(root) / plugin_id;
    for (const auto& seg : parts) p /= seg;
    return p.string();
}
} // anonymous namespace

truellm_error_t PluginBridge::vtable_plugin_fs_write(
    const char* plugin_id, const char* relpath,
    const void* bytes, size_t n_bytes)
{
    if (plugin_data_root_.empty())   return TRUELLM_ERR_PERMISSION;
    if (!plugin_id || !relpath)      return TRUELLM_ERR_INVALID;
    if (!bytes && n_bytes > 0)       return TRUELLM_ERR_INVALID;

    std::string abs = resolve_plugin_path(plugin_data_root_, plugin_id, relpath);
    if (abs.empty()) return TRUELLM_ERR_INVALID;

    fs::path target(abs);
    fs::path tmp = target;
    tmp += ".tmp";

    std::error_code ec;
    fs::create_directories(target.parent_path(), ec);
    if (ec) {
        spdlog::warn("[PluginBridge] fs_write: mkdir failed for {}: {}",
                     target.parent_path().string(), ec.message());
        return TRUELLM_ERR_INTERNAL;
    }

    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) return TRUELLM_ERR_INTERNAL;
        if (n_bytes > 0) out.write(reinterpret_cast<const char*>(bytes),
                                    static_cast<std::streamsize>(n_bytes));
        if (!out) return TRUELLM_ERR_INTERNAL;
        out.flush();
    }

    fs::rename(tmp, target, ec);
    if (ec) {
        // On Windows, rename onto an existing file fails — fall back to
        // remove+rename.  Linux/macOS rename(2) is already atomic.
        std::error_code rm_ec;
        fs::remove(target, rm_ec);
        fs::rename(tmp, target, ec);
        if (ec) {
            spdlog::warn("[PluginBridge] fs_write: rename failed: {}",
                         ec.message());
            fs::remove(tmp, rm_ec);
            return TRUELLM_ERR_INTERNAL;
        }
    }
    return TRUELLM_OK;
}

truellm_error_t PluginBridge::vtable_plugin_fs_read(
    const char* plugin_id, const char* relpath,
    void** out_bytes, size_t* out_n)
{
    if (out_bytes) *out_bytes = nullptr;
    if (out_n)     *out_n     = 0;
    if (plugin_data_root_.empty()) return TRUELLM_ERR_PERMISSION;
    if (!plugin_id || !relpath || !out_bytes || !out_n)
        return TRUELLM_ERR_INVALID;

    std::string abs = resolve_plugin_path(plugin_data_root_, plugin_id, relpath);
    if (abs.empty()) return TRUELLM_ERR_INVALID;

    std::error_code ec;
    if (!fs::exists(abs, ec) || !fs::is_regular_file(abs, ec))
        return TRUELLM_ERR_NOT_FOUND;

    std::ifstream in(abs, std::ios::binary | std::ios::ate);
    if (!in) return TRUELLM_ERR_INTERNAL;
    auto size = static_cast<std::streamsize>(in.tellg());
    if (size < 0) return TRUELLM_ERR_INTERNAL;
    in.seekg(0, std::ios::beg);

    void* buf = tramp_alloc(static_cast<size_t>(size) + 1);
    if (!buf) return TRUELLM_ERR_INTERNAL;
    in.read(reinterpret_cast<char*>(buf), size);
    if (!in && !in.eof()) {
        tramp_dealloc(buf);
        return TRUELLM_ERR_INTERNAL;
    }
    static_cast<char*>(buf)[size] = '\0';   // NUL terminator for text use
    *out_bytes = buf;
    *out_n     = static_cast<size_t>(size);
    return TRUELLM_OK;
}

truellm_error_t PluginBridge::vtable_plugin_fs_unlink(
    const char* plugin_id, const char* relpath)
{
    if (plugin_data_root_.empty()) return TRUELLM_ERR_PERMISSION;
    if (!plugin_id || !relpath)    return TRUELLM_ERR_INVALID;
    std::string abs = resolve_plugin_path(plugin_data_root_, plugin_id, relpath);
    if (abs.empty()) return TRUELLM_ERR_INVALID;

    std::error_code ec;
    if (!fs::exists(abs, ec)) return TRUELLM_ERR_NOT_FOUND;
    if (!fs::remove(abs, ec)) return TRUELLM_ERR_INTERNAL;
    return TRUELLM_OK;
}

truellm_error_t PluginBridge::vtable_plugin_fs_listdir(
    const char* plugin_id, const char* relpath,
    char*** out_names, size_t* out_count)
{
    if (out_names) *out_names = nullptr;
    if (out_count) *out_count = 0;
    if (plugin_data_root_.empty()) return TRUELLM_ERR_PERMISSION;
    if (!plugin_id || !relpath || !out_names || !out_count)
        return TRUELLM_ERR_INVALID;

    // Empty relpath = the plugin's root.
    fs::path target;
    if (relpath[0] == '\0') {
        if (!plugin_id_ok(plugin_id)) return TRUELLM_ERR_INVALID;
        target = fs::path(plugin_data_root_) / plugin_id;
    } else {
        std::string abs = resolve_plugin_path(plugin_data_root_,
                                               plugin_id, relpath);
        if (abs.empty()) return TRUELLM_ERR_INVALID;
        target = abs;
    }

    std::error_code ec;
    if (!fs::is_directory(target, ec)) return TRUELLM_ERR_NOT_FOUND;

    std::vector<std::string> names;
    for (const auto& e : fs::directory_iterator(target, ec)) {
        if (ec) break;
        names.push_back(e.path().filename().string());
    }

    // Allocate a NULL-terminated array of c_strs through the engine heap.
    char** arr = static_cast<char**>(tramp_alloc((names.size() + 1) * sizeof(char*)));
    if (!arr) return TRUELLM_ERR_INTERNAL;
    for (size_t i = 0; i < names.size(); ++i) {
        char* slot = static_cast<char*>(tramp_alloc(names[i].size() + 1));
        if (!slot) {
            for (size_t j = 0; j < i; ++j) tramp_dealloc(arr[j]);
            tramp_dealloc(arr);
            return TRUELLM_ERR_INTERNAL;
        }
        std::memcpy(slot, names[i].c_str(), names[i].size() + 1);
        arr[i] = slot;
    }
    arr[names.size()] = nullptr;
    *out_names = arr;
    *out_count = names.size();
    return TRUELLM_OK;
}

// ---------------------------------------------------------------------------
// vtable_call_tool — cross-plugin tool dispatch  (Code_Capability_design.md
// §9.3.4 / api_minor=3).
//
// Routes through PluginBridge::dispatch_tool so the agent.allowed_tools
// enforcement and the tool's own error mapping behave exactly as they do
// for a model-emitted tool_call from openai_router.  Result is a JSON
// envelope so the caller can distinguish "the dispatch path failed"
// (TRUELLM_ERR_*) from "the tool returned an error code" (ok=false +
// error_code in the body).
// ---------------------------------------------------------------------------
truellm_error_t PluginBridge::vtable_call_tool(
    const truellm_context_t* ctx,
    const char* qualified_name,
    const char* args_json,
    char** result_json_out)
{
    if (result_json_out) *result_json_out = nullptr;
    if (!qualified_name)  return TRUELLM_ERR_INVALID;

    // Enforce agent.allowed_tools just as the router does, so a plugin
    // that's already had its scope narrowed by plan_mode_guard cannot
    // sneak through a privileged call via call_tool.
    std::string allow_csv;
    if (ctx) {
        std::lock_guard<std::mutex> lk(ctx->metadata_mu);
        auto it = ctx->metadata.find("agent.allowed_tools");
        if (it != ctx->metadata.end()) allow_csv = it->second;
    }
    auto allowed = [&]() {
        if (allow_csv.empty()) return true;
        const std::string qn = qualified_name;
        const std::string sn =
            qn.find('.') == std::string::npos ? qn
                                              : qn.substr(qn.find('.') + 1);
        size_t pos = 0;
        while (pos < allow_csv.size()) {
            size_t end = allow_csv.find(',', pos);
            if (end == std::string::npos) end = allow_csv.size();
            size_t a = allow_csv.find_first_not_of(" \t", pos);
            if (a < end) {
                size_t b = allow_csv.find_last_not_of(" \t", end - 1);
                std::string tok = allow_csv.substr(a, b - a + 1);
                if (tok == qn || tok == sn) return true;
                if (!tok.empty() && tok.back() == '*' && tok.size() >= 2
                    && tok[tok.size() - 2] == '.')
                {
                    const std::string prefix = tok.substr(0, tok.size() - 1);
                    if (qn.rfind(prefix, 0) == 0) return true;
                }
            }
            pos = end + 1;
        }
        return false;
    }();

    auto build_envelope = [&](bool ok, int code,
                              const std::string& msg,
                              const std::string& payload) -> truellm_error_t {
        if (!result_json_out) return TRUELLM_OK;
        const std::string out = json{
            {"ok",         ok},
            {"error_code", code},
            {"error_msg",  msg},
            {"payload",    payload}
        }.dump();
        char* buf = static_cast<char*>(tramp_alloc(out.size() + 1));
        if (!buf) return TRUELLM_ERR_INTERNAL;
        std::copy(out.begin(), out.end(), buf);
        buf[out.size()] = '\0';
        *result_json_out = buf;
        return TRUELLM_OK;
    };

    if (!allowed) {
        return build_envelope(false,
                              static_cast<int>(TRUELLM_ERR_PERMISSION),
                              std::string("tool '") + qualified_name +
                                "' is not in the allow-list (agent.allowed_tools)",
                              "");
    }

    ToolCallResult tc = dispatch_tool(
        qualified_name, args_json ? args_json : "",
        const_cast<truellm_context_t*>(ctx));

    return build_envelope(
        tc.error == TRUELLM_OK,
        static_cast<int>(tc.error),
        tc.error_msg,
        tc.error == TRUELLM_OK ? tc.payload : "");
}

} // namespace truellm
