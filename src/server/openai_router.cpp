// ---------------------------------------------------------------------------
// openai_router.cpp
// ---------------------------------------------------------------------------

#include "openai_router.h"
#include "chat_template.h"
#include "../inference/engine_interface.h"
#include "../plugin_bridge/plugin_bridge.h"

#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace truellm {

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
static int64_t unix_now()
{
    return static_cast<int64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
}

static json error_response(int status, const std::string& msg,
                            const std::string& type = "server_error")
{
    return {{"error", {{"message", msg}, {"type", type}}}};
}

// ---------------------------------------------------------------------------
// sanitize_utf8 — drop a trailing partial UTF-8 sequence so the string is
// always valid UTF-8. Model output truncated mid-multibyte character (e.g.
// when max_tokens lands between BPE bytes) would otherwise cause
// nlohmann::json::dump() to throw type_error.316 and turn the whole
// response into a 500.  We truncate to the last complete code point.
// ---------------------------------------------------------------------------
static std::string sanitize_utf8(const std::string& s)
{
    size_t i = 0;
    size_t last_good = 0;
    while (i < s.size()) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        size_t need;
        if      ((c & 0x80) == 0x00) need = 1;   // 0xxxxxxx
        else if ((c & 0xE0) == 0xC0) need = 2;   // 110xxxxx
        else if ((c & 0xF0) == 0xE0) need = 3;   // 1110xxxx
        else if ((c & 0xF8) == 0xF0) need = 4;   // 11110xxx
        else { ++i; continue; }                  // stray continuation byte
        if (i + need > s.size()) break;          // incomplete tail → drop it
        bool ok = true;
        for (size_t k = 1; k < need; ++k) {
            if ((static_cast<unsigned char>(s[i + k]) & 0xC0) != 0x80) { ok = false; break; }
        }
        if (!ok) { ++i; continue; }
        i += need;
        last_good = i;
    }
    if (last_good == s.size()) return s;
    return s.substr(0, last_good);
}

// ---------------------------------------------------------------------------
// resolve_template — pick the right template name for a model when the user
// has not explicitly configured one ("auto").
//
// GGUF `general.architecture` values:
//   "llama"          → LLaMA 2 / 3 / 3.1 / 3.2 / Mistral (use llama3 format)
//   "gemma", "gemma2", "gemma3" → Google Gemma
//   "phi3"           → Microsoft Phi-3 / Phi-3.5
//   anything else    → fall back to ChatML (Qwen, Mistral-v0.3+, etc.)
// ---------------------------------------------------------------------------
// resolve_template / apply_chat_template / resolve_message_content live in
// "chat_template.h" so plugin_bridge can apply the same templating logic to
// the messages_json that arrives from host->call_engine() sub-calls.  The
// originals here have been replaced with thin file-local aliases that keep
// the existing call-sites in this translation unit working unchanged.
using ::truellm::resolve_template;
using ::truellm::resolve_message_content;
using ::truellm::apply_chat_template;

// ---------------------------------------------------------------------------
// tool_args_string — extract the "arguments" field from a tool_calls entry
// as a JSON string.  The OpenAI spec says arguments must be a string, but
// models (including Nemotron-H) often emit it as a raw JSON object.
// We handle both: if it's already a string, return it as-is; if it's an
// object or array, serialise it; otherwise return "{}".
// ---------------------------------------------------------------------------
static std::string tool_args_string(const json& call)
{
    const json fn = call.value("function", json::object());
    if (!fn.contains("arguments")) return "{}";
    const json& a = fn["arguments"];
    if (a.is_string()) return a.get<std::string>();
    if (a.is_object() || a.is_array()) return a.dump();
    return "{}";
}

// resolve_message_content + apply_chat_template moved to "chat_template.h"

// ---------------------------------------------------------------------------
// Context splitter helpers
// ---------------------------------------------------------------------------

// Extract the first user message content — used as the "question" framing
// for per-chunk summarisation prompts.
static std::string extract_user_question(const json& messages)
{
    for (const auto& m : messages)
        if (m.value("role", "") == "user")
            return m.value("content", "");
    return "";
}

// Split `content` into chunks of at most `target_chars` characters, trying
// to break at newline boundaries so chunks stay semantically coherent.
static std::vector<std::string> chunk_content(const std::string& content,
                                               std::size_t        target_chars)
{
    std::vector<std::string> chunks;
    std::size_t pos = 0;
    while (pos < content.size()) {
        std::size_t end = pos + target_chars;
        if (end >= content.size()) {
            chunks.push_back(content.substr(pos));
            break;
        }
        // Prefer to break at the last newline before the hard cut.
        std::size_t nl = content.rfind('\n', end);
        if (nl != std::string::npos && nl > pos)
            end = nl + 1;
        chunks.push_back(content.substr(pos, end - pos));
        pos = end;
    }
    return chunks;
}

// summarize_chunks — run a short inference over each chunk and return the
// concatenated summaries as a single string.
std::string OpenAIRouter::summarize_chunks(const std::vector<std::string>& chunks,
                                           const std::string& user_question,
                                           const std::string& tmpl,
                                           int                summary_tokens,
                                           float              temperature) const
{
    const int n = std::min(static_cast<int>(chunks.size()),
                           cfg_.plugins.context_split_max_chunks);
    spdlog::info("[OpenAI] context-split: {} chunk(s) to summarise (max={})",
                 chunks.size(), cfg_.plugins.context_split_max_chunks);

    std::string all_summaries;

    for (int i = 0; i < n; ++i) {
        // Build a short summarisation prompt for this chunk.
        // We do content-summary (not relevance-filter) so that tool results
        // like web pages are faithfully compressed rather than discarded.
        json chunk_msgs = json::array();
        chunk_msgs.push_back({
            {"role",    "system"},
            {"content", "You are a content summariser. "
                        "Summarise the key topics, facts, and information in the "
                        "provided content section. Be concise but informative. "
                        "Preserve important details such as names, versions, URLs, "
                        "and technical terms. If the section contains code, "
                        "describe what it does."}
        });
        std::string user_body =
            "Content section " + std::to_string(i + 1) + " of " +
            std::to_string(n) + ":\n---\n" + chunks[i] + "\n---\n\n"
            "Summarise the key content of this section.";
        chunk_msgs.push_back({{"role", "user"}, {"content", user_body}});

        std::string chunk_prompt = apply_chat_template(chunk_msgs, tmpl);
        GenerateRequest cr;
        cr.tokens      = engine_->tokenize(chunk_prompt);
        cr.max_tokens  = summary_tokens;
        cr.temperature = temperature;
        cr.top_p       = cfg_.sampling.top_p;

        if (cr.tokens.empty()) continue;

        GenerateResult res = engine_->generate(cr);
        if (res.error != ErrorCode::Ok || res.text.empty()) {
            spdlog::warn("[OpenAI] context-split chunk {}/{} inference error: {}",
                         i + 1, n, res.error_message);
            continue;
        }

        spdlog::debug("[OpenAI] context-split chunk {}/{}: {} gen_toks",
                      i + 1, n, res.generated_tokens);
        all_summaries += "[Section " + std::to_string(i + 1) + "]: " +
                         res.text + "\n\n";
    }

    if (static_cast<int>(chunks.size()) > n) {
        all_summaries += "[Note: content had " + std::to_string(chunks.size()) +
                         " sections; only first " + std::to_string(n) +
                         " were processed due to context_split_max_chunks limit]\n";
    }

    return all_summaries.empty() ? "[Tool result could not be summarised]" : all_summaries;
}

// ---------------------------------------------------------------------------
// Constructor / register_routes
// ---------------------------------------------------------------------------
OpenAIRouter::OpenAIRouter(const TrueLLMConfig& cfg, EngineInterface* engine)
    : cfg_(cfg), engine_(engine)
{}

void OpenAIRouter::register_routes(httplib::Server& svr, AuthCheck auth)
{
    svr.Get("/v1/models",
        [this, auth](const httplib::Request& req, httplib::Response& res) {
            if (!auth(req, res)) return;
            handle_models(req, res);
        });

    svr.Post("/v1/completions",
        [this, auth](const httplib::Request& req, httplib::Response& res) {
            if (!auth(req, res)) return;
            handle_completions(req, res);
        });

    svr.Post("/v1/chat/completions",
        [this, auth](const httplib::Request& req, httplib::Response& res) {
            if (!auth(req, res)) return;
            handle_chat_completions(req, res);
        });

    // §9.4 — /v1/research alias.  Thin wrapper that forces
    // truellm_metadata.mode = "research" before delegating to the chat
    // completions handler so the research_orchestrator generator picks
    // the request up.  Clients can still set effort / session.id through
    // the existing metadata channels.
    svr.Post("/v1/research",
        [this, auth](const httplib::Request& req, httplib::Response& res) {
            if (!auth(req, res)) return;
            // Cheap trick: parse, mutate, replace req body in-place.
            // httplib's Request.body is non-const, so we can rewrite it.
            // If the body isn't valid JSON, fall through; the chat handler
            // will surface the parse error to the caller with the same
            // shape as a direct /v1/chat/completions call.
            auto& mutable_req = const_cast<httplib::Request&>(req);
            json body = json::parse(mutable_req.body, nullptr, false);
            if (body.is_object()) {
                if (!body.contains("truellm_metadata") || !body["truellm_metadata"].is_object())
                    body["truellm_metadata"] = json::object();
                body["truellm_metadata"]["mode"] = "research";
                if (!body["truellm_metadata"].contains("effort"))
                    body["truellm_metadata"]["effort"] = "medium";
                mutable_req.body = body.dump();
            }
            handle_chat_completions(req, res);
        });
}

// ---------------------------------------------------------------------------
// Trace helper — dump a JSON value as a compact one-liner (capped at max_chars)
// ---------------------------------------------------------------------------
static std::string trace_json(const json& j, std::size_t max_chars = 256)
{
    std::string s = j.dump();
    if (s.size() > max_chars) s = s.substr(0, max_chars) + "…";
    return s;
}

// ---------------------------------------------------------------------------
// parse_tool_calls — extract a tool_calls JSON array from raw model output.
// Handles multiple output formats produced by different model families:
//
//   1. Raw JSON object:  {"tool_calls":[...]}
//   2. Fenced block:     ```json\n{"tool_calls":[...]}\n```
//   3. Single call obj:  {"name":"fn","arguments":{...}}
//   4. OpenAI call obj:  {"function":{"name":"fn","arguments":{...}}}
//   5. LLaMA 3.1 tag:    <|python_tag|>{"name":"fn","parameters":{...}}
//   6. JSON in <tool_call>...</tool_call> tags
//   7. Tool shorthand:   {"tool":"fn","parameters":{...}}
//
// Returns an empty array if no tool calls are found.
// ---------------------------------------------------------------------------
static json normalise_single_call(const json& j)
{
    // Build an OpenAI-style tool_call object from a bare call object.
    std::string name;
    json args;

    if (j.contains("name")) {
        name = j.value("name", "");
        args = j.contains("arguments")  ? j["arguments"]
             : j.contains("parameters") ? j["parameters"]
             : json::object();
    } else if (j.contains("tool")) {
        name = j.value("tool", "");
        args = j.contains("arguments")  ? j["arguments"]
             : j.contains("parameters") ? j["parameters"]
             : json::object();
    } else if (j.contains("function") && j["function"].is_object()) {
        name = j["function"].value("name", "");
        args = j["function"].contains("arguments")  ? j["function"]["arguments"]
             : j["function"].contains("parameters") ? j["function"]["parameters"]
             : json::object();
    } else {
        return json{};   // not recognisable
    }

    if (name.empty()) return json{};

    // args might be a string that itself is JSON.
    if (args.is_string()) {
        auto parsed = json::parse(args.get<std::string>(), nullptr, false);
        if (!parsed.is_discarded()) args = parsed;
    }

    return json{
        {"id",       "call_0"},
        {"type",     "function"},
        {"function", {{"name", name}, {"arguments", args}}}
    };
}

static json parse_tool_calls(const std::string& raw)
{
    // Helper: check a parsed JSON object for tool call content.
    auto extract = [](const json& j) -> json {
        if (j.is_discarded()) return json::array();

        // {"tool_calls":[...]}
        if (j.contains("tool_calls") && j["tool_calls"].is_array())
            return j["tool_calls"];

        // Single call object {"name":..., "arguments":...} or {"function":{...}}
        auto single = normalise_single_call(j);
        if (!single.is_null() && !single.empty())
            return json::array({single});

        return json::array();
    };

    // Attempt 1: entire output is JSON
    {
        auto j = json::parse(raw, nullptr, false);
        auto tc = extract(j);
        if (!tc.empty()) return tc;
    }

    // Attempt 2: fenced ```json ... ``` block
    {
        auto start = raw.find("```json");
        if (start != std::string::npos) {
            auto end = raw.find("```", start + 7);
            if (end != std::string::npos) {
                auto block = raw.substr(start + 7, end - start - 7);
                auto j = json::parse(block, nullptr, false);
                auto tc = extract(j);
                if (!tc.empty()) return tc;
            }
        }
    }

    // Attempt 3: fenced ``` block (no "json" tag)
    {
        auto start = raw.find("```\n");
        if (start != std::string::npos) {
            auto end = raw.find("```", start + 4);
            if (end != std::string::npos) {
                auto block = raw.substr(start + 4, end - start - 4);
                auto j = json::parse(block, nullptr, false);
                auto tc = extract(j);
                if (!tc.empty()) return tc;
            }
        }
    }

    // Attempt 4: LLaMA 3.1 <|python_tag|>...<|eom_id|> or bare after the tag
    {
        const std::string tag = "<|python_tag|>";
        auto pos = raw.find(tag);
        if (pos != std::string::npos) {
            auto content = raw.substr(pos + tag.size());
            // strip optional trailing <|eom_id|>
            auto eom = content.find("<|eom_id|>");
            if (eom != std::string::npos) content = content.substr(0, eom);
            auto j = json::parse(content, nullptr, false);
            auto tc = extract(j);
            if (!tc.empty()) return tc;
        }
    }

    // Attempt 5: <tool_call>...</tool_call> tags
    {
        const std::string open  = "<tool_call>";
        const std::string close = "</tool_call>";
        auto s = raw.find(open);
        if (s != std::string::npos) {
            auto e = raw.find(close, s);
            if (e != std::string::npos) {
                auto content = raw.substr(s + open.size(), e - s - open.size());
                auto j = json::parse(content, nullptr, false);
                auto tc = extract(j);
                if (!tc.empty()) return tc;
            }
        }
    }

    // Attempt 6: first bare JSON object anywhere in the output
    {
        auto s = raw.find('{');
        while (s != std::string::npos) {
            // Find the matching closing brace.
            int depth = 0;
            std::size_t e = s;
            for (; e < raw.size(); ++e) {
                if (raw[e] == '{') ++depth;
                else if (raw[e] == '}') { if (--depth == 0) break; }
            }
            if (depth == 0) {
                auto block = raw.substr(s, e - s + 1);
                auto j = json::parse(block, nullptr, false);
                auto tc = extract(j);
                if (!tc.empty()) return tc;
            }
            s = raw.find('{', s + 1);
        }
    }

    return json::array();
}

// ---------------------------------------------------------------------------
// make_context — construct a per-request plugin context
// ---------------------------------------------------------------------------
static std::unique_ptr<truellm_context_t> make_context(
    const TrueLLMConfig& cfg, EngineInterface* engine,
    int32_t max_tokens, float temperature)
{
    auto ctx           = std::make_unique<truellm_context_t>();
    ctx->request_id    = "req-" + std::to_string(unix_now());
    ctx->model_id      = cfg.model.alias.empty() ? "default" : cfg.model.alias;
    ctx->model_arch    = engine ? engine->get_model_info().architecture : "unknown";
    ctx->token_budget  = engine ? engine->get_token_budget() : 0;
    ctx->max_tokens    = max_tokens;
    ctx->temperature   = temperature;
    ctx->status_sink   = nullptr;
    return ctx;
}

// ---------------------------------------------------------------------------
// is_tool_allowed — enforce agent.allowed_tools (Code_Capability_design.md
// §3.3.2 plan_mode_guard).  When the CSV is empty / absent, every tool is
// allowed.  When non-empty, accept a tool call iff its qualified name OR
// its short (post-dot) form is on the list.  The match is whitespace-
// trimmed but case-sensitive — tool names are [a-z0-9_]+, not user prose.
// ---------------------------------------------------------------------------
static bool is_tool_allowed(const std::string& qname,
                            const std::string& allow_csv)
{
    if (allow_csv.empty()) return true;
    const std::string short_name =
        qname.find('.') == std::string::npos ? qname
                                              : qname.substr(qname.find('.') + 1);
    size_t pos = 0;
    while (pos < allow_csv.size()) {
        size_t end = allow_csv.find(',', pos);
        if (end == std::string::npos) end = allow_csv.size();
        size_t a = allow_csv.find_first_not_of(" \t", pos);
        if (a < end) {
            size_t b = allow_csv.find_last_not_of(" \t", end - 1);
            std::string tok = allow_csv.substr(a, b - a + 1);
            if (tok == qname || tok == short_name) return true;
            // wildcard form "plugin.*" matches every tool under a plugin
            if (!tok.empty() && tok.back() == '*' && tok.size() >= 2
                && tok[tok.size() - 2] == '.')
            {
                const std::string prefix = tok.substr(0, tok.size() - 1);
                if (qname.rfind(prefix, 0) == 0) return true;
            }
        }
        pos = end + 1;
    }
    return false;
}

// ---------------------------------------------------------------------------
// merge_subcall_usage — Code_Capability_design.md §9.6.
// Read the per-request sub-call usage counters that vtable_call_engine
// accumulates on ctx.metadata and merge them into the OpenAI-style usage
// object the response builder is about to emit.  Returns the input
// `usage` json modified in-place.  Safe to call with ctx == nullptr.
// ---------------------------------------------------------------------------
static void merge_subcall_usage(nlohmann::json& usage,
                                const truellm_context_t* ctx)
{
    if (!ctx) return;
    int64_t sub_prompt = 0, sub_completion = 0, sub_count = 0;
    {
        std::lock_guard<std::mutex> lk(ctx->metadata_mu);
        auto pull = [&](const char* k) -> int64_t {
            auto it = ctx->metadata.find(k);
            if (it == ctx->metadata.end()) return 0;
            try { return std::stoll(it->second); } catch (...) { return 0; }
        };
        sub_prompt     = pull("research.subcall_prompt_tokens");
        sub_completion = pull("research.subcall_completion_tokens");
        sub_count      = pull("research.subcall_count");
    }
    if (sub_count <= 0) return;
    usage["research_prompt_tokens"]     = sub_prompt;
    usage["research_completion_tokens"] = sub_completion;
    usage["research_total_tokens"]      = sub_prompt + sub_completion;
    usage["research_subcall_count"]     = sub_count;
}

// ---------------------------------------------------------------------------
// format_tool_denial — Code_Capability_design.md §4.5.
// Build the role-`tool` message body used when a tool call is refused
// (either by the agent.allowed_tools allow-list, or because the plugin
// itself returned TRUELLM_ERR_PERMISSION).  Returning structured JSON
// instead of free text lets the model parse the result deterministically.
// ---------------------------------------------------------------------------
static std::string format_tool_denial(const std::string& reason)
{
    return nlohmann::json{
        {"denied", true},
        {"reason", reason}
    }.dump();
}

// ---------------------------------------------------------------------------
// absorb_client_metadata — copy per-request metadata from well-known HTTP
// fields into ctx->metadata so that opt-in flags like "rlm_eligible"
// reach the plugin layer.  Three channels are supported, matching the
// integration test harness:
//   1. JSON body:   extra_body.truellm.context_metadata  (object of strings)
//   2. JSON body:   truellm_metadata                     (object of strings)
//   3. HTTP header: x-truellm-context-metadata           (JSON object string)
//
// Non-string values are JSON-stringified so the metadata map is always
// std::string → std::string.  Invalid JSON in any channel is ignored
// without failing the request.
// ---------------------------------------------------------------------------
static void absorb_client_metadata(truellm_context_t* ctx,
                                   const nlohmann::json& body,
                                   const httplib::Request& req)
{
    if (!ctx) return;

    auto merge_object = [&](const nlohmann::json& obj) {
        if (!obj.is_object()) return;
        std::lock_guard<std::mutex> lk(ctx->metadata_mu);
        for (auto it = obj.begin(); it != obj.end(); ++it) {
            if (it.value().is_string()) {
                ctx->metadata[it.key()] = it.value().get<std::string>();
            } else {
                ctx->metadata[it.key()] = it.value().dump();
            }
        }
    };

    // Channel 1: extra_body.truellm.context_metadata
    if (body.contains("extra_body") && body["extra_body"].is_object()) {
        const auto& eb = body["extra_body"];
        if (eb.contains("truellm") && eb["truellm"].is_object()) {
            const auto& tl = eb["truellm"];
            if (tl.contains("context_metadata")) merge_object(tl["context_metadata"]);
        }
    }
    // Channel 2: truellm_metadata at root
    if (body.contains("truellm_metadata")) merge_object(body["truellm_metadata"]);

    // Channel 3: header (JSON object string)
    const auto hv = req.get_header_value("x-truellm-context-metadata");
    if (!hv.empty()) {
        auto parsed = nlohmann::json::parse(hv, nullptr, false);
        if (!parsed.is_discarded()) merge_object(parsed);
    }
}

// ---------------------------------------------------------------------------
// GET /v1/models
// ---------------------------------------------------------------------------
void OpenAIRouter::handle_models(const httplib::Request&, httplib::Response& res)
{
    std::string model_id = cfg_.model.alias.empty()
                         ? cfg_.model.path
                         : cfg_.model.alias;
    if (model_id.empty()) model_id = "default";

    json body = {
        {"object", "list"},
        {"data",   json::array({
            {{"id",       model_id},
             {"object",   "model"},
             {"created",  unix_now()},
             {"owned_by", "truellm"}}
        })}
    };
    res.set_content(body.dump(), "application/json");
}

// ---------------------------------------------------------------------------
// POST /v1/completions
// ---------------------------------------------------------------------------
void OpenAIRouter::handle_completions(const httplib::Request& req,
                                      httplib::Response&      res)
{
    spdlog::debug("[OpenAI] POST /v1/completions");

    json body;
    try { body = json::parse(req.body); }
    catch (...) {
        spdlog::warn("[OpenAI] /v1/completions — invalid JSON body");
        res.status = 400;
        res.set_content(error_response(400, "Invalid JSON").dump(), "application/json");
        return;
    }

    if (!engine_ || !engine_->is_model_loaded()) {
        spdlog::warn("[OpenAI] /v1/completions — no model loaded");
        res.status = 503;
        res.set_content(error_response(503, "No model loaded").dump(), "application/json");
        return;
    }

    std::string prompt   = body.value("prompt", "");
    int max_tokens       = body.value("max_tokens",  cfg_.sampling.max_tokens);
    float temperature    = body.value("temperature", cfg_.sampling.temperature);
    bool do_stream       = body.value("stream",      false);

    spdlog::debug("[OpenAI] completions  max_tokens={} temp={:.2f} stream={} "
                  "prompt_chars={}", max_tokens, temperature, do_stream, prompt.size());
    spdlog::trace("[OpenAI] completions  prompt: {}",
                  prompt.size() > 256 ? prompt.substr(0, 256) + "…" : prompt);

    auto t_tok = std::chrono::steady_clock::now();
    auto tokens = engine_->tokenize(prompt);
    spdlog::debug("[OpenAI] tokenize -> {} tokens ({:.1f}ms)", tokens.size(),
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t_tok).count() / 1000.0);

    GenerateRequest greq;
    greq.tokens      = tokens;
    greq.max_tokens  = max_tokens;
    greq.temperature = temperature;
    greq.top_p       = body.value("top_p", cfg_.sampling.top_p);

    if (body.contains("stop")) {
        const auto& sv = body["stop"];
        if (sv.is_string())      greq.stop.push_back(sv.get<std::string>());
        else if (sv.is_array())  for (const auto& s : sv) if (s.is_string()) greq.stop.push_back(s.get<std::string>());
    }

    std::string req_id   = "cmpl-" + std::to_string(unix_now());
    greq.request_id      = req_id;
    std::string model_id = cfg_.model.alias.empty() ? "default" : cfg_.model.alias;

    // ── Build per-request plugin context ──────────────────────────────────────
    std::unique_ptr<truellm_context_t> ctx;
    if (plugin_bridge_) {
        ctx = make_context(cfg_, engine_, max_tokens, temperature);
        ctx->request_id = req_id;
        absorb_client_metadata(ctx.get(), body, req);
    }

    // ── Preprocessor chain ────────────────────────────────────────────────────
    if (plugin_bridge_ && plugin_bridge_->has_preprocessors()) {
        auto pr = plugin_bridge_->run_preprocessors(
            json({{"prompt", prompt}}).dump(), ctx.get());
        if (pr.decision == TRUELLM_PREPROC_RESPOND) {
            json resp = {
                {"id",      req_id},
                {"object",  "text_completion"},
                {"created", unix_now()},
                {"model",   model_id},
                {"choices", json::array({
                    {{"text",          pr.response_text},
                     {"index",         0},
                     {"finish_reason", "stop"}}
                })}
            };
            res.set_content(resp.dump(), "application/json");
            return;
        }
        if (pr.decision == TRUELLM_PREPROC_ABORT) {
            res.status = 400;
            res.set_content(error_response(400, "Request blocked by preprocessor").dump(),
                            "application/json");
            return;
        }
    }

    // ── Tool schema injection for /v1/completions ─────────────────────────────
    // Prepend a concise tool instruction to the prompt text.
    if (plugin_bridge_ && plugin_bridge_->has_tools()) {
        std::string schemas = plugin_bridge_->get_tool_schemas_json();
        if (!schemas.empty() && schemas != "[]") {
            prompt = "You have access to the following tools:\n" + schemas + "\n\n"
                     "To call a tool respond with a JSON object of the form:\n"
                     "{\"tool_calls\":[{\"id\":\"call_0\",\"type\":\"function\","
                     "\"function\":{\"name\":\"<qualified_name>\",\"arguments\":{<args>}}}]}\n\n"
                     + prompt;
            greq.tokens = engine_->tokenize(prompt);
            spdlog::debug("[OpenAI] completions  tool schemas injected  tools_json_len={}",
                          schemas.size());
        }
    }

    if (do_stream) {
        spdlog::debug("[OpenAI] completions stream starting  req_id={}", req_id);
        res.set_chunked_content_provider("text/event-stream",
            [this, greq = std::move(greq), req_id, model_id]
            (std::size_t /*offset*/, httplib::DataSink& sink) mutable -> bool
        {
            int n_tok = 0;
            greq.on_token = [&](int32_t id, const std::string& piece) {
                ++n_tok;
                spdlog::trace("[OpenAI] stream tok[{}] id={} piece={}", n_tok, id,
                              piece.empty() ? "<empty>" : piece);
                json delta = {
                    {"id",      req_id},
                    {"object",  "text_completion"},
                    {"model",   model_id},
                    {"choices", json::array({
                        {{"text",          piece},
                         {"index",         0},
                         {"finish_reason", nullptr}}
                    })}
                };
                std::string msg = "data: " + delta.dump() + "\n\n";
                sink.write(msg.data(), msg.size());
            };
            auto result = engine_->generate(greq);
            sink.write("data: [DONE]\n\n", 14);
            sink.done();
            spdlog::debug("[OpenAI] completions stream done  "
                          "prompt={} gen={} {:.1f}ms",
                          result.prompt_tokens, result.generated_tokens, result.time_ms);
            return true;
        });
    } else {
        // ── Tool call loop (non-streaming) ────────────────────────────────────
        const int kMaxToolRounds = cfg_.plugins.tool_max_rounds;
        int tool_round = 0;
        GenerateResult result;

        while (tool_round <= kMaxToolRounds) {
            result = engine_->generate(greq);
            if (result.error != ErrorCode::Ok) {
                spdlog::error("[OpenAI] completions generate failed  error={} msg='{}' "
                              "prompt_tokens={} tool_round={}",
                              static_cast<int>(result.error),
                              result.error_message,
                              result.prompt_tokens,
                              tool_round);
                res.status = 503;
                res.set_content(
                    error_response(503, "Inference engine error: " + result.error_message).dump(),
                    "application/json");
                return;
            }

            if (!plugin_bridge_ || !plugin_bridge_->has_tools()) break;

            auto tool_calls = parse_tool_calls(result.text);
            if (tool_calls.empty()) break;

            spdlog::debug("[OpenAI] completions tool_calls found  round={} n={}",
                          tool_round, tool_calls.size());

            // Build a follow-up prompt appending the tool results.
            std::string tool_results_block;
            for (const auto& call : tool_calls) {
                std::string qname   = call.value("function", json{}).value("name",      "");
                std::string args    = tool_args_string(call);
                std::string call_id = call.value("id", "call_" + std::to_string(tool_round));

                // Allow-list enforcement (§3.3.2).  Refuse before dispatch when
                // the call is not in agent.allowed_tools.
                std::string allow_csv;
                if (ctx) {
                    std::lock_guard<std::mutex> lk(ctx->metadata_mu);
                    auto it = ctx->metadata.find("agent.allowed_tools");
                    if (it != ctx->metadata.end()) allow_csv = it->second;
                }
                if (!is_tool_allowed(qname, allow_csv)) {
                    spdlog::info("[OpenAI/completions] tool '{}' refused by "
                                 "agent.allowed_tools (scope={})",
                                 qname, allow_csv.empty() ? "(none)" : allow_csv);
                    tool_results_block += "\n[Tool " + call_id + " result]: " +
                        format_tool_denial(
                            "tool '" + qname + "' is not in the allow-list "
                            "(agent.allowed_tools)") + "\n";
                    continue;
                }

                spdlog::debug("[OpenAI] dispatching tool '{}'  args_len={}", qname, args.size());
                ToolCallResult tc = plugin_bridge_->dispatch_tool(qname, args, ctx.get());

                std::string body;
                if (tc.error == TRUELLM_OK) {
                    body = tc.payload;
                } else if (tc.error == TRUELLM_ERR_PERMISSION) {
                    body = format_tool_denial(
                        tc.error_msg.empty()
                            ? std::string("tool refused the request")
                            : tc.error_msg);
                } else {
                    body = "[Tool error] " + tc.error_msg;
                }
                tool_results_block += "\n[Tool " + call_id + " result]: " + body + "\n";
            }

            // Append results and regenerate.
            prompt += result.text + tool_results_block;
            greq.tokens   = engine_->tokenize(prompt);
            greq.on_token = nullptr;
            ++tool_round;
        }

        float tps = result.generated_tokens > 0 && result.time_ms > 0
                  ? result.generated_tokens / (result.time_ms / 1000.f) : 0.f;

        spdlog::info("[OpenAI] completions done  prompt={} gen={} {:.0f}ms ({:.1f} tok/s) "
                     "tool_rounds={}",
                     result.prompt_tokens, result.generated_tokens,
                     result.time_ms, tps, tool_round);
        spdlog::debug("[OpenAI] completions response: {}",
                      result.text.size() > 256
                          ? result.text.substr(0, 256) + "..." : result.text);

        json usage = {
            {"prompt_tokens",     result.prompt_tokens},
            {"completion_tokens", result.generated_tokens},
            {"total_tokens",      result.prompt_tokens + result.generated_tokens}
        };
        merge_subcall_usage(usage, ctx.get());
        json resp = {
            {"id",      req_id},
            {"object",  "text_completion"},
            {"created", unix_now()},
            {"model",   model_id},
            {"choices", json::array({
                {{"text",          result.text},
                 {"index",         0},
                 {"finish_reason", result.finish_reason.empty()
                                      ? "stop" : result.finish_reason}}
            })},
            {"usage",   usage}
        };
        res.set_content(resp.dump(), "application/json");
    }
}

// ---------------------------------------------------------------------------
// POST /v1/chat/completions
// ---------------------------------------------------------------------------
void OpenAIRouter::handle_chat_completions(const httplib::Request& req,
                                            httplib::Response&      res)
{
    spdlog::debug("[OpenAI] POST /v1/chat/completions");

    json body;
    try { body = json::parse(req.body); }
    catch (...) {
        spdlog::warn("[OpenAI] /v1/chat/completions — invalid JSON body");
        res.status = 400;
        res.set_content(error_response(400, "Invalid JSON").dump(), "application/json");
        return;
    }

    if (!engine_ || !engine_->is_model_loaded()) {
        spdlog::warn("[OpenAI] /v1/chat/completions — no model loaded");
        res.status = 503;
        res.set_content(error_response(503, "No model loaded").dump(), "application/json");
        return;
    }

    try {

    auto messages  = body.value("messages",    json::array());
    int max_tokens = body.value("max_tokens",  cfg_.sampling.max_tokens);
    float temp     = body.value("temperature", cfg_.sampling.temperature);
    bool do_stream = body.value("stream",      false);

    spdlog::debug("[OpenAI] chat  messages={} max_tokens={} temp={:.2f} stream={}",
                  messages.size(), max_tokens, temp, do_stream);
    spdlog::trace("[OpenAI] chat  messages: {}", trace_json(messages));

    // Apply chat template — auto-detect from GGUF architecture when not explicit.
    std::string arch = engine_->get_model_info().architecture;
    std::string tmpl = resolve_template(cfg_.model.chat_template, arch);
    std::string prompt = apply_chat_template(messages, tmpl);

    spdlog::debug("[OpenAI] chat  template={} arch={} prompt_chars={}", tmpl, arch, prompt.size());
    spdlog::trace("[OpenAI] chat  formatted prompt:\n{}", prompt);

    auto t_tok = std::chrono::steady_clock::now();
    auto tokens = engine_->tokenize(prompt);
    spdlog::debug("[OpenAI] tokenize -> {} tokens ({:.1f}ms)", tokens.size(),
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t_tok).count() / 1000.0);

    GenerateRequest greq;
    greq.tokens      = tokens;
    greq.max_tokens  = max_tokens;
    greq.temperature = temp;
    greq.top_p       = body.value("top_p", cfg_.sampling.top_p);

    if (body.contains("stop")) {
        const auto& sv = body["stop"];
        if (sv.is_string())      greq.stop.push_back(sv.get<std::string>());
        else if (sv.is_array())  for (const auto& s : sv) if (s.is_string()) greq.stop.push_back(s.get<std::string>());
    }

    std::string req_id   = "chatcmpl-" + std::to_string(unix_now());
    std::string model_id = cfg_.model.alias.empty() ? "default" : cfg_.model.alias;
    greq.request_id      = req_id;
    // §4.2 — surface the client's session.id (when supplied) to the
    // scheduler so KV-prefix reuse can fire across turns of the same
    // chat session.  We resolve it from the body / header channels
    // BEFORE absorb_client_metadata so the value is available even
    // when ctx isn't built yet (plugin_bridge_ might be off).
    {
        auto pick_sid = [&]() -> std::string {
            try {
                if (body.contains("truellm_metadata")
                    && body["truellm_metadata"].is_object()
                    && body["truellm_metadata"].contains("session.id")
                    && body["truellm_metadata"]["session.id"].is_string())
                    return body["truellm_metadata"]["session.id"].get<std::string>();
                if (body.contains("extra_body")
                    && body["extra_body"].is_object()
                    && body["extra_body"].contains("truellm")
                    && body["extra_body"]["truellm"].is_object()
                    && body["extra_body"]["truellm"].contains("context_metadata")
                    && body["extra_body"]["truellm"]["context_metadata"].is_object())
                {
                    const auto& cm = body["extra_body"]["truellm"]["context_metadata"];
                    if (cm.contains("session.id") && cm["session.id"].is_string())
                        return cm["session.id"].get<std::string>();
                }
            } catch (...) {}
            const auto hdr = req.get_header_value("x-truellm-context-metadata");
            if (!hdr.empty()) {
                try {
                    auto j = json::parse(hdr);
                    if (j.is_object() && j.contains("session.id")
                        && j["session.id"].is_string())
                        return j["session.id"].get<std::string>();
                } catch (...) {}
            }
            return {};
        };
        greq.session_id = pick_sid();
    }

    // ── Build per-request plugin context ──────────────────────────────────────
    std::unique_ptr<truellm_context_t> ctx;
    if (plugin_bridge_) {
        ctx = make_context(cfg_, engine_, max_tokens, temp);
        // Align ctx->request_id with greq.request_id so plugin callbacks
        // calling host->inject_tokens land on the same key the engine uses
        // to track active sessions.
        ctx->request_id = req_id;
        absorb_client_metadata(ctx.get(), body, req);
    }
    // RAII: at the end of the chat-completions request — by any exit path —
    // surface the federated compaction tier's per-request decisions to the
    // /truellm/v1/chat/status endpoint (Code_Capability_design.md §4.4).
    //
    // Capture &ctx (the unique_ptr) rather than ctx.get() so the streaming-
    // generator path's `ctx.release()` (line ~1011) cleanly disables the
    // recorder — that path transfers ownership into a chunked-content
    // provider lambda, where the actual metadata is populated; the lambda
    // is responsible for calling record_chat_outcome itself.
    struct ChatOutcomeRecorder {
        PluginBridge* bridge;
        std::unique_ptr<truellm_context_t>* ctx_holder;
        ~ChatOutcomeRecorder() {
            if (bridge && ctx_holder && *ctx_holder) {
                bridge->record_chat_outcome(ctx_holder->get());
                bridge->record_research_outcome(ctx_holder->get());
            }
        }
    } chat_outcome_recorder{ plugin_bridge_, &ctx };

    // ── Preprocessor chain ────────────────────────────────────────────────────
    if (plugin_bridge_ && plugin_bridge_->has_preprocessors()) {
        auto pr = plugin_bridge_->run_preprocessors(messages.dump(), ctx.get());
        if (pr.decision == TRUELLM_PREPROC_RESPOND) {
            json resp = {
                {"id",      req_id},
                {"object",  "chat.completion"},
                {"created", unix_now()},
                {"model",   model_id},
                {"choices", json::array({
                    {{"message",      {{"role", "assistant"}, {"content", pr.response_text}}},
                     {"index",        0},
                     {"finish_reason","stop"}}
                })}
            };
            res.set_content(resp.dump(), "application/json");
            return;
        }
        if (pr.decision == TRUELLM_PREPROC_ABORT) {
            res.status = 400;
            res.set_content(error_response(400, "Request blocked by preprocessor").dump(),
                            "application/json");
            return;
        }
    }

    // ── Detect explicit tool intent ──────────────────────────────────────────
    // When the client passes `tools` in the request body, or sets
    // tool_choice="required"/"any", we should NOT route the request through
    // a generator hook even if rlm_eligible=1: the generator's compressed
    // context interferes with the model's ability to emit a structured
    // tool-call JSON.  The flag is consulted post-injection where the
    // generator dispatch lives.
    bool client_requested_tools = false;
    {
        const auto tc_raw = body.value("tool_choice", json("auto"));
        std::string tc = tc_raw.is_string() ? tc_raw.get<std::string>() : "auto";
        const auto ct   = body.value("tools", json::array());
        client_requested_tools = (!ct.empty()) || (tc == "required" || tc == "any");
    }

    // ── Tool schema injection ─────────────────────────────────────────────────
    // Priority: client-provided tools (request body) > plugin-bridge registry.
    // The schema is MERGED into the existing system message (or a new system
    // message is prepended if none exists).  A second system block is never
    // created, because LLaMA 3 family models immediately emit EOS when they
    // see two consecutive <|start_header_id|>system<|end_header_id|> blocks.
    //
    // Stash the pre-injection messages so the generator-dispatch path can
    // hand the unbloated form to RLM when injection alone overflows the
    // context window (rlm-suite Check 3 case).  Snapshotting before the
    // mutation costs one O(n) JSON copy per request — small price for a
    // crash-free recovery path when the auto-injected schemas would not
    // fit anyway.
    json   messages_pre_injection = messages;
    bool   tool_schemas_were_injected = false;
    {
        // 1. Collect the schemas to inject.
        json client_tools = body.value("tools", json::array());
        std::string tool_choice = body.value("tool_choice", "auto");
        // tool_choice may also be an object {"type":"function",...}; normalise.
        if (tool_choice.empty()) tool_choice = "auto";

        json all_schemas = json::array();
        if (!client_tools.empty()) {
            for (const auto& t : client_tools)
                if (t.is_object()) all_schemas.push_back(t);
        } else if (plugin_bridge_ && plugin_bridge_->has_tools()) {
            auto s = plugin_bridge_->get_tool_schemas_json();
            if (!s.empty() && s != "[]") {
                auto parsed = json::parse(s, nullptr, false);
                if (!parsed.is_discarded() && parsed.is_array()) {
                    // Strip plugin prefix from function names so the model sees
                    // short names (e.g. "fetch_url" not "tool_web_fetch.fetch_url").
                    // The registry's short-name fallback in dispatch() resolves them back.
                    for (auto& t : parsed) {
                        if (!t.is_object() || !t.contains("function")) continue;
                        auto& fn = t["function"];
                        if (!fn.is_object()) continue;
                        const std::string qn = fn.value("name", "");
                        auto dot = qn.rfind('.');
                        if (dot != std::string::npos)
                            fn["name"] = qn.substr(dot + 1);
                    }
                    all_schemas = parsed;
                }
            }
        }

        if (!all_schemas.empty()) {
            // 2. Build the instruction block.
            std::string tool_instr =
                "You have access to the following tools.\n\n"
                "To call a tool, output ONLY this JSON (no surrounding text):\n"
                "{\"tool_calls\":[{\"id\":\"call_0\",\"type\":\"function\","
                "\"function\":{\"name\":\"TOOL_NAME\",\"arguments\":{\"param\":\"value\"}}}]}\n\n"
                "Available tools (OpenAI format):\n" +
                all_schemas.dump(2) +
                "\n\nOnly use the tool names listed above. "
                "Do not use code_execution, python, browser, or any other tool "
                "not explicitly listed. Do not produce error messages about missing tools.";

            if (tool_choice == "required") {
                tool_instr +=
                    "\n\nIMPORTANT: You MUST call one of the above tools to answer "
                    "this request.  Do not reply with plain text until after you "
                    "have called a tool and received its result.";
            }

            // 3. Merge into the existing system message; never add a second one.
            bool merged = false;
            for (auto& m : messages) {
                if (m.value("role", "") == "system") {
                    m["content"] = tool_instr + "\n\n" + m.value("content", "");
                    merged = true;
                    break;
                }
            }
            if (!merged) {
                json patched = json::array();
                patched.push_back({{"role", "system"}, {"content", tool_instr}});
                for (const auto& m : messages) patched.push_back(m);
                messages = std::move(patched);
            }

            // 4. Rebuild prompt and retokenize.
            prompt      = apply_chat_template(messages, tmpl);
            greq.tokens = engine_->tokenize(prompt);
            tool_schemas_were_injected = true;
            spdlog::debug("[OpenAI] chat  tool schemas injected  "
                          "tools_json_len={} tool_choice={} merged_into_system={}",
                          all_schemas.dump().size(), tool_choice, merged);
        }
    }

    // ── Auto-set rlm_eligible on (post-injection) overflow ───────────────────
    // The token count above already reflects any tool-schema bloat, so this
    // catches the case where the *original* user request fit but the
    // injected schemas pushed us past the context window.  The plugin's
    // own on_claims() also implements a defence-in-depth overflow check.
    //
    // When injection itself caused the overflow, we ALSO revert messages /
    // greq.tokens back to their pre-injection form before handing them to
    // the generator.  The model could not have used the auto-injected
    // tools anyway (the engine would crash on the bloated prompt), and
    // RLM's reconstruct_context preserves the system message verbatim — so
    // without the revert RLM would just re-emit the same multi-thousand-
    // token tool schema and the final inference would loop into the same
    // overflow.
    if (!client_requested_tools
        && ctx && plugin_bridge_ && plugin_bridge_->has_generators()) {
        // The configured context_size in [inference] is the engine's actual
        // serving budget — the GGUF-declared `model_info.context_length` may
        // be larger (32 768 on Nemotron-3-Nano, for example) and would let
        // overflow slip through the check.  batch_scheduler enforces the
        // configured value, so we must too.
        const int64_t ctx_len   = static_cast<int64_t>(cfg_.inference.context_size);
        const int32_t reserved  = std::max<int32_t>(max_tokens, 0);
        const int64_t budget    = ctx_len > 0 ? (ctx_len - reserved) : 0;
        const int64_t n_tokens  = static_cast<int64_t>(greq.tokens.size());
        if (ctx_len > 0 && n_tokens > budget) {
            {
                std::lock_guard<std::mutex> lk(ctx->metadata_mu);
                ctx->metadata["rlm_eligible"]    = "1";
                ctx->metadata["overflow_tokens"] = std::to_string(n_tokens);
                ctx->metadata["context_budget"]  = std::to_string(budget);
            }
            spdlog::info("[OpenAI] rlm_eligible=1 (post-injection prompt={} "
                         "toks > budget={} [ctx_size={} max_tokens={}])",
                         n_tokens, budget, ctx_len, reserved);

            if (tool_schemas_were_injected) {
                messages    = std::move(messages_pre_injection);
                prompt      = apply_chat_template(messages, tmpl);
                greq.tokens = engine_->tokenize(prompt);
                spdlog::info("[OpenAI] reverted tool-schema injection for "
                             "generator path (n_tokens now={})",
                             greq.tokens.size());
            }
        }
    }

    // ── Generator dispatch — priority-ordered claim-and-run ──────────────────
    // Runs AFTER tool-schema injection so the auto-overflow check above sees
    // the bloated prompt, AND so any system-message tools the client sees
    // remain available to the generator if it preserves them.
    //
    // Skipped entirely when the client explicitly requested tools via the
    // request body (`tools` non-empty or `tool_choice` ∈ {required, any}).
    // RLM-style compression makes structured tool-call emission unreliable;
    // honouring the client's explicit "use tools" intent takes priority.
    if (!client_requested_tools
        && plugin_bridge_ && plugin_bridge_->has_generators()) {
        if (do_stream) {
            bool claimed = plugin_bridge_->generator_claims(messages.dump(),
                                                            ctx.get());
            if (claimed) {
                spdlog::info("[OpenAI] chat  generator claimed streaming request "
                             "req_id={}", req_id);
                res.set_chunked_content_provider("text/event-stream",
                    [this, messages_dump = messages.dump(),
                     ctx_ptr = ctx.release(), req_id, model_id,
                     completed = false]
                    (std::size_t /*offset*/, httplib::DataSink& sink) mutable -> bool
                {
                    if (completed) return false;
                    completed = true;
                    std::unique_ptr<truellm_context_t> owned(ctx_ptr);
                    ctx_ptr = nullptr;

                    json first = {
                        {"id",      req_id},
                        {"object",  "chat.completion.chunk"},
                        {"model",   model_id},
                        {"choices", json::array({
                            {{"delta",        {{"role", "assistant"}}},
                             {"index",        0},
                             {"finish_reason", nullptr}}
                        })}
                    };
                    std::string first_msg = "data: " + first.dump() + "\n\n";
                    sink.write(first_msg.data(), first_msg.size());

                    struct StreamCtx {
                        httplib::DataSink* sink;
                        std::string        req_id;
                        std::string        model_id;
                        std::string        utf8_carry;
                    } sc{ &sink, req_id, model_id, {} };

                    auto on_token = [](void* cb, const char* tok, int /*done*/) {
                        if (!tok || !*tok) return;
                        auto* st = static_cast<StreamCtx*>(cb);
                        st->utf8_carry += tok;
                        std::string emit = sanitize_utf8(st->utf8_carry);
                        st->utf8_carry.erase(0, emit.size());
                        if (emit.empty()) return;
                        json chunk = {
                            {"id",      st->req_id},
                            {"object",  "chat.completion.chunk"},
                            {"model",   st->model_id},
                            {"choices", json::array({
                                {{"delta",        {{"content", emit}}},
                                 {"index",        0},
                                 {"finish_reason", nullptr}}
                            })}
                        };
                        std::string msg = "data: " + chunk.dump() + "\n\n";
                        st->sink->write(msg.data(), msg.size());
                    };

                    auto outcome = plugin_bridge_->try_dispatch_generator(
                        messages_dump, owned.get(), on_token, &sc);

                    if (!outcome.ok) {
                        spdlog::warn("[OpenAI] chat  generator stream failed "
                                     "plugin='{}' err={} msg='{}'",
                                     outcome.plugin_id,
                                     static_cast<int>(outcome.error),
                                     outcome.error_message);
                    } else {
                        spdlog::info("[OpenAI] chat  generator stream done "
                                     "plugin='{}' chars={}",
                                     outcome.plugin_id, outcome.text.size());
                    }
                    sink.write("data: [DONE]\n\n", 14);
                    sink.done();
                    // §4.4: record per-request outcomes now — the outer
                    // ChatOutcomeRecorder is disabled on this path because
                    // we released ctx into this lambda above.
                    if (plugin_bridge_ && owned) {
                        plugin_bridge_->record_chat_outcome(owned.get());
                        plugin_bridge_->record_research_outcome(owned.get());
                    }
                    return true;
                });
                return;
            }
        } else {
            auto outcome = plugin_bridge_->try_dispatch_generator(
                messages.dump(), ctx.get(), nullptr, nullptr);
            if (outcome.claimed) {
                if (!outcome.ok) {
                    spdlog::error("[OpenAI] chat  generator '{}' failed "
                                  "err={} msg='{}'",
                                  outcome.plugin_id,
                                  static_cast<int>(outcome.error),
                                  outcome.error_message);
                    res.status = 503;
                    res.set_content(
                        error_response(503, "Generator '" + outcome.plugin_id +
                                             "' error: " + outcome.error_message).dump(),
                        "application/json");
                    return;
                }

                spdlog::info("[OpenAI] chat  generator '{}' produced {} chars",
                             outcome.plugin_id, outcome.text.size());

                const int completion_tokens =
                    outcome.text.empty() ? 0
                                         : static_cast<int>(
                                             engine_->tokenize(outcome.text).size());
                const int prompt_tokens = static_cast<int>(greq.tokens.size());
                json usage = {
                    {"prompt_tokens",     prompt_tokens},
                    {"completion_tokens", completion_tokens},
                    {"total_tokens",      prompt_tokens + completion_tokens}
                };
                merge_subcall_usage(usage, ctx.get());
                // §9.5: emit "length" when the generator's output was
                // capped by the requested max_tokens; the plugin C ABI
                // doesn't surface this directly, so we infer it from
                // completion_tokens vs the request's max_tokens.
                const char* fr = (max_tokens > 0
                                  && completion_tokens >= max_tokens)
                                 ? "length" : "stop";
                json resp = {
                    {"id",      req_id},
                    {"object",  "chat.completion"},
                    {"created", unix_now()},
                    {"model",   model_id},
                    {"choices", json::array({
                        {{"message",      {{"role",    "assistant"},
                                           {"content", sanitize_utf8(outcome.text)}}},
                         {"index",        0},
                         {"finish_reason", fr}}
                    })},
                    {"usage",   usage}
                };
                res.set_content(resp.dump(), "application/json");
                return;
            }
        }
    } else if (client_requested_tools && plugin_bridge_
               && plugin_bridge_->has_generators()) {
        spdlog::debug("[OpenAI] chat  generator dispatch skipped "
                      "(client_requested_tools=true)");
    }

    if (do_stream) {
        spdlog::debug("[OpenAI] chat stream starting  req_id={}", req_id);
        res.set_chunked_content_provider("text/event-stream",
            [this, greq = std::move(greq), req_id, model_id]
            (std::size_t /*offset*/, httplib::DataSink& sink) mutable -> bool
        {
            // Send role delta first
            json first = {
                {"id",      req_id},
                {"object",  "chat.completion.chunk"},
                {"model",   model_id},
                {"choices", json::array({
                    {{"delta",        {{"role", "assistant"}}},
                     {"index",        0},
                     {"finish_reason", nullptr}}
                })}
            };
            std::string first_msg = "data: " + first.dump() + "\n\n";
            sink.write(first_msg.data(), first_msg.size());

            int n_tok = 0;
            // Per-stream UTF-8 carry buffer: BPE pieces may split a multibyte
            // character across two on_token calls.  We emit only bytes up to
            // the last complete code point, and stash the trailing partial
            // sequence for the next piece.
            greq.on_token = [&, utf8_carry = std::string()](int32_t id,
                                                            const std::string& piece) mutable {
                ++n_tok;
                spdlog::trace("[OpenAI] stream tok[{}] id={} piece={}", n_tok, id,
                              piece.empty() ? "<empty>" : piece);
                utf8_carry += piece;
                std::string emit = sanitize_utf8(utf8_carry);
                utf8_carry.erase(0, emit.size());
                if (emit.empty()) return;   // waiting on more bytes
                json chunk = {
                    {"id",      req_id},
                    {"object",  "chat.completion.chunk"},
                    {"model",   model_id},
                    {"choices", json::array({
                        {{"delta",        {{"content", emit}}},
                         {"index",        0},
                         {"finish_reason", nullptr}}
                    })}
                };
                std::string msg = "data: " + chunk.dump() + "\n\n";
                sink.write(msg.data(), msg.size());
            };
            auto result = engine_->generate(greq);
            sink.write("data: [DONE]\n\n", 14);
            sink.done();

            float tps = result.generated_tokens > 0 && result.time_ms > 0
                      ? result.generated_tokens / (result.time_ms / 1000.f) : 0.f;
            spdlog::info("[OpenAI] chat stream done  "
                         "prompt={} gen={} {:.0f}ms ({:.1f} tok/s)",
                         result.prompt_tokens, result.generated_tokens,
                         result.time_ms, tps);
            return true;
        });
    } else {
        // ── Tool call loop ────────────────────────────────────────────────────
        // Runs until model produces no tool_calls or max rounds is reached.
        const int kMaxToolRounds = cfg_.plugins.tool_max_rounds;
        int tool_round = 0;
        GenerateResult result;

        while (tool_round <= kMaxToolRounds) {
            result = engine_->generate(greq);
            if (result.error != ErrorCode::Ok) {
                spdlog::error("[OpenAI] chat generate failed  error={} msg='{}' "
                              "prompt_tokens={} tool_round={}",
                              static_cast<int>(result.error),
                              result.error_message,
                              result.prompt_tokens,
                              tool_round);
                res.status = 503;
                res.set_content(
                    error_response(503, "Inference engine error: " + result.error_message).dump(),
                    "application/json");
                return;
            }

            // Only attempt tool dispatch if a plugin bridge is wired.
            if (!plugin_bridge_ || !plugin_bridge_->has_tools()) break;

            auto tool_calls = parse_tool_calls(result.text);
            if (tool_calls.empty()) break;  // plain response — done

            spdlog::debug("[OpenAI] chat tool_calls found  round={} n={}",
                          tool_round, tool_calls.size());

            // Append assistant turn.  Include result.text as "content" so that
            // apply_chat_template() can reconstruct the prompt faithfully — an
            // empty content field produces a blank assistant turn which confuses
            // the model on the next inference pass.
            messages.push_back({
                {"role",       "assistant"},
                {"content",    result.text},
                {"tool_calls", tool_calls},
            });

            // Execute each tool call and append its result.
            for (const auto& call : tool_calls) {
                std::string qname   = call.value("function", json{}).value("name",      "");
                std::string args    = tool_args_string(call);
                std::string call_id = call.value("id", "call_" + std::to_string(tool_round));

                // §3.3.2 / §3.0 — honour the per-request agent.allowed_tools
                // contract set by plan_mode_guard or fork_agent_pool.  Refuse
                // the call before it reaches the plugin so the engine, not
                // the agent harness, owns enforcement.
                std::string allow_csv;
                if (ctx) {
                    std::lock_guard<std::mutex> lk(ctx->metadata_mu);
                    auto it = ctx->metadata.find("agent.allowed_tools");
                    if (it != ctx->metadata.end()) allow_csv = it->second;
                }
                if (!is_tool_allowed(qname, allow_csv)) {
                    spdlog::info("[OpenAI] tool '{}' refused by agent.allowed_tools "
                                 "(scope={})", qname,
                                 allow_csv.empty() ? "(none)" : allow_csv);
                    messages.push_back({
                        {"role",         "tool"},
                        {"tool_call_id", call_id},
                        {"content",      format_tool_denial(
                            "tool '" + qname + "' is not in the allow-list "
                            "(agent.allowed_tools).  Use a different tool or "
                            "ask the user to widen the scope.")}
                    });
                    continue;
                }

                spdlog::debug("[OpenAI] dispatching tool '{}'  args_len={}", qname, args.size());
                ToolCallResult tc = plugin_bridge_->dispatch_tool(qname, args, ctx.get());

                std::string body;
                if (tc.error == TRUELLM_OK) {
                    body = tc.payload;
                } else if (tc.error == TRUELLM_ERR_PERMISSION) {
                    body = format_tool_denial(
                        tc.error_msg.empty()
                            ? std::string("tool refused the request")
                            : tc.error_msg);
                } else {
                    body = "[Tool error] " + tc.error_msg;
                }
                messages.push_back({
                    {"role",         "tool"},
                    {"tool_call_id", call_id},
                    {"content",      body}
                });
            }

            // ── Strip tool_choice=required after first dispatch ───────────────
            // The "MUST call a tool" directive stays in the system message from
            // the initial injection.  If it is still present on the second+
            // inference pass, the model will generate another tool call instead
            // of a text answer.  Remove it now that at least one tool result
            // has been injected.
            {
                const std::string required_suffix =
                    "\n\nIMPORTANT: You MUST call one of the above tools to answer "
                    "this request.  Do not reply with plain text until after you "
                    "have called a tool and received its result.";
                for (auto& m : messages) {
                    if (m.value("role", "") != "system") continue;
                    std::string c = m.value("content", "");
                    auto pos = c.find(required_suffix);
                    if (pos != std::string::npos) {
                        c.erase(pos, required_suffix.size());
                        m["content"] = c;
                        spdlog::debug("[OpenAI] stripped tool_choice=required from system "
                                      "message before second inference pass");
                    }
                    break;
                }
            }

            // ── Context splitter ─────────────────────────────────────────────
            // If any tool result is too large to fit in the context window,
            // reduce it before rebuilding the prompt.
            //
            // strategy = "summarize" (default):
            //   Split into chunks, run a per-chunk summarisation inference pass,
            //   replace the tool message with the concatenated summaries.
            //
            // strategy = "truncate":
            //   Hard-clip the tool content to fit the available token budget
            //   without running any extra inference.  Fast and deterministic.
            {
                const int64_t budget = engine_->get_token_budget();
                const bool    strategy_truncate =
                    cfg_.plugins.context_split_strategy == "truncate";

                if (budget > 0 && cfg_.plugins.context_split_enabled) {
                    for (auto& m : messages) {
                        if (m.value("role", "") != "tool") continue;
                        std::string content = m.value("content", "");

                        // Only act if the tool content alone is large enough
                        // to plausibly overflow the context.
                        auto ctoks = engine_->tokenize(content);
                        const int64_t trigger_threshold =
                            static_cast<int64_t>(budget * cfg_.plugins.context_split_trigger_fraction);
                        if (static_cast<int64_t>(ctoks.size()) < trigger_threshold) continue;

                        spdlog::info("[OpenAI] context-split triggered: "
                                     "strategy={} tool_toks={} budget={} threshold={}",
                                     cfg_.plugins.context_split_strategy,
                                     ctoks.size(), budget, trigger_threshold);

                        if (strategy_truncate) {
                            // Clip to truncate_fraction of budget (chars ≈ toks * 3).
                            const int keep_toks = static_cast<int>(
                                budget * cfg_.plugins.context_split_truncate_fraction);
                            const std::size_t keep_chars =
                                static_cast<std::size_t>(keep_toks) * 3;
                            if (content.size() > keep_chars) {
                                content.resize(keep_chars);
                                // Trim to last newline so we don't cut mid-word.
                                auto nl = content.rfind('\n');
                                if (nl != std::string::npos) content.resize(nl + 1);
                                content += "\n[... content truncated to fit context window ...]";
                            }
                            m["content"] = content;
                        } else {
                            // Summarise: split into chunks and run per-chunk inference.
                            int chunk_toks = cfg_.plugins.context_split_chunk_tokens > 0
                                           ? cfg_.plugins.context_split_chunk_tokens
                                           : static_cast<int>(budget * cfg_.plugins.context_split_auto_chunk_fraction);
                            const std::size_t chunk_chars =
                                static_cast<std::size_t>(chunk_toks) * 3;

                            auto chunks  = chunk_content(content, chunk_chars);
                            std::string summary = summarize_chunks(
                                chunks, extract_user_question(messages), tmpl,
                                cfg_.plugins.context_split_summary_tokens,
                                greq.temperature);
                            m["content"] = "[Summarised tool output]:\n" + summary;
                        }
                    }
                }
            }

            // Rebuild prompt with full conversation and retokenize.
            std::string updated = apply_chat_template(messages, tmpl);
            greq.tokens   = engine_->tokenize(updated);
            greq.on_token = nullptr;

            // Final safety net: if still over budget (e.g. splitting didn't
            // help enough), replace the last tool message with a short notice
            // and run one final inference so the model can produce a text answer.
            // IMPORTANT: do NOT just break here — result still holds the previous
            // tool call JSON; we must generate a fresh response with the trimmed prompt.
            {
                const int64_t budget = engine_->get_token_budget();
                if (budget > 0 && static_cast<int64_t>(greq.tokens.size()) >= budget) {
                    spdlog::warn("[OpenAI] prompt still {} toks >= budget {} after "
                                 "context-split — hard-truncating last tool message",
                                 greq.tokens.size(), budget);
                    for (int mi = static_cast<int>(messages.size()) - 1; mi >= 0; --mi) {
                        if (messages[mi].value("role","") == "tool") {
                            messages[mi]["content"] =
                                "[Tool result omitted: response too large for context window. "
                                "Please answer based on what you know about the requested URL.]";
                            break;
                        }
                    }
                    updated     = apply_chat_template(messages, tmpl);
                    greq.tokens = engine_->tokenize(updated);

                    // Run the final inference and use its output as the answer.
                    result = engine_->generate(greq);
                    ++tool_round;
                    break;
                }
            }

            ++tool_round;
        }

        float tps = result.generated_tokens > 0 && result.time_ms > 0
                  ? result.generated_tokens / (result.time_ms / 1000.f) : 0.f;

        spdlog::info("[OpenAI] chat done  prompt={} gen={} {:.0f}ms ({:.1f} tok/s) tool_rounds={}",
                     result.prompt_tokens, result.generated_tokens,
                     result.time_ms, tps, tool_round);
        spdlog::debug("[OpenAI] chat response: {}",
                      result.text.size() > 256
                          ? result.text.substr(0, 256) + "…" : result.text);

        json usage = {
            {"prompt_tokens",     result.prompt_tokens},
            {"completion_tokens", result.generated_tokens},
            {"total_tokens",      result.prompt_tokens + result.generated_tokens}
        };
        merge_subcall_usage(usage, ctx.get());
        json resp = {
            {"id",      req_id},
            {"object",  "chat.completion"},
            {"created", unix_now()},
            {"model",   model_id},
            {"choices", json::array({
                {{"message",      {{"role",    "assistant"},
                                   {"content", sanitize_utf8(result.text)}}},
                 {"index",        0},
                 {"finish_reason", result.finish_reason.empty()
                                      ? "stop" : result.finish_reason}}
            })},
            {"usage",   usage}
        };
        res.set_content(resp.dump(), "application/json");
    }
    } catch (const std::exception& ex) {
        spdlog::error("[OpenAI] /v1/chat/completions — unhandled exception: {}", ex.what());
        res.status = 500;
        res.set_content(
            error_response(500, std::string("Internal server error: ") + ex.what()).dump(),
            "application/json");
    } catch (...) {
        spdlog::error("[OpenAI] /v1/chat/completions — unknown exception");
        res.status = 500;
        res.set_content(
            error_response(500, "Internal server error (unknown exception)").dump(),
            "application/json");
    }
}

} // namespace truellm
