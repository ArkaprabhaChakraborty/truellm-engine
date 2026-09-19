// ---------------------------------------------------------------------------
// anthropic_router.cpp — native Anthropic Messages API (Code_Capability_design.md §12)
//
// POST /v1/messages              — Anthropic Messages API
// POST /v1/messages/count_tokens — token-count probe Claude Code calls per turn
//
// The router translates the Anthropic request to the OpenAI chat-completions
// shape, delegates to OpenAIRouter::run_chat_completions (the full existing
// preprocessor + generate + tool pipeline), and translates the response back
// to an Anthropic `Message`.  No inference logic is duplicated here.
// ---------------------------------------------------------------------------

#include "anthropic_router.h"
#include "chat_template.h"
#include "../inference/engine_interface.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <atomic>
#include <chrono>
#include <string>
#include <utility>
#include <vector>

namespace truellm {

using json = nlohmann::json;

namespace {

long long unix_now() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch()).count();
}

// Opaque ids: "msg_…" for messages, "req_…" for the request-id header.
// Process-local uniqueness is all that matters.
std::string new_id(const char* prefix) {
    static std::atomic<unsigned long long> seq{0};
    return std::string(prefix) + std::to_string(unix_now()) + "_" +
           std::to_string(seq.fetch_add(1));
}
std::string new_msg_id() { return new_id("msg_"); }

// An Anthropic error envelope.
json anthropic_error(const std::string& type, const std::string& message) {
    return {{"type", "error"},
            {"error", {{"type", type}, {"message", message}}}};
}

// ── System-prompt composition (§12.5) ──────────────────────────────────────
// Anthropic `system` + truellm.system_prompt + a rendered agents block +
// a rendered skills block.  This is how Claude Code itself surfaces agents
// and skills: progressive disclosure through the system prompt.
std::string compose_system(const json& areq) {
    std::string sys;

    auto append = [&](const std::string& s) {
        if (s.empty()) return;
        if (!sys.empty()) sys += "\n\n";
        sys += s;
    };

    // 1. Standard Anthropic `system` field (string or [{type:text,text}]).
    if (areq.contains("system")) {
        const json& s = areq["system"];
        if (s.is_string()) {
            append(s.get<std::string>());
        } else if (s.is_array()) {
            std::string joined;
            for (const auto& b : s) {
                if (b.value("type", "") == "text") {
                    if (!joined.empty()) joined += "\n";
                    joined += b.value("text", "");
                }
            }
            append(joined);
        }
    }

    // 2. TrueLLM per-request capability extension.
    if (areq.contains("truellm") && areq["truellm"].is_object()) {
        const json& t = areq["truellm"];

        if (t.contains("system_prompt") && t["system_prompt"].is_string())
            append(t["system_prompt"].get<std::string>());

        if (t.contains("agents") && t["agents"].is_array()
            && !t["agents"].empty()) {
            std::string block = "# Available agents\n"
                "Specialised agents you may delegate focused sub-tasks to. "
                "Each has its own remit; use the closest match.\n";
            for (const auto& a : t["agents"]) {
                block += "\n## " + a.value("name", "agent") + "\n";
                if (a.contains("description"))
                    block += a.value("description", "") + "\n";
                if (a.contains("prompt"))
                    block += a.value("prompt", "") + "\n";
            }
            append(block);
        }

        if (t.contains("skills") && t["skills"].is_array()
            && !t["skills"].empty()) {
            std::string block = "# Available skills\n"
                "Skills extend what you can do.  Apply a skill's "
                "instructions when its description matches the task.\n";
            for (const auto& sk : t["skills"]) {
                block += "\n## " + sk.value("name", "skill") + "\n";
                if (sk.contains("description"))
                    block += sk.value("description", "") + "\n";
                if (sk.contains("instructions"))
                    block += sk.value("instructions", "") + "\n";
                // §12.8 Phase 3 — when a skill scopes the tool surface,
                // disclose that the offered tools are intentionally limited.
                if (sk.contains("allowed_tools")
                    && !sk["allowed_tools"].is_null()) {
                    block += "While this skill is active only its permitted "
                             "tools are offered; do not expect others.\n";
                }
            }
            append(block);
        }
    }

    return sys;
}

// Flatten one Anthropic content value (string | array of blocks) to plain
// text.  tool_use / tool_result blocks are handled by translate_messages;
// here `image` blocks become a short placeholder (Phase 1 — vision wiring
// is §12.8 Phase 2).
std::string blocks_to_text(const json& content) {
    if (content.is_string()) return content.get<std::string>();
    if (!content.is_array()) return "";
    std::string text;
    for (const auto& blk : content) {
        const std::string bt = blk.value("type", "");
        if (bt == "text") {
            if (!text.empty()) text += "\n";
            text += blk.value("text", "");
        } else if (bt == "image") {
            if (!text.empty()) text += "\n";
            text += "[image]";
        }
    }
    return text;
}

// ── Message translation (Anthropic → OpenAI) ────────────────────────────────
// Anthropic folds tool results into user turns and tool calls into assistant
// turns.  OpenAI keeps tool results as standalone role:"tool" messages and
// tool calls on the assistant message.  We unfold accordingly.
json translate_messages(const json& amsgs) {
    json omsgs = json::array();
    if (!amsgs.is_array()) return omsgs;

    for (const auto& m : amsgs) {
        const std::string role = m.value("role", "user");
        const json content = m.contains("content") ? m["content"]
                                                    : json("");

        if (content.is_string()) {
            omsgs.push_back({{"role", role},
                             {"content", content.get<std::string>()}});
            continue;
        }
        if (!content.is_array()) {
            omsgs.push_back({{"role", role}, {"content", ""}});
            continue;
        }

        std::string text;
        json tool_calls   = json::array();   // assistant tool_use blocks
        json tool_results = json::array();   // user tool_result blocks

        for (const auto& blk : content) {
            const std::string bt = blk.value("type", "");
            if (bt == "text") {
                if (!text.empty()) text += "\n";
                text += blk.value("text", "");
            } else if (bt == "image") {
                if (!text.empty()) text += "\n";
                text += "[image]";
            } else if (bt == "tool_use") {
                json input = blk.contains("input") ? blk["input"]
                                                   : json::object();
                tool_calls.push_back({
                    {"id",   blk.value("id", "")},
                    {"type", "function"},
                    {"function", {
                        {"name",      blk.value("name", "")},
                        {"arguments", input.dump()},
                    }},
                });
            } else if (bt == "tool_result") {
                tool_results.push_back({
                    {"role",         "tool"},
                    {"tool_call_id", blk.value("tool_use_id", "")},
                    {"content",      blocks_to_text(
                        blk.contains("content") ? blk["content"] : json(""))},
                });
            }
        }

        // Tool results answer the *previous* assistant turn — emit them
        // first, as standalone role:"tool" messages.
        for (auto& tr : tool_results) omsgs.push_back(tr);

        if (role == "assistant" && !tool_calls.empty()) {
            omsgs.push_back({{"role", "assistant"},
                             {"content", text},
                             {"tool_calls", tool_calls}});
        } else if (!text.empty() || tool_results.empty()) {
            // Emit the turn's text.  Skip only when this user turn was
            // purely tool_result blocks (already emitted above).
            omsgs.push_back({{"role", role}, {"content", text}});
        }
    }
    return omsgs;
}

// Anthropic tools  → OpenAI tools.
json translate_tools(const json& atools) {
    json otools = json::array();
    if (!atools.is_array()) return otools;
    for (const auto& t : atools) {
        otools.push_back({
            {"type", "function"},
            {"function", {
                {"name",        t.value("name", "")},
                {"description", t.value("description", "")},
                {"parameters",  t.contains("input_schema") &&
                                t["input_schema"].is_object()
                                    ? t["input_schema"]
                                    : json::object()},
            }},
        });
    }
    return otools;
}

// §12.8 — per-request agents (`truellm.agents`) become dispatchable
// function tools.  The model can call `agent_<name>`; because the
// Anthropic path runs in client_tools mode, that call is returned to the
// caller as a `tool_use` block and the agent harness runs the sub-agent
// (§5.3 — fork / sub-agent execution is the harness's job, not the
// engine's).  This is exactly how Claude Code surfaces its own subagents.
json agent_tools(const json& areq) {
    json out = json::array();
    if (!areq.contains("truellm") || !areq["truellm"].is_object())
        return out;
    const json agents = areq["truellm"].value("agents", json::array());
    if (!agents.is_array()) return out;
    for (const auto& a : agents) {
        const std::string name = a.value("name", "");
        if (name.empty()) continue;
        out.push_back({
            {"type", "function"},
            {"function", {
                {"name", "agent_" + name},
                {"description",
                 "Delegate a focused, self-contained sub-task to the '" +
                 name + "' agent. " + a.value("description", "")},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"prompt", {
                            {"type", "string"},
                            {"description",
                             "Self-contained task description for the agent"}}}}},
                    {"required", json::array({"prompt"})}}},
            }},
        });
    }
    return out;
}

// ── §12.8 Phase 3 — skill-scoped tool gating ────────────────────────────────
// A skill in truellm.skills may declare `allowed_tools`: a string array, or a
// comma-separated string, of glob patterns ('*' matches any run of chars).
// When *any* active skill declares allowed_tools, the request's tool surface
// is narrowed to the union those skills permit — mirroring how a Claude Code
// skill's `allowed-tools` frontmatter scopes the harness.  When no skill
// declares it, gating is inert and every tool is offered.

// Glob match: '*' is the only metacharacter; everything else is literal.
// Iterative wildcard match with backtracking — no recursion, no regex.
bool tool_glob_match(const std::string& pat, const std::string& name) {
    std::size_t p = 0, n = 0;
    std::size_t star = std::string::npos, mark = 0;
    while (n < name.size()) {
        if (p < pat.size() && pat[p] == name[n]) { ++p; ++n; }
        else if (p < pat.size() && pat[p] == '*') { star = p++; mark = n; }
        else if (star != std::string::npos) { p = star + 1; n = ++mark; }
        else return false;
    }
    while (p < pat.size() && pat[p] == '*') ++p;
    return p == pat.size();
}

// Collect the union of allowed-tool patterns declared across truellm.skills.
// An empty result means no skill imposed a restriction → no gating.
std::vector<std::string> skill_tool_patterns(const json& areq) {
    std::vector<std::string> pats;
    if (!areq.contains("truellm") || !areq["truellm"].is_object()) return pats;
    const json& t = areq["truellm"];
    if (!t.contains("skills") || !t["skills"].is_array()) return pats;

    // Split a comma-separated string into trimmed, non-empty tokens.
    auto add_csv = [&](const std::string& raw) {
        std::size_t pos = 0;
        while (pos <= raw.size()) {
            std::size_t end = raw.find(',', pos);
            if (end == std::string::npos) end = raw.size();
            std::size_t a = raw.find_first_not_of(" \t", pos);
            if (a != std::string::npos && a < end) {
                std::size_t b = raw.find_last_not_of(" \t", end - 1);
                std::string tok = raw.substr(a, b - a + 1);
                if (!tok.empty()) pats.push_back(tok);
            }
            if (end == raw.size()) break;
            pos = end + 1;
        }
    };

    for (const auto& sk : t["skills"]) {
        if (!sk.is_object() || !sk.contains("allowed_tools")) continue;
        const json& at = sk["allowed_tools"];
        if (at.is_string()) {
            add_csv(at.get<std::string>());
        } else if (at.is_array()) {
            for (const auto& e : at)
                if (e.is_string()) add_csv(e.get<std::string>());
        }
    }
    return pats;
}

// Filter an OpenAI function-tool array to entries whose function.name matches
// at least one pattern.  An empty `pats` returns `tools` unchanged.
json gate_tools(const json& tools, const std::vector<std::string>& pats) {
    if (pats.empty() || !tools.is_array()) return tools;
    json out = json::array();
    for (const auto& t : tools) {
        const std::string name =
            t.value("function", json::object()).value("name", "");
        for (const auto& p : pats) {
            if (tool_glob_match(p, name)) { out.push_back(t); break; }
        }
    }
    return out;
}

// Build the OpenAI chat-completions request body from the Anthropic request.
json translate_request(const json& areq) {
    json obody;
    obody["model"] = areq.value("model", std::string("default"));

    json omsgs = json::array();
    std::string sys = compose_system(areq);
    if (!sys.empty())
        omsgs.push_back({{"role", "system"}, {"content", sys}});
    for (auto& m : translate_messages(areq.value("messages", json::array())))
        omsgs.push_back(m);
    obody["messages"] = omsgs;

    obody["max_tokens"] = areq.value("max_tokens", 1024);
    if (areq.contains("temperature")) obody["temperature"] = areq["temperature"];
    if (areq.contains("top_p"))       obody["top_p"]       = areq["top_p"];
    if (areq.contains("stop_sequences") && areq["stop_sequences"].is_array())
        obody["stop"] = areq["stop_sequences"];

    // Client tools + per-request agents (exposed as agent_<name> tools)
    // are merged into a single OpenAI tools array, then narrowed to the
    // surface the active skills permit (§12.8 Phase 3 — skill-scoped gating).
    json tools = json::array();
    if (areq.contains("tools") && areq["tools"].is_array())
        tools = translate_tools(areq["tools"]);
    for (auto& at : agent_tools(areq)) tools.push_back(at);
    tools = gate_tools(tools, skill_tool_patterns(areq));

    if (!tools.empty()) {
        obody["tools"] = tools;

        // tool_choice only makes sense when tools survive gating.  If a
        // skill gated every tool away, drop tool_choice too so the engine
        // is not told to "MUST call a tool" with none on offer.
        if (areq.contains("tool_choice") && areq["tool_choice"].is_object()) {
            const std::string tt = areq["tool_choice"].value("type", "auto");
            if (tt == "auto")
                obody["tool_choice"] = "auto";
            else if (tt == "any")
                obody["tool_choice"] = "required";
            else if (tt == "tool")
                obody["tool_choice"] = {
                    {"type", "function"},
                    {"function", {{"name",
                        areq["tool_choice"].value("name", "")}}}};
        }
    }

    // The router always buffers; the OpenAI call is non-streaming.
    obody["stream"] = false;

    // extra_body.truellm — client_tools is mandatory (Anthropic semantics:
    // the caller executes tools), context_metadata is forwarded verbatim.
    json tmeta = json::object();
    if (areq.contains("truellm") && areq["truellm"].is_object()
        && areq["truellm"].contains("context_metadata")
        && areq["truellm"]["context_metadata"].is_object())
        tmeta = areq["truellm"]["context_metadata"];
    // Anthropic clients pass a session handle as metadata.user_id.
    if (!tmeta.contains("session.id")
        && areq.contains("metadata") && areq["metadata"].is_object()
        && areq["metadata"].contains("user_id")
        && areq["metadata"]["user_id"].is_string())
        tmeta["session.id"] = areq["metadata"]["user_id"];

    json truellm_ext = {{"client_tools", true}};
    if (!tmeta.empty()) truellm_ext["context_metadata"] = tmeta;
    obody["extra_body"] = {{"truellm", truellm_ext}};

    return obody;
}

// Build the Anthropic `Message` from the OpenAI chat-completion response.
json translate_response(const json& oresp) {
    json choice = (oresp.contains("choices") && oresp["choices"].is_array()
                   && !oresp["choices"].empty())
                      ? oresp["choices"][0] : json::object();
    json msg = choice.contains("message") ? choice["message"] : json::object();
    const std::string finish    = choice.value("finish_reason", "stop");
    const std::string text      = msg.value("content", "");
    const std::string reasoning = msg.value("reasoning_content", "");

    json content = json::array();
    // Reasoning ("thinking") comes first, mirroring Anthropic's extended
    // thinking ordering.  Populated from the OpenAI message's reasoning_content
    // (the common_chat reasoning split — see chat_format.cpp / openai_router).
    if (!reasoning.empty())
        content.push_back({{"type", "thinking"}, {"thinking", reasoning}});
    if (!text.empty())
        content.push_back({{"type", "text"}, {"text", text}});

    if (msg.contains("tool_calls") && msg["tool_calls"].is_array()) {
        for (const auto& tc : msg["tool_calls"]) {
            json fn = tc.contains("function") ? tc["function"] : json::object();
            json input;
            try { input = json::parse(fn.value("arguments", "{}")); }
            catch (...) { input = json::object(); }
            if (!input.is_object()) input = json::object();
            content.push_back({
                {"type",  "tool_use"},
                {"id",    tc.value("id", "")},
                {"name",  fn.value("name", "")},
                {"input", input},
            });
        }
    }
    if (content.empty())
        content.push_back({{"type", "text"}, {"text", ""}});

    const std::string stop_reason =
        finish == "length"     ? "max_tokens" :
        finish == "tool_calls" ? "tool_use"   :
                                 "end_turn";

    json usage = oresp.contains("usage") ? oresp["usage"] : json::object();

    return {
        {"id",            new_msg_id()},
        {"type",          "message"},
        {"role",          "assistant"},
        {"model",         oresp.value("model", std::string("default"))},
        {"content",       content},
        {"stop_reason",   stop_reason},
        {"stop_sequence", nullptr},
        {"usage", {
            {"input_tokens",  usage.value("prompt_tokens",     0)},
            {"output_tokens", usage.value("completion_tokens", 0)},
        }},
    };
}

// One SSE event frame.
std::string sse(const std::string& event, const json& data) {
    return "event: " + event + "\ndata: " + data.dump() + "\n\n";
}

// Replay a finished Anthropic `Message` as the Anthropic SSE event
// sequence (§12.7 — buffered streaming).  Functionally identical to true
// token streaming from the client's side.
std::string render_sse_stream(const json& amsg) {
    std::string out;

    json start = {
        {"type",          "message"},
        {"id",            amsg["id"]},
        {"role",          "assistant"},
        {"model",         amsg["model"]},
        {"content",       json::array()},
        {"stop_reason",   nullptr},
        {"stop_sequence", nullptr},
        {"usage", {
            {"input_tokens",  amsg["usage"]["input_tokens"]},
            {"output_tokens", 0},
        }},
    };
    out += sse("message_start",
               {{"type", "message_start"}, {"message", start}});

    int idx = 0;
    for (const auto& blk : amsg["content"]) {
        const std::string bt = blk.value("type", "text");
        if (bt == "thinking") {
            out += sse("content_block_start", {
                {"type", "content_block_start"}, {"index", idx},
                {"content_block", {{"type", "thinking"}, {"thinking", ""}}}});
            out += sse("content_block_delta", {
                {"type", "content_block_delta"}, {"index", idx},
                {"delta", {{"type", "thinking_delta"},
                           {"thinking", blk.value("thinking", "")}}}});
        } else if (bt == "tool_use") {
            out += sse("content_block_start", {
                {"type", "content_block_start"}, {"index", idx},
                {"content_block", {
                    {"type", "tool_use"},
                    {"id",   blk.value("id", "")},
                    {"name", blk.value("name", "")},
                    {"input", json::object()}}}});
            out += sse("content_block_delta", {
                {"type", "content_block_delta"}, {"index", idx},
                {"delta", {
                    {"type", "input_json_delta"},
                    {"partial_json",
                     (blk.contains("input") ? blk["input"]
                                            : json::object()).dump()}}}});
        } else {
            out += sse("content_block_start", {
                {"type", "content_block_start"}, {"index", idx},
                {"content_block", {{"type", "text"}, {"text", ""}}}});
            out += sse("content_block_delta", {
                {"type", "content_block_delta"}, {"index", idx},
                {"delta", {{"type", "text_delta"},
                           {"text", blk.value("text", "")}}}});
        }
        out += sse("content_block_stop",
                   {{"type", "content_block_stop"}, {"index", idx}});
        ++idx;
    }

    out += sse("message_delta", {
        {"type", "message_delta"},
        {"delta", {{"stop_reason",   amsg["stop_reason"]},
                   {"stop_sequence", nullptr}}},
        {"usage", {{"output_tokens", amsg["usage"]["output_tokens"]}}}});
    out += sse("message_stop", {{"type", "message_stop"}});
    return out;
}

} // namespace

// ---------------------------------------------------------------------------

AnthropicRouter::AnthropicRouter(const TrueLLMConfig& cfg,
                                 EngineInterface* engine,
                                 OpenAIRouter* openai)
    : cfg_(cfg), engine_(engine), openai_(openai)
{}

void AnthropicRouter::register_routes(httplib::Server& svr, AuthCheck auth)
{
    svr.Post("/v1/messages",
        [this, auth](const httplib::Request& req, httplib::Response& res) {
            if (!auth(req, res)) return;
            handle_messages(req, res);
        });

    svr.Post("/v1/messages/count_tokens",
        [this, auth](const httplib::Request& req, httplib::Response& res) {
            if (!auth(req, res)) return;
            handle_count_tokens(req, res);
        });
}

void AnthropicRouter::handle_messages(const httplib::Request& req,
                                      httplib::Response&      res)
{
    spdlog::debug("[Anthropic] POST /v1/messages");

    json areq;
    try { areq = json::parse(req.body); }
    catch (...) {
        res.status = 400;
        res.set_content(
            anthropic_error("invalid_request_error", "Invalid JSON body").dump(),
            "application/json");
        return;
    }

    if (!engine_ || !openai_) {
        res.status = 503;
        res.set_content(
            anthropic_error("api_error", "Server not ready").dump(),
            "application/json");
        return;
    }

    const bool want_stream = areq.value("stream", false);

    try {
        // 1. Translate Anthropic → OpenAI and run the existing pipeline.
        json obody = translate_request(areq);

        httplib::Request  oreq;
        oreq.method  = "POST";
        oreq.path    = "/v1/chat/completions";
        oreq.headers = req.headers;          // carry auth + metadata headers
        oreq.body    = obody.dump();

        httplib::Response ores;
        openai_->run_chat_completions(oreq, ores);

        // 2. Propagate an upstream error as an Anthropic error envelope.
        if (ores.status >= 400) {
            std::string msg = "upstream error";
            try {
                json e = json::parse(ores.body);
                if (e.contains("error") && e["error"].is_object())
                    msg = e["error"].value("message", msg);
                else if (e.contains("error") && e["error"].is_string())
                    msg = e["error"].get<std::string>();
            } catch (...) {}
            res.status = ores.status;
            res.set_content(anthropic_error("api_error", msg).dump(),
                            "application/json");
            return;
        }

        // 3. Translate OpenAI → Anthropic.  Echo the model name the
        //    client requested rather than the engine's internal alias.
        json oresp = json::parse(ores.body);
        json amsg  = translate_response(oresp);
        if (areq.contains("model") && areq["model"].is_string())
            amsg["model"] = areq["model"];

        // Standard Anthropic response headers.
        res.set_header("request-id", new_id("req_"));
        {
            const std::string ver = req.get_header_value("anthropic-version");
            if (!ver.empty()) res.set_header("anthropic-version", ver);
        }

        res.status = 200;
        if (want_stream) {
            // Proper chunked SSE response (Transfer-Encoding: chunked, no
            // Content-Length) — the HTTP shape an Anthropic streaming
            // client expects.  The event sequence is computed up front;
            // §12.7 documents why token-level streaming is bounded by
            // TrueLLM's parse-tool-calls-from-completed-text design.
            std::string sse_body = render_sse_stream(amsg);
            res.set_chunked_content_provider("text/event-stream",
                [sse_body = std::move(sse_body), sent = false]
                (std::size_t /*offset*/, httplib::DataSink& sink) mutable -> bool {
                    if (sent) return false;
                    sent = true;
                    sink.write(sse_body.data(), sse_body.size());
                    sink.done();
                    return true;
                });
        } else {
            res.set_content(amsg.dump(), "application/json");
        }
    } catch (const std::exception& ex) {
        spdlog::error("[Anthropic] /v1/messages — {}", ex.what());
        res.status = 500;
        res.set_content(
            anthropic_error("api_error",
                            std::string("Internal error: ") + ex.what()).dump(),
            "application/json");
    } catch (...) {
        res.status = 500;
        res.set_content(
            anthropic_error("api_error", "Internal error").dump(),
            "application/json");
    }
}

void AnthropicRouter::handle_count_tokens(const httplib::Request& req,
                                          httplib::Response&      res)
{
    spdlog::debug("[Anthropic] POST /v1/messages/count_tokens");

    json areq;
    try { areq = json::parse(req.body); }
    catch (...) {
        res.status = 400;
        res.set_content(
            anthropic_error("invalid_request_error", "Invalid JSON body").dump(),
            "application/json");
        return;
    }

    if (!engine_ || !engine_->is_model_loaded()) {
        res.status = 503;
        res.set_content(
            anthropic_error("api_error", "No model loaded").dump(),
            "application/json");
        return;
    }

    try {
        // Reuse the request translation, then format + tokenise.  This is
        // the same prompt the model would actually see, so the count is
        // accurate up to the tool-schema injection (§12.8 Phase 2).
        json obody = translate_request(areq);

        const std::string arch =
            engine_->get_model_info().architecture;
        const std::string tmpl =
            resolve_template(cfg_.model.chat_template, arch);
        const std::string prompt =
            apply_chat_template(obody.value("messages", json::array()), tmpl);

        const auto tokens = engine_->tokenize(prompt);

        res.status = 200;
        res.set_content(
            json{{"input_tokens", static_cast<int>(tokens.size())}}.dump(),
            "application/json");
    } catch (const std::exception& ex) {
        spdlog::error("[Anthropic] /v1/messages/count_tokens — {}", ex.what());
        res.status = 500;
        res.set_content(
            anthropic_error("api_error",
                            std::string("Internal error: ") + ex.what()).dump(),
            "application/json");
    }
}

} // namespace truellm
