// ---------------------------------------------------------------------------
// chat_format.cpp — the single TU that bridges TrueLLM to llama.cpp's
// common_chat_* layer.  Includes the common headers (and their vendored
// nlohmann/json); exposes only std types via chat_format.h.
// ---------------------------------------------------------------------------

#include "chat_format.h"

#include <spdlog/spdlog.h>
#include <exception>

// ---------------------------------------------------------------------------
// Two builds:
//   TRUELLM_HAVE_COMMON_CHAT defined  → full common_chat_* implementation.
//   otherwise                         → no-op fallback (valid()==false) so the
//                                        routers transparently use the legacy
//                                        chat_template.h path.
// The macro is set per-source by src/server/CMakeLists.txt when the llama.cpp
// `common` target exists.
// ---------------------------------------------------------------------------
#ifdef TRUELLM_HAVE_COMMON_CHAT

// llama.cpp common library (LLAMA_BUILD_COMMON=ON).
#include "chat.h"
#include "common.h"
#include "json-schema-to-grammar.h"

namespace truellm {

namespace {
using json = nlohmann::ordered_json;

common_reasoning_format reasoning_from_string(const std::string& s)
{
    if (s.empty() || s == "none") return COMMON_REASONING_FORMAT_NONE;
    // "auto" and "deepseek" both surface reasoning as message.reasoning_content,
    // including in streaming deltas.  common_reasoning_format_from_name knows
    // the canonical names; fall back to AUTO for anything truthy.
    try {
        return common_reasoning_format_from_name(s);
    } catch (...) {
        return COMMON_REASONING_FORMAT_AUTO;
    }
}
} // namespace

// ===========================================================================
// ChatParser
// ===========================================================================
struct ChatParser::Impl {
    common_chat_parser_params params;
    common_chat_msg           prev;          // last parsed state (for diffs)
    std::vector<std::string>  tool_call_ids; // stable ids across diffs
    int                       next_id = 0;
};

ChatParser::ChatParser() : impl_(std::make_unique<Impl>()) {}
ChatParser::~ChatParser() = default;
ChatParser::ChatParser(ChatParser&&) noexcept = default;
ChatParser& ChatParser::operator=(ChatParser&&) noexcept = default;

ChatStreamDelta ChatParser::push(const std::string& accumulated_text)
{
    ChatStreamDelta out;
    if (!impl_) return out;
    try {
        common_chat_msg cur = common_chat_parse(accumulated_text,
                                                /*is_partial=*/true, impl_->params);
        if (cur.empty()) return out;

        // Assign stable ids to any newly-seen tool calls.
        cur.set_tool_call_ids(impl_->tool_call_ids,
                              [this]() { return "call_" + std::to_string(impl_->next_id++); });

        auto diffs = common_chat_msg_diff::compute_diffs(impl_->prev, cur);
        impl_->prev = cur;

        for (const auto& d : diffs) {
            out.reasoning += d.reasoning_content_delta;
            out.content   += d.content_delta;
            if (d.tool_call_index != std::string::npos) {
                ChatToolCallDelta tc;
                tc.index     = static_cast<int>(d.tool_call_index);
                tc.id        = d.tool_call_delta.id;
                tc.name      = d.tool_call_delta.name;
                tc.arguments = d.tool_call_delta.arguments;
                out.tool_calls.push_back(std::move(tc));
            }
        }
    } catch (const std::exception& ex) {
        // Partial parses routinely throw mid-token; that's expected — we just
        // emit nothing for this step and try again on the next token.
        spdlog::trace("[chat_format] partial parse pending: {}", ex.what());
    }
    return out;
}

ChatParsedMessage ChatParser::finish(const std::string& full_text) const
{
    ChatParsedMessage out;
    if (!impl_) { out.content = full_text; return out; }
    try {
        common_chat_msg msg = common_chat_parse(full_text,
                                                /*is_partial=*/false, impl_->params);
        out.content           = msg.content;
        out.reasoning_content = msg.reasoning_content;
        int idx = 0;
        for (const auto& c : msg.tool_calls) {
            ChatToolCall tc;
            tc.id        = c.id.empty() ? ("call_" + std::to_string(idx)) : c.id;
            tc.name      = c.name;
            tc.arguments = c.arguments;
            out.tool_calls.push_back(std::move(tc));
            ++idx;
        }
    } catch (const std::exception& ex) {
        spdlog::warn("[chat_format] final parse failed ({}); returning raw text", ex.what());
        out.content = full_text;
    }
    return out;
}

// ===========================================================================
// ChatFormatter
// ===========================================================================
struct ChatFormatter::Impl {
    common_chat_templates_ptr tmpls;
    common_reasoning_format   reasoning = COMMON_REASONING_FORMAT_NONE;
};

ChatFormatter::ChatFormatter() : impl_(std::make_unique<Impl>()) {}
ChatFormatter::~ChatFormatter() = default;
ChatFormatter::ChatFormatter(ChatFormatter&&) noexcept = default;
ChatFormatter& ChatFormatter::operator=(ChatFormatter&&) noexcept = default;

bool ChatFormatter::init(const std::string& chat_template,
                         const std::string& bos_token,
                         const std::string& eos_token,
                         const std::string& reasoning_format)
{
    impl_->reasoning = reasoning_from_string(reasoning_format);
    // common_chat_templates_init asserts on a null model only when the
    // override is empty; pass "chatml" so a template-less GGUF still works.
    const std::string tmpl = chat_template.empty() ? std::string("chatml") : chat_template;
    try {
        impl_->tmpls = common_chat_templates_init(/*model=*/nullptr, tmpl,
                                                  bos_token, eos_token);
    } catch (const std::exception& ex) {
        spdlog::warn("[chat_format] template init failed ({}); legacy path will be used",
                     ex.what());
        impl_->tmpls.reset();
        return false;
    }
    return impl_->tmpls != nullptr;
}

bool ChatFormatter::valid() const { return impl_ && impl_->tmpls != nullptr; }

ChatApplied ChatFormatter::apply(const std::string& messages_json,
                                 const std::string& tools_json,
                                 const std::string& tool_choice,
                                 const std::string& json_schema,
                                 bool               enable_thinking,
                                 ChatParser&        out_parser) const
{
    ChatApplied result;
    if (!valid()) return result;

    try {
        common_chat_templates_inputs inputs;
        inputs.use_jinja             = true;
        inputs.add_generation_prompt = true;
        inputs.reasoning_format      = impl_->reasoning;
        inputs.enable_thinking       = enable_thinking;

        inputs.messages = common_chat_msgs_parse_oaicompat(
            json::parse(messages_json));

        if (!tools_json.empty() && tools_json != "[]" && tools_json != "null") {
            inputs.tools = common_chat_tools_parse_oaicompat(json::parse(tools_json));
            inputs.tool_choice = tool_choice.empty()
                ? COMMON_CHAT_TOOL_CHOICE_AUTO
                : common_chat_tool_choice_parse_oaicompat(tool_choice);
        }

        if (!json_schema.empty())
            inputs.json_schema = json_schema;

        common_chat_params cp = common_chat_templates_apply(impl_->tmpls.get(), inputs);

        result.prompt           = cp.prompt;
        result.grammar          = cp.grammar;
        result.grammar_lazy     = cp.grammar_lazy;
        result.additional_stops = cp.additional_stops;
        result.valid            = true;

        // Configure the parser to match the format this prompt was built with.
        // Construct FROM cp so format + generation_prompt are carried over — the
        // latter is the forced-thinking prefix (e.g. "<think>") that the PEG
        // parser prepends to the model output to make it parseable.  Dropping it
        // makes every parse fail at pos 0 for forced-reasoning models (R1).
        auto& pi = *out_parser.impl_;
        pi.params = common_chat_parser_params(cp);
        pi.params.reasoning_format     = impl_->reasoning;
        pi.params.reasoning_in_content = false;  // keep reasoning in its own channel
        pi.params.parse_tool_calls     = !inputs.tools.empty();
        if (!cp.parser.empty())
            pi.params.parser.load(cp.parser);
        pi.prev = common_chat_msg{};
        pi.tool_call_ids.clear();
        pi.next_id = 0;
    } catch (const std::exception& ex) {
        spdlog::warn("[chat_format] apply failed ({}); legacy path will be used", ex.what());
        result.valid = false;
    }
    return result;
}

std::string ChatFormatter::json_schema_to_grammar(const std::string& schema_json)
{
    if (schema_json.empty()) return {};
    try {
        return ::json_schema_to_grammar(json::parse(schema_json), /*force_gbnf=*/true);
    } catch (const std::exception& ex) {
        spdlog::warn("[chat_format] json_schema_to_grammar failed: {}", ex.what());
        return {};
    }
}

} // namespace truellm

#else  // !TRUELLM_HAVE_COMMON_CHAT — fallback: common_chat_* unavailable

namespace truellm {

struct ChatParser::Impl {};
ChatParser::ChatParser() : impl_(nullptr) {}
ChatParser::~ChatParser() = default;
ChatParser::ChatParser(ChatParser&&) noexcept = default;
ChatParser& ChatParser::operator=(ChatParser&&) noexcept = default;
ChatStreamDelta   ChatParser::push(const std::string&) { return {}; }
ChatParsedMessage ChatParser::finish(const std::string& full_text) const {
    ChatParsedMessage out; out.content = full_text; return out;
}

struct ChatFormatter::Impl {};
ChatFormatter::ChatFormatter() : impl_(nullptr) {}
ChatFormatter::~ChatFormatter() = default;
ChatFormatter::ChatFormatter(ChatFormatter&&) noexcept = default;
ChatFormatter& ChatFormatter::operator=(ChatFormatter&&) noexcept = default;
bool ChatFormatter::init(const std::string&, const std::string&,
                         const std::string&, const std::string&) { return false; }
bool ChatFormatter::valid() const { return false; }
ChatApplied ChatFormatter::apply(const std::string&, const std::string&,
                                 const std::string&, const std::string&,
                                 bool, ChatParser&) const { return {}; }
std::string ChatFormatter::json_schema_to_grammar(const std::string&) { return {}; }

} // namespace truellm

#endif // TRUELLM_HAVE_COMMON_CHAT
