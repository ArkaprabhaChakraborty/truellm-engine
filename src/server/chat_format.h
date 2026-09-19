#pragma once
// ---------------------------------------------------------------------------
// chat_format.h — TrueLLM's wrapper around llama.cpp's common_chat_* layer.
//
// This is the ONLY place TrueLLM touches common/chat.h.  The implementation
// (chat_format.cpp) is the single translation unit that includes the common
// headers (and their vendored nlohmann/json); everything here is plain std
// types so the rest of the server never sees a common_chat_* or nlohmann type.
//
// What it gives us that the old hand-rolled chat_template.h could not:
//   * faithful prompts rendered from the model's own embedded GGUF Jinja
//     template (not one of four approximations)
//   * per-model tool-call parsing
//   * reasoning ("thinking") separation, both streaming (incremental deltas)
//     and final, via common_chat_parse + common_chat_msg_diff::compute_diffs
//   * GBNF grammars for tool calls and response_format: json_schema
//
// See analysis.md (Part 3) for the design rationale.
// ---------------------------------------------------------------------------

#include <memory>
#include <string>
#include <vector>

namespace truellm {

// ── Streaming deltas ────────────────────────────────────────────────────────
struct ChatToolCallDelta {
    int         index = -1;   // tool-call slot (stable across deltas)
    std::string id;           // set once, on the first delta for this slot
    std::string name;         // set once, on the first delta for this slot
    std::string arguments;    // incremental JSON-arguments fragment
};

struct ChatStreamDelta {
    std::string                    content;     // answer text delta
    std::string                    reasoning;   // thinking text delta
    std::vector<ChatToolCallDelta> tool_calls;  // tool-call deltas
    bool empty() const {
        return content.empty() && reasoning.empty() && tool_calls.empty();
    }
};

// ── Final parse ─────────────────────────────────────────────────────────────
struct ChatToolCall {
    std::string id;
    std::string name;
    std::string arguments;    // complete JSON-arguments string
};

struct ChatParsedMessage {
    std::string               content;
    std::string               reasoning_content;
    std::vector<ChatToolCall> tool_calls;
};

// ChatParser — incremental + final parser for a single generation.
// Obtain a configured instance from ChatFormatter::apply().  Feed it the
// running accumulated text per token (push) and/or the full text once (finish).
class ChatParser {
public:
    ChatParser();
    ~ChatParser();
    ChatParser(ChatParser&&) noexcept;
    ChatParser& operator=(ChatParser&&) noexcept;
    ChatParser(const ChatParser&)            = delete;
    ChatParser& operator=(const ChatParser&) = delete;

    // Feed the full accumulated generated text so far; returns the deltas
    // produced since the previous push().  Stateful — call once per streamed
    // token with the cumulative text.
    ChatStreamDelta push(const std::string& accumulated_text);

    // Parse the complete generated text in one shot (is_partial = false).
    // Independent of push() state.
    ChatParsedMessage finish(const std::string& full_text) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend class ChatFormatter;
};

// ChatApplied — the rendered prompt + decoding constraints for one request.
struct ChatApplied {
    std::string              prompt;            // model-ready templated prompt
    std::string              grammar;           // GBNF ("" = unconstrained)
    bool                     grammar_lazy = false;
    std::vector<std::string> additional_stops;  // template-mandated stop strings
    bool                     valid = false;     // false → fall back to legacy path
};

// ChatFormatter — owns the parsed chat template for the loaded model.
// Construct once per model load (cheap to rebuild per request too).
class ChatFormatter {
public:
    ChatFormatter();
    ~ChatFormatter();
    ChatFormatter(ChatFormatter&&) noexcept;
    ChatFormatter& operator=(ChatFormatter&&) noexcept;
    ChatFormatter(const ChatFormatter&)            = delete;
    ChatFormatter& operator=(const ChatFormatter&) = delete;

    // Build from ModelInfo's chat_template + bos/eos token strings.  When
    // chat_template is empty (GGUF carries none) it falls back to chatml.
    // reasoning_format: "none" | "auto" | "deepseek" (auto == deepseek).
    // Returns false if the template fails to parse — caller should fall back.
    bool init(const std::string& chat_template,
              const std::string& bos_token,
              const std::string& eos_token,
              const std::string& reasoning_format);

    bool valid() const;

    // Render the prompt + grammar for a request and configure out_parser for
    // the matching streaming/final parse.  All JSON args are OpenAI-shaped
    // strings:
    //   messages_json   : the "messages" array
    //   tools_json      : the "tools" array ("" or "[]" = no tools)
    //   tool_choice     : "auto" | "none" | "required" | "" (= auto)
    //   json_schema     : response_format json schema ("" = none)
    //   enable_thinking : false suppresses reasoning where the template supports it
    ChatApplied apply(const std::string& messages_json,
                      const std::string& tools_json,
                      const std::string& tool_choice,
                      const std::string& json_schema,
                      bool               enable_thinking,
                      ChatParser&        out_parser) const;

    // Standalone JSON-Schema → GBNF (response_format path that does not need a
    // template).  Returns "" on failure.
    static std::string json_schema_to_grammar(const std::string& schema_json);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace truellm
