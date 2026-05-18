#pragma once
// ---------------------------------------------------------------------------
// chat_template.h — header-only helpers for building a model-formatted
//                   prompt string out of a messages JSON array.
//
// Why header-only
//   These helpers are needed by both the OpenAI-compatible router AND the
//   plugin bridge's call_engine trampoline (sub-calls coming back from
//   plugins must also be formatted before tokenisation, otherwise the
//   model sees raw `{"role":"...", ...}` JSON and produces garbage).
//   Putting them in a header avoids a third translation unit and the
//   accompanying CMake plumbing.
//
// Templates supported
//   llama3 — LLaMA 3 / 3.1 / Nemotron families.  Native system/user/
//            assistant/tool roles.
//   gemma  — Gemma 2 / 3.  Only user/model; assistant→model; tool→user
//            with "[Tool result]" prefix.
//   phi3   — Phi-3 / 3.5.  <|role|>content<|end|>.
//   chatml — Qwen 2+, Mistral instruct v0.3+, and the catch-all default.
// ---------------------------------------------------------------------------

#include <nlohmann/json.hpp>

#include <cctype>
#include <string>

namespace truellm {

// resolve_template — pick the right template name for a model when the user
// has not explicitly configured one ("auto").  Falls through to "chatml"
// for unknown architectures.
inline std::string resolve_template(const std::string& cfg_hint,
                                    const std::string& arch)
{
    if (cfg_hint != "auto") return cfg_hint;

    std::string a = arch;
    for (char& c : a) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    if (a.rfind("llama", 0) == 0 || a == "mistral")     return "llama3";
    if (a.rfind("gemma", 0) == 0)                        return "gemma";
    if (a.rfind("phi3",  0) == 0 || a == "phi-3")        return "phi3";
    return "chatml";
}

// resolve_message_content — content text to embed in the prompt for a
// single message object.  Assistant messages with no content but a
// non-empty tool_calls array are serialised back to JSON so the rebuilt
// multi-turn prompt reflects what the model said.
inline std::string resolve_message_content(const nlohmann::json& msg)
{
    std::string content = msg.value("content", "");
    if (content.empty()
        && msg.value("role", "") == "assistant"
        && msg.contains("tool_calls")
        && msg["tool_calls"].is_array()
        && !msg["tool_calls"].empty())
    {
        content = nlohmann::json{{"tool_calls", msg["tool_calls"]}}.dump();
    }
    return content;
}

// apply_chat_template — concatenate a messages JSON array into the
// model-specific prompt string.  Always opens an assistant turn at the
// end so the model fills in from there.  No BOS embedding (the tokenizer
// adds the model's BOS via add_special=true).
inline std::string apply_chat_template(const nlohmann::json& messages,
                                       const std::string& tmpl_hint)
{
    std::string prompt;
    for (const auto& msg : messages) {
        const std::string role    = msg.value("role", "user");
        const std::string content = resolve_message_content(msg);

        if (tmpl_hint == "llama3") {
            prompt += "<|start_header_id|>" + role + "<|end_header_id|>\n\n";
            prompt += content + "<|eot_id|>";
        } else if (tmpl_hint == "gemma") {
            std::string gemma_role;
            std::string gemma_content = content;
            if (role == "assistant") {
                gemma_role = "model";
            } else if (role == "tool") {
                gemma_role    = "user";
                gemma_content = "[Tool result]\n" + content;
            } else {
                gemma_role = role;
            }
            prompt += "<start_of_turn>" + gemma_role + "\n"
                    + gemma_content + "<end_of_turn>\n";
        } else if (tmpl_hint == "phi3") {
            prompt += "<|" + role + "|>\n" + content + "<|end|>\n";
        } else {
            // ChatML — also the catch-all default.
            prompt += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
        }
    }
    if      (tmpl_hint == "llama3") prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n";
    else if (tmpl_hint == "gemma")  prompt += "<start_of_turn>model\n";
    else if (tmpl_hint == "phi3")   prompt += "<|assistant|>\n";
    else                            prompt += "<|im_start|>assistant\n";
    return prompt;
}

// apply_chat_template_str — convenience wrapper that parses a JSON string
// before delegating to apply_chat_template.  Returns an empty string on
// parse failure so callers can fall back to passing the raw text through.
inline std::string apply_chat_template_str(const std::string& messages_json,
                                           const std::string& tmpl_hint)
{
    auto j = nlohmann::json::parse(messages_json, nullptr, false);
    if (j.is_discarded()) return {};
    if (!j.is_array())    return {};
    return apply_chat_template(j, tmpl_hint);
}

} // namespace truellm
