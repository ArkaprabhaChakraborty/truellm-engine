// ---------------------------------------------------------------------------
// test_chat_format.cpp — smoke tests for the common_chat wrapper
// (src/server/chat_format.*).  Validates the std-only public contract:
// template init, prompt rendering, and reasoning/content separation.
//
// These run against the chatml fallback template (no GGUF model needed), so
// they exercise the wrapper plumbing rather than any specific model's parser.
// ---------------------------------------------------------------------------

#include "server/chat_format.h"

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>
#include <string>

using namespace truellm;

namespace {

// Build a formatter from the chatml fallback (empty template → "chatml").
ChatFormatter make_formatter(const std::string& reasoning = "auto") {
    ChatFormatter f;
    f.init(/*chat_template=*/"", /*bos=*/"<s>", /*eos=*/"</s>", reasoning);
    return f;
}

const char* kMessages =
    R"([{"role":"user","content":"Hello world"}])";

// Reasoning blocks legitimately carry surrounding whitespace (R1's "<think>\n"
// forced-open prefix leaves a leading newline); compare on trimmed text.
std::string trim(const std::string& s) {
    const auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    const auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

} // namespace

#ifdef TRUELLM_HAVE_COMMON_CHAT

TEST(ChatFormat, InitFromChatmlFallbackIsValid) {
    ChatFormatter f = make_formatter();
    EXPECT_TRUE(f.valid());
}

TEST(ChatFormat, ApplyRendersPromptContainingUserText) {
    ChatFormatter f = make_formatter();
    ASSERT_TRUE(f.valid());

    ChatParser parser;
    ChatApplied a = f.apply(kMessages, /*tools=*/"", /*tool_choice=*/"",
                            /*json_schema=*/"", /*enable_thinking=*/true, parser);
    ASSERT_TRUE(a.valid);
    EXPECT_NE(a.prompt.find("Hello world"), std::string::npos);
}

TEST(ChatFormat, FinishOnPlainTextHasNoReasoning) {
    ChatFormatter f = make_formatter();
    ASSERT_TRUE(f.valid());

    ChatParser parser;
    f.apply(kMessages, "", "", "", true, parser);

    ChatParsedMessage m = parser.finish("Just a plain answer.");
    EXPECT_EQ(m.content, "Just a plain answer.");
    EXPECT_TRUE(m.reasoning_content.empty());
    EXPECT_TRUE(m.tool_calls.empty());
}

TEST(ChatFormat, ThinkTagsSplitIntoReasoningWhenExtracted) {
    ChatFormatter f = make_formatter("auto");
    ASSERT_TRUE(f.valid());

    ChatParser parser;
    f.apply(kMessages, "", "", "", true, parser);

    ChatParsedMessage m = parser.finish("<think>step one</think>The answer is 42.");
    // When the parser extracts reasoning, the think text moves to
    // reasoning_content and the answer text is left clean.  If a given build's
    // chatml format leaves it inline, content must still hold the answer — but
    // it must never silently drop the answer text.
    EXPECT_NE(m.content.find("The answer is 42."), std::string::npos);
    if (!m.reasoning_content.empty()) {
        EXPECT_NE(m.reasoning_content.find("step one"), std::string::npos);
        EXPECT_EQ(m.content.find("<think>"), std::string::npos);
    }
}

#ifdef TRUELLM_R1_TEMPLATE_PATH
// Proves end-to-end reasoning extraction with a real reasoning template, not
// just the chatml fallback.  DeepSeek-R1 force-opens <think>, so the model
// output is "<reasoning></think><answer>".
TEST(ChatFormat, DeepSeekR1SeparatesReasoning) {
    std::ifstream in(TRUELLM_R1_TEMPLATE_PATH, std::ios::binary);
    ASSERT_TRUE(in.good()) << "template not found: " << TRUELLM_R1_TEMPLATE_PATH;
    std::stringstream ss; ss << in.rdbuf();
    const std::string tmpl = ss.str();

    ChatFormatter f;
    ASSERT_TRUE(f.init(tmpl, "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "auto"));

    ChatParser parser;
    ChatApplied a = f.apply(kMessages, "", "", "", /*enable_thinking=*/true, parser);
    ASSERT_TRUE(a.valid);

    // R1 forces <think> open, so the generated text is the reasoning, then
    // </think>, then the answer.
    ChatParsedMessage m = parser.finish("Let me work it out.</think>The answer is 42.");
    EXPECT_EQ(trim(m.reasoning_content), "Let me work it out.");
    EXPECT_EQ(trim(m.content), "The answer is 42.");
}
#endif

TEST(ChatFormat, JsonSchemaProducesGrammar) {
    const std::string schema =
        R"({"type":"object","properties":{"x":{"type":"integer"}},"required":["x"]})";
    std::string grammar = ChatFormatter::json_schema_to_grammar(schema);
    EXPECT_FALSE(grammar.empty());
    EXPECT_NE(grammar.find("root"), std::string::npos);
}

#else  // fallback build — wrapper must degrade gracefully

TEST(ChatFormat, FallbackIsInvalidButFinishReturnsRawText) {
    ChatFormatter f = make_formatter();
    EXPECT_FALSE(f.valid());

    ChatParser parser;
    ChatParsedMessage m = parser.finish("raw text");
    EXPECT_EQ(m.content, "raw text");
}

#endif // TRUELLM_HAVE_COMMON_CHAT
