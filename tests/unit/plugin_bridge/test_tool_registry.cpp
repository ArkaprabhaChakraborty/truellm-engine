#include <gtest/gtest.h>

#include "plugin_bridge/hooks/tool_registry.h"
#include <truellm/plugin.h>
#include <truellm/types.h>

#include <nlohmann/json.hpp>
#include <string>

using namespace truellm;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static ToolCallResult make_ok(const std::string& payload)
{
    ToolCallResult r;
    r.payload = payload;
    r.error   = TRUELLM_OK;
    return r;
}

// ---------------------------------------------------------------------------
// Empty registry
// ---------------------------------------------------------------------------
TEST(ToolRegistryTest, StartsEmpty)
{
    ToolRegistry reg;
    EXPECT_EQ(reg.tool_count(), 0u);
}

TEST(ToolRegistryTest, SchemasJsonIsEmptyArrayWhenNoTools)
{
    ToolRegistry reg;
    EXPECT_EQ(reg.get_schemas_json(), "[]");
}

// ---------------------------------------------------------------------------
// Remote tool registration
// ---------------------------------------------------------------------------
TEST(ToolRegistryTest, RegisterRemoteIncreasesToolCount)
{
    ToolRegistry reg;
    reg.register_remote("my_plugin", "get_weather",
                        R"({"name":"get_weather","description":"fetch weather"})",
                        [](const std::string&, truellm_context_t*) {
                            return make_ok(R"({"temp":22})");
                        });
    EXPECT_EQ(reg.tool_count(), 1u);
}

TEST(ToolRegistryTest, RegisterMultipleRemoteToolsDifferentPlugins)
{
    ToolRegistry reg;
    reg.register_remote("plugin_a", "tool_1", "{}", [](const std::string&, truellm_context_t*) {
        return make_ok("a1");
    });
    reg.register_remote("plugin_a", "tool_2", "{}", [](const std::string&, truellm_context_t*) {
        return make_ok("a2");
    });
    reg.register_remote("plugin_b", "tool_3", "{}", [](const std::string&, truellm_context_t*) {
        return make_ok("b3");
    });
    EXPECT_EQ(reg.tool_count(), 3u);
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------
TEST(ToolRegistryTest, DispatchUnknownToolReturnsError)
{
    ToolRegistry reg;
    auto result = reg.dispatch("nonexistent.tool", "{}", nullptr);
    EXPECT_NE(result.error, TRUELLM_OK);
    EXPECT_FALSE(result.error_msg.empty());
}

TEST(ToolRegistryTest, DispatchCallsRemoteDispatchFunction)
{
    ToolRegistry reg;
    bool called = false;
    reg.register_remote("my_plugin", "my_tool", "{}",
                        [&called](const std::string& args, truellm_context_t*) {
                            called = true;
                            EXPECT_EQ(args, R"({"q":"hello"})");
                            return make_ok("result_payload");
                        });

    auto result = reg.dispatch("my_plugin.my_tool", R"({"q":"hello"})", nullptr);
    EXPECT_TRUE(called);
    EXPECT_EQ(result.error, TRUELLM_OK);
    EXPECT_EQ(result.payload, "result_payload");
}

TEST(ToolRegistryTest, DispatchReturnsErrorFromRemoteFunctionFailure)
{
    ToolRegistry reg;
    reg.register_remote("my_plugin", "fail_tool", "{}",
                        [](const std::string&, truellm_context_t*) {
                            ToolCallResult r;
                            r.error     = TRUELLM_ERR_INTERNAL;
                            r.error_msg = "backend failure";
                            return r;
                        });

    auto result = reg.dispatch("my_plugin.fail_tool", "{}", nullptr);
    EXPECT_NE(result.error, TRUELLM_OK);
    EXPECT_EQ(result.error_msg, "backend failure");
}

// ---------------------------------------------------------------------------
// Unregister
// ---------------------------------------------------------------------------
TEST(ToolRegistryTest, UnregisterRemotePluginRemovesAllItsTools)
{
    ToolRegistry reg;
    reg.register_remote("plugin_a", "tool_1", "{}", [](const std::string&, truellm_context_t*) {
        return make_ok("");
    });
    reg.register_remote("plugin_a", "tool_2", "{}", [](const std::string&, truellm_context_t*) {
        return make_ok("");
    });
    reg.register_remote("plugin_b", "tool_3", "{}", [](const std::string&, truellm_context_t*) {
        return make_ok("");
    });
    ASSERT_EQ(reg.tool_count(), 3u);

    reg.unregister_remote_plugin("plugin_a");
    EXPECT_EQ(reg.tool_count(), 1u);

    // plugin_b tool should still work
    auto result = reg.dispatch("plugin_b.tool_3", "{}", nullptr);
    EXPECT_EQ(result.error, TRUELLM_OK);
}

TEST(ToolRegistryTest, UnregisterNonExistentPluginIsNoop)
{
    ToolRegistry reg;
    reg.register_remote("plugin_a", "tool_1", "{}", [](const std::string&, truellm_context_t*) {
        return make_ok("");
    });
    EXPECT_NO_THROW(reg.unregister_remote_plugin("does_not_exist"));
    EXPECT_EQ(reg.tool_count(), 1u);
}

// ---------------------------------------------------------------------------
// Schema JSON
// ---------------------------------------------------------------------------
TEST(ToolRegistryTest, SchemasJsonIncludesRegisteredTools)
{
    ToolRegistry reg;
    const std::string schema = R"({"type":"function","function":{"name":"get_weather"}})";
    reg.register_remote("w_plugin", "get_weather", schema,
                        [](const std::string&, truellm_context_t*) {
                            return make_ok("");
                        });

    const std::string json_str = reg.get_schemas_json();
    EXPECT_NE(json_str, "[]");

    // Should parse as a valid JSON array
    auto arr = nlohmann::json::parse(json_str);
    EXPECT_TRUE(arr.is_array());
    EXPECT_EQ(arr.size(), 1u);
}
