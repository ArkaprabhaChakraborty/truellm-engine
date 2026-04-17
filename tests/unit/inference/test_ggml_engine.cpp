#include <gtest/gtest.h>

#include "backends/cpu/ggml_engine.h"
#include <truellm/config_schema.h>
#include <truellm/types.h>

#include <cstdlib>
#include <string>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static truellm::InferenceConfig default_inf()
{
    truellm::InferenceConfig c;
    c.context_size    = 512;
    c.batch_size      = 64;
    c.ubatch_size     = 64;
    c.threads         = 2;
    c.flash_attention = false;
    return c;
}

static truellm::HardwareConfig default_hw()
{
    truellm::HardwareConfig c;
    c.offload.gpu_layers = 0;  // CPU-only for unit tests
    return c;
}

static truellm::SamplingConfig default_smp()
{
    truellm::SamplingConfig c;
    c.temperature = 0.7f;
    c.top_p       = 0.95f;
    c.top_k       = 40;
    c.seed        = 42;
    return c;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
TEST(GgmlEngineTest, NotLoadedInitially)
{
    truellm::GgmlEngine eng(default_inf(), default_hw(), default_smp());
    EXPECT_FALSE(eng.is_model_loaded());
}

TEST(GgmlEngineTest, LoadModelWithEmptyPathReturnsInvalidArgument)
{
    truellm::GgmlEngine eng(default_inf(), default_hw(), default_smp());
    auto ec = eng.load_model("");
    EXPECT_EQ(ec, truellm::ErrorCode::InvalidArgument);
    EXPECT_FALSE(eng.is_model_loaded());
}

TEST(GgmlEngineTest, LoadModelWithNonExistentPathReturnsNotFound)
{
    truellm::GgmlEngine eng(default_inf(), default_hw(), default_smp());
    auto ec = eng.load_model("/nonexistent/path/model.gguf");
    EXPECT_EQ(ec, truellm::ErrorCode::NotFound);
    EXPECT_FALSE(eng.is_model_loaded());
}

TEST(GgmlEngineTest, GetModelInfoReturnsEmptyWhenNotLoaded)
{
    truellm::GgmlEngine eng(default_inf(), default_hw(), default_smp());
    auto info = eng.get_model_info();
    EXPECT_TRUE(info.model_id.empty());
    EXPECT_EQ(info.context_length, 0);
}

TEST(GgmlEngineTest, TokenizeReturnsEmptyWhenNotLoaded)
{
    truellm::GgmlEngine eng(default_inf(), default_hw(), default_smp());
    auto toks = eng.tokenize("hello world");
    EXPECT_TRUE(toks.empty());
}

TEST(GgmlEngineTest, GenerateReturnsErrorWhenNotLoaded)
{
    truellm::GgmlEngine eng(default_inf(), default_hw(), default_smp());
    truellm::GenerateRequest req;
    req.tokens = {1, 2, 3};
    auto result = eng.generate(req);
    EXPECT_NE(result.error, truellm::ErrorCode::Ok);
}

TEST(GgmlEngineTest, GenerateReturnsInvalidArgumentForEmptyTokens)
{
    truellm::GgmlEngine eng(default_inf(), default_hw(), default_smp());
    truellm::GenerateRequest req;  // req.tokens is empty
    auto result = eng.generate(req);
    EXPECT_EQ(result.error, truellm::ErrorCode::InvalidArgument);
}

// ---------------------------------------------------------------------------
// Integration test — only runs when TRUELLM_TEST_MODEL_PATH is set
// ---------------------------------------------------------------------------
#ifdef TRUELLM_TEST_MODEL_PATH
TEST(GgmlEngineIntegrationTest, LoadRealModelAndTokenize)
{
    const char* model_path = TRUELLM_TEST_MODEL_PATH;
    ASSERT_NE(model_path, nullptr);
    ASSERT_GT(std::string(model_path).size(), 0u);

    truellm::InferenceConfig inf = default_inf();
    inf.context_size = 256;

    truellm::HardwareConfig hw = default_hw();
    const char* gpu_layers_env = std::getenv("TRUELLM_TEST_GPU_LAYERS");
    if (gpu_layers_env) hw.offload.gpu_layers = std::atoi(gpu_layers_env);

    truellm::GgmlEngine eng(inf, hw, default_smp());

    ASSERT_EQ(eng.load_model(model_path), truellm::ErrorCode::Ok);
    ASSERT_TRUE(eng.is_model_loaded());

    auto info = eng.get_model_info();
    EXPECT_GT(info.context_length, 0);
    EXPECT_GT(info.vocab_size, 0);

    auto toks = eng.tokenize("Hello, world!");
    ASSERT_GT(toks.size(), 0u);

    std::string detok = eng.detokenize(toks);
    EXPECT_NE(detok.find("Hello"), std::string::npos);

    eng.unload_model();
    EXPECT_FALSE(eng.is_model_loaded());
}
#endif
