#include <gtest/gtest.h>
#include "engine_interface.h"
#include <truellm/config_schema.h>
#include <stdexcept>

static truellm::InferenceConfig make_inf(const std::string& backend)
{
    truellm::InferenceConfig c;
    c.backend = backend;
    return c;
}

TEST(EngineFactoryTest, RejectsUnknownBackend)
{
    truellm::InferenceConfig inf = make_inf("nonexistent_backend");
    EXPECT_THROW(truellm::create_engine(inf, {}, {}), std::runtime_error);
}

TEST(EngineFactoryTest, CpuBackendReturnsValidEngine)
{
    truellm::InferenceConfig inf = make_inf("cpu");
    auto engine = truellm::create_engine(inf, {}, {});
    ASSERT_NE(engine, nullptr);
    EXPECT_FALSE(engine->is_model_loaded());
}

TEST(EngineFactoryTest, GgmlBackendReturnsValidEngine)
{
    truellm::InferenceConfig inf = make_inf("ggml");
    auto engine = truellm::create_engine(inf, {}, {});
    ASSERT_NE(engine, nullptr);
}

TEST(EngineFactoryTest, CudaBackendReturnsValidEngine)
{
    truellm::InferenceConfig inf = make_inf("cuda");
    auto engine = truellm::create_engine(inf, {}, {});
    ASSERT_NE(engine, nullptr);
}
