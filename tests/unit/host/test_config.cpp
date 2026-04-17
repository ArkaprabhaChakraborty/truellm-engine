#include <gtest/gtest.h>

#include "config/config.h"
#include "config/profile.h"
#include "config/secrets.h"

#include <spdlog/spdlog.h>
#include <cstdlib>

using namespace truellm;
using namespace truellm::config;

static TrueLLMConfig defaults() { return TrueLLMConfig{}; }

// ---------------------------------------------------------------------------
// [server] defaults
// ---------------------------------------------------------------------------
TEST(ServerConfigTest, HasSaneDefaults)
{
    TrueLLMConfig cfg = defaults();
    EXPECT_EQ(cfg.server.host, "127.0.0.1");
    EXPECT_EQ(cfg.server.port, 8080);
    EXPECT_TRUE(cfg.server.api_key.empty());
    EXPECT_EQ(cfg.server.max_concurrent, 4);
    EXPECT_FALSE(cfg.server.tls_enabled);
}

// ---------------------------------------------------------------------------
// [inference] defaults
// ---------------------------------------------------------------------------
TEST(InferenceConfigTest, HasSaneDefaults)
{
    TrueLLMConfig cfg = defaults();
    EXPECT_EQ(cfg.inference.backend,      "cpu");
    EXPECT_EQ(cfg.inference.context_size, 4096);
    EXPECT_EQ(cfg.inference.batch_size,   512);
    EXPECT_EQ(cfg.inference.threads,      0);
    EXPECT_TRUE(cfg.inference.mmap);
    EXPECT_FALSE(cfg.inference.mlock);
    EXPECT_EQ(cfg.inference.kv_cache.type_k, "f16");
    EXPECT_EQ(cfg.inference.kv_cache.type_v, "f16");
    EXPECT_EQ(cfg.inference.cuda.gpu_layers, -1);
    EXPECT_EQ(cfg.inference.cuda.main_gpu,   0);
}

// ---------------------------------------------------------------------------
// [sampling] defaults
// ---------------------------------------------------------------------------
TEST(SamplingConfigTest, HasSaneDefaults)
{
    TrueLLMConfig cfg = defaults();
    EXPECT_NEAR(cfg.sampling.temperature, 0.7f,  0.001f);
    EXPECT_NEAR(cfg.sampling.top_p,       0.95f, 0.001f);
    EXPECT_EQ(cfg.sampling.top_k,        40);
    EXPECT_EQ(cfg.sampling.max_tokens,   2048);
    EXPECT_EQ(cfg.sampling.seed,         -1);
    EXPECT_TRUE(cfg.sampling.stop.empty());
}

// ---------------------------------------------------------------------------
// [hardware] defaults
// ---------------------------------------------------------------------------
TEST(HardwareConfigTest, HasSaneDefaults)
{
    TrueLLMConfig cfg = defaults();
    EXPECT_EQ(cfg.hardware.cpu_threads,            0);
    EXPECT_TRUE(cfg.hardware.gpu.enabled);
    EXPECT_EQ(cfg.hardware.gpu.device_id,          0);
    EXPECT_EQ(cfg.hardware.gpu.backend_api,        "cuda");
    EXPECT_TRUE(cfg.hardware.offload.use_gpu_vram);
    EXPECT_TRUE(cfg.hardware.offload.use_system_ram);
    EXPECT_FALSE(cfg.hardware.offload.use_shared_memory);
    EXPECT_EQ(cfg.hardware.offload.gpu_layers,     -1);
    EXPECT_EQ(cfg.hardware.guardrails.preset,      "balanced");
    EXPECT_FALSE(cfg.hardware.guardrails.allow_overcommit);
}

// ---------------------------------------------------------------------------
// [runtime] defaults
// ---------------------------------------------------------------------------
TEST(RuntimeConfigTest, HasSaneDefaults)
{
    TrueLLMConfig cfg = defaults();
    EXPECT_EQ(cfg.runtime.engine,          "cpu-llama-cpp");
    EXPECT_EQ(cfg.runtime.engine_version,  "latest");
    EXPECT_FALSE(cfg.runtime.auto_update);
    EXPECT_EQ(cfg.runtime.update_channel,  "stable");
    EXPECT_FALSE(cfg.runtime.harmony.enabled);
}

// ---------------------------------------------------------------------------
// Validation — valid config passes
// ---------------------------------------------------------------------------
TEST(ConfigValidationTest, DefaultConfigIsValid)
{
    TrueLLMConfig cfg = defaults();
    EXPECT_TRUE(validate(cfg).empty());
}

TEST(ConfigValidationTest, RejectsInvalidBackend)
{
    TrueLLMConfig cfg = defaults();
    cfg.inference.backend = "tpu";
    EXPECT_FALSE(validate(cfg).empty());
}

TEST(ConfigValidationTest, RejectsInvalidLogLevel)
{
    TrueLLMConfig cfg = defaults();
    cfg.logging.level = "verbose";
    EXPECT_FALSE(validate(cfg).empty());
}

TEST(ConfigValidationTest, RejectsInvalidGuardrailsPreset)
{
    TrueLLMConfig cfg = defaults();
    cfg.hardware.guardrails.preset = "yolo";
    EXPECT_FALSE(validate(cfg).empty());
}

TEST(ConfigValidationTest, RejectsUnknownRuntimeEngine)
{
    TrueLLMConfig cfg = defaults();
    cfg.runtime.engine = "magic-llm";
    EXPECT_FALSE(validate(cfg).empty());
}

TEST(ConfigValidationTest, RejectsNegativeTemperature)
{
    TrueLLMConfig cfg = defaults();
    cfg.sampling.temperature = -0.1f;
    EXPECT_FALSE(validate(cfg).empty());
}

TEST(ConfigValidationTest, RejectsOutOfRangeTopP)
{
    TrueLLMConfig cfg = defaults();
    cfg.sampling.top_p = 1.5f;
    EXPECT_FALSE(validate(cfg).empty());
}

// ---------------------------------------------------------------------------
// Profiles
// ---------------------------------------------------------------------------
TEST(ProfileTest, IdentifiesKnownPresets)
{
    EXPECT_TRUE(is_known_profile("local-cpu"));
    EXPECT_TRUE(is_known_profile("workstation"));
    EXPECT_TRUE(is_known_profile("server"));
    EXPECT_FALSE(is_known_profile(""));
    EXPECT_FALSE(is_known_profile("production"));
}

TEST(ProfileTest, LocalCpuSetsCpuBackendAndConservativeLimits)
{
    TrueLLMConfig cfg = defaults();
    apply_profile(cfg, "local-cpu");
    EXPECT_EQ(cfg.inference.backend,          "cpu");
    EXPECT_EQ(cfg.server.max_concurrent,      2);
    EXPECT_EQ(cfg.runtime.engine,             "cpu-llama-cpp");
    EXPECT_EQ(cfg.hardware.guardrails.preset, "balanced");
    EXPECT_EQ(cfg.logging.format,             "text");
    EXPECT_EQ(cfg.profile,                    "local-cpu");
}

TEST(ProfileTest, WorkstationSelectsCudaAndLargerContext)
{
    TrueLLMConfig cfg = defaults();
    apply_profile(cfg, "workstation");
    EXPECT_EQ(cfg.inference.backend,           "cuda");
    EXPECT_EQ(cfg.inference.context_size,      8192);
    EXPECT_EQ(cfg.runtime.engine,              "cuda12-llama-cpp");
    EXPECT_TRUE(cfg.hardware.gpu.enabled);
    EXPECT_EQ(cfg.hardware.offload.gpu_layers, -1);
}

TEST(ProfileTest, ServerProfileUsesJsonLoggingAndStrictGuardrails)
{
    TrueLLMConfig cfg = defaults();
    apply_profile(cfg, "server");
    EXPECT_EQ(cfg.server.host,                "0.0.0.0");
    EXPECT_GE(cfg.server.max_concurrent,      8);
    EXPECT_EQ(cfg.logging.format,             "json");
    EXPECT_EQ(cfg.hardware.guardrails.preset, "strict");
    EXPECT_TRUE(cfg.inference.mlock);
}

TEST(ProfileTest, UnknownProfileNameIsIgnored)
{
    TrueLLMConfig cfg = defaults();
    apply_profile(cfg, "does-not-exist");
    EXPECT_EQ(cfg.inference.backend, "cpu");
}

// ---------------------------------------------------------------------------
// Secrets / env-var substitution
// ---------------------------------------------------------------------------
TEST(SecretsTest, PassthroughWhenNoTokensPresent)
{
    EXPECT_EQ(resolve_secrets("hello world"), "hello world");
    EXPECT_EQ(resolve_secrets(""),            "");
    EXPECT_EQ(resolve_secrets("127.0.0.1"),   "127.0.0.1");
}

TEST(SecretsTest, SubstitutesSetEnvironmentVariable)
{
#ifdef _WIN32
    _putenv_s("TRUELLM_TEST_SECRET", "my_api_key_value");
#else
    ::setenv("TRUELLM_TEST_SECRET", "my_api_key_value", 1);
#endif
    EXPECT_EQ(resolve_secrets("${ENV:TRUELLM_TEST_SECRET}"), "my_api_key_value");
}

TEST(SecretsTest, UnsetVariableResolvesToEmptyString)
{
#ifdef _WIN32
    _putenv_s("TRUELLM_UNSET_VAR_XYZ", "");
#else
    ::unsetenv("TRUELLM_UNSET_VAR_XYZ");
#endif
    EXPECT_EQ(resolve_secrets("${ENV:TRUELLM_UNSET_VAR_XYZ}"), "");
}

TEST(SecretsTest, SubstitutesMidStringToken)
{
#ifdef _WIN32
    _putenv_s("TRUELLM_HOST_VAR", "192.168.1.100");
#else
    ::setenv("TRUELLM_HOST_VAR", "192.168.1.100", 1);
#endif
    EXPECT_EQ(resolve_secrets("http://${ENV:TRUELLM_HOST_VAR}:8080"),
              "http://192.168.1.100:8080");
}

TEST(SecretsTest, MalformedTokenEmittedAsIs)
{
    const std::string bad = "${ENV:MISSING_BRACE";
    EXPECT_EQ(resolve_secrets(bad), bad);
}
