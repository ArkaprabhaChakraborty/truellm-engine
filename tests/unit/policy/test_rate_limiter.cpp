#include <gtest/gtest.h>

#include "policy/rate_limiter.h"
#include "policy/guardrail_policy.h"

#include <chrono>
#include <thread>

using namespace truellm::policy;

// ---------------------------------------------------------------------------
// RateLimiter — basic behaviour
// ---------------------------------------------------------------------------
TEST(RateLimiterTest, FirstRequestIsAlwaysAllowed)
{
    RateLimiterConfig cfg{/*rate=*/10.0, /*burst=*/10.0, /*enabled=*/true};
    RateLimiter rl(cfg);
    EXPECT_TRUE(rl.try_acquire("client1"));
}

TEST(RateLimiterTest, DisabledLimiterAlwaysPasses)
{
    RateLimiterConfig cfg{/*rate=*/0.001, /*burst=*/1.0, /*enabled=*/false};
    RateLimiter rl(cfg);
    for (int i = 0; i < 100; ++i) {
        EXPECT_TRUE(rl.try_acquire("client1"));
    }
}

TEST(RateLimiterTest, BurstSizeIsRespected)
{
    // burst=3: first 3 requests pass, 4th should fail
    RateLimiterConfig cfg{/*rate=*/0.0, /*burst=*/3.0, /*enabled=*/true};
    RateLimiter rl(cfg);

    EXPECT_TRUE(rl.try_acquire("ip1"));   // token 3 -> 2
    EXPECT_TRUE(rl.try_acquire("ip1"));   // token 2 -> 1
    EXPECT_TRUE(rl.try_acquire("ip1"));   // token 1 -> 0
    EXPECT_FALSE(rl.try_acquire("ip1"));  // empty bucket
}

TEST(RateLimiterTest, DifferentClientsHaveIndependentBuckets)
{
    // burst=1 so that only one request per client is allowed without refill
    RateLimiterConfig cfg{/*rate=*/0.0, /*burst=*/1.0, /*enabled=*/true};
    RateLimiter rl(cfg);

    EXPECT_TRUE(rl.try_acquire("clientA"));
    EXPECT_FALSE(rl.try_acquire("clientA"));  // exhausted

    EXPECT_TRUE(rl.try_acquire("clientB"));   // independent bucket, still full
    EXPECT_FALSE(rl.try_acquire("clientB"));
}

TEST(RateLimiterTest, TokensRefillOverTime)
{
    // rate=1000/s so that 1ms of sleep is enough to refill 1 token
    RateLimiterConfig cfg{/*rate=*/1000.0, /*burst=*/2.0, /*enabled=*/true};
    RateLimiter rl(cfg);

    EXPECT_TRUE(rl.try_acquire("c"));
    EXPECT_TRUE(rl.try_acquire("c"));
    EXPECT_FALSE(rl.try_acquire("c"));  // empty

    std::this_thread::sleep_for(std::chrono::milliseconds(2));

    EXPECT_TRUE(rl.try_acquire("c"));  // refilled
}

TEST(RateLimiterTest, ResetClearsBucket)
{
    RateLimiterConfig cfg{/*rate=*/0.0, /*burst=*/1.0, /*enabled=*/true};
    RateLimiter rl(cfg);

    EXPECT_TRUE(rl.try_acquire("c"));
    EXPECT_FALSE(rl.try_acquire("c"));

    rl.reset("c");
    EXPECT_TRUE(rl.try_acquire("c"));  // new bucket, starts full
}

// ---------------------------------------------------------------------------
// config_for_guardrail_preset
// ---------------------------------------------------------------------------
TEST(RateLimiterTest, StrictPresetHasLowRate)
{
    auto cfg = config_for_guardrail_preset("strict");
    EXPECT_TRUE(cfg.enabled);
    EXPECT_LT(cfg.rate_per_second, 10.0);
    EXPECT_LT(cfg.burst_size,      20.0);
}

TEST(RateLimiterTest, BalancedPresetIsEnabled)
{
    auto cfg = config_for_guardrail_preset("balanced");
    EXPECT_TRUE(cfg.enabled);
}

TEST(RateLimiterTest, RelaxedPresetIsDisabled)
{
    auto cfg = config_for_guardrail_preset("relaxed");
    EXPECT_FALSE(cfg.enabled);
}

TEST(RateLimiterTest, OffPresetIsDisabled)
{
    auto cfg = config_for_guardrail_preset("off");
    EXPECT_FALSE(cfg.enabled);
}

// ---------------------------------------------------------------------------
// GuardrailPolicy — constraints
// ---------------------------------------------------------------------------
TEST(GuardrailPolicyTest, StrictPresetEnforcesLowLimits)
{
    auto c = constraints_for_preset("strict");
    EXPECT_GT(c.max_tokens,     0);
    EXPECT_GT(c.max_ctx_tokens, 0);
    EXPECT_LE(c.max_tokens,     2048);
    EXPECT_FALSE(c.allow_overcommit);
}

TEST(GuardrailPolicyTest, BalancedPresetHasModerateTokenLimit)
{
    auto c = constraints_for_preset("balanced");
    EXPECT_GT(c.max_tokens, 2048);
    EXPECT_LE(c.max_tokens, 8192);
}

TEST(GuardrailPolicyTest, OffPresetHasNoLimits)
{
    auto c = constraints_for_preset("off");
    EXPECT_EQ(c.max_tokens,     0);
    EXPECT_EQ(c.max_ctx_tokens, 0);
}

TEST(GuardrailPolicyTest, CheckRequestPassesWithinLimits)
{
    auto c = constraints_for_preset("balanced");
    std::string violation;
    EXPECT_TRUE(check_request(c, 1024, 2048, violation));
    EXPECT_TRUE(violation.empty());
}

TEST(GuardrailPolicyTest, CheckRequestFailsWhenMaxTokensExceeded)
{
    auto c = constraints_for_preset("strict");
    std::string violation;
    EXPECT_FALSE(check_request(c, 99999, 100, violation));
    EXPECT_FALSE(violation.empty());
}

TEST(GuardrailPolicyTest, CheckRequestFailsWhenContextExceeded)
{
    auto c = constraints_for_preset("strict");
    std::string violation;
    EXPECT_FALSE(check_request(c, 512, 999999, violation));
    EXPECT_FALSE(violation.empty());
}

TEST(GuardrailPolicyTest, OffPresetAlwaysPasses)
{
    auto c = constraints_for_preset("off");
    std::string violation;
    EXPECT_TRUE(check_request(c, 999999, 999999, violation));
}

TEST(GuardrailPolicyTest, ClampMaxTokensRespectsBound)
{
    auto c = constraints_for_preset("strict");
    EXPECT_LE(clamp_max_tokens(c, 99999), c.max_tokens);
    EXPECT_EQ(clamp_max_tokens(c, 100),   100);
}

TEST(GuardrailPolicyTest, ClampMaxTokensNoopWhenNoLimit)
{
    auto c = constraints_for_preset("off");  // max_tokens == 0 means no limit
    EXPECT_EQ(clamp_max_tokens(c, 99999), 99999);
}
