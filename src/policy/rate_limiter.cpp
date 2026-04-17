// ---------------------------------------------------------------------------
// rate_limiter.cpp
// ---------------------------------------------------------------------------

#include "rate_limiter.h"

#include <algorithm>

namespace truellm::policy {

// ---------------------------------------------------------------------------
// config_for_guardrail_preset
// ---------------------------------------------------------------------------
RateLimiterConfig config_for_guardrail_preset(const std::string& preset)
{
    if (preset == "strict") {
        return {/*rate=*/5.0, /*burst=*/10.0, /*enabled=*/true};
    }
    if (preset == "balanced") {
        return {/*rate=*/20.0, /*burst=*/40.0, /*enabled=*/true};
    }
    if (preset == "relaxed" || preset == "off") {
        return {/*rate=*/100.0, /*burst=*/200.0, /*enabled=*/false};
    }
    // "custom" or unknown: use balanced defaults
    return {/*rate=*/20.0, /*burst=*/40.0, /*enabled=*/true};
}

// ---------------------------------------------------------------------------
// RateLimiter
// ---------------------------------------------------------------------------
RateLimiter::RateLimiter(const RateLimiterConfig& cfg)
    : cfg_(cfg)
{}

bool RateLimiter::try_acquire(const std::string& key)
{
    if (!cfg_.enabled) return true;

    std::lock_guard<std::mutex> lk(mu_);

    auto& b = buckets_[key];
    if (b.last_refill == std::chrono::steady_clock::time_point{}) {
        // New bucket: start full
        b.tokens      = cfg_.burst_size;
        b.last_refill = std::chrono::steady_clock::now();
    } else {
        refill(b);
    }

    if (b.tokens >= 1.0) {
        b.tokens -= 1.0;
        return true;
    }
    return false;
}

void RateLimiter::reset(const std::string& key)
{
    std::lock_guard<std::mutex> lk(mu_);
    buckets_.erase(key);
}

void RateLimiter::evict_stale(std::chrono::seconds max_age)
{
    auto cutoff = std::chrono::steady_clock::now() - max_age;
    std::lock_guard<std::mutex> lk(mu_);
    for (auto it = buckets_.begin(); it != buckets_.end(); ) {
        if (it->second.last_refill < cutoff) {
            it = buckets_.erase(it);
        } else {
            ++it;
        }
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------
double RateLimiter::refill(Bucket& b) const
{
    auto now     = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration<double>(now - b.last_refill).count();
    b.tokens     = std::min(b.tokens + elapsed * cfg_.rate_per_second, cfg_.burst_size);
    b.last_refill = now;
    return b.tokens;
}

} // namespace truellm::policy
