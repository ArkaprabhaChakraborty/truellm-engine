#pragma once
// ---------------------------------------------------------------------------
// rate_limiter.h — Token bucket rate limiter (per-key, thread-safe)
//
// One bucket per client key (IP address or API key).
// Buckets refill at `rate_per_second` tokens/sec up to `burst_size` capacity.
// A single request consumes 1 token.  When a bucket is empty the request is
// rejected.  Buckets are created on first access and never evicted (memory is
// bounded by the number of unique clients that have ever connected).
// ---------------------------------------------------------------------------

#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>

namespace truellm::policy {

struct RateLimiterConfig {
    double rate_per_second = 20.0;  // sustained token replenishment rate
    double burst_size      = 40.0;  // bucket capacity; peak request burst
    bool   enabled         = true;  // false disables limiting (all requests pass)
};

// Returns a RateLimiterConfig appropriate for the given guardrail preset.
// preset: "strict" | "balanced" | "relaxed" | "off" | "custom"
RateLimiterConfig config_for_guardrail_preset(const std::string& preset);

class RateLimiter {
public:
    explicit RateLimiter(const RateLimiterConfig& cfg);

    // Returns true if the request from `key` is allowed (token consumed).
    // Returns false if the bucket is empty (request should be rejected 429).
    // Thread-safe.
    bool try_acquire(const std::string& key);

    // Removes the bucket for `key`.  Useful for tests or explicit client resets.
    void reset(const std::string& key);

    // Removes all buckets whose last activity is older than `max_age`.
    void evict_stale(std::chrono::seconds max_age);

    const RateLimiterConfig& config() const { return cfg_; }

private:
    struct Bucket {
        double tokens;
        std::chrono::steady_clock::time_point last_refill;
    };

    // Refills `b` based on elapsed time; returns the updated token count.
    // Must be called with mu_ held.
    double refill(Bucket& b) const;

    RateLimiterConfig cfg_;
    mutable std::mutex mu_;
    std::unordered_map<std::string, Bucket> buckets_;
};

} // namespace truellm::policy
