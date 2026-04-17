#pragma once
// ---------------------------------------------------------------------------
// health_monitor.h — Periodic heartbeat monitor for sidecar processes.
//
// Sends a Heartbeat JSON message every interval_ms milliseconds.
// If no HeartbeatAck is received within ack_timeout_ms, the sidecar is
// declared unhealthy and the on_unhealthy callback fires.
// ---------------------------------------------------------------------------

#include "ipc/ipc_client.h"

#include <atomic>
#include <chrono>
#include <functional>
#include <thread>

namespace truellm {

class HealthMonitor {
public:
    using UnhealthyCallback = std::function<void()>;

    explicit HealthMonitor(IpcClient* client,
                           int interval_ms   = 5'000,
                           int ack_timeout_ms = 3'000);
    ~HealthMonitor();

    // Non-copyable.
    HealthMonitor(const HealthMonitor&)            = delete;
    HealthMonitor& operator=(const HealthMonitor&) = delete;

    void set_unhealthy_callback(UnhealthyCallback cb) { on_unhealthy_ = std::move(cb); }

    // Starts the heartbeat loop.
    void start();
    // Stops the heartbeat loop.
    void stop();

    bool is_healthy() const { return healthy_.load(); }

private:
    IpcClient*        client_;
    int               interval_ms_;
    int               ack_timeout_ms_;
    std::atomic<bool> running_  {false};
    std::atomic<bool> healthy_  {true};
    std::thread       beat_thread_;
    UnhealthyCallback on_unhealthy_;

    void beat_loop();
};

} // namespace truellm
