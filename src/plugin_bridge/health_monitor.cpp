// ---------------------------------------------------------------------------
// health_monitor.cpp
// ---------------------------------------------------------------------------

#include "health_monitor.h"

#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

namespace truellm {

using json = nlohmann::json;

HealthMonitor::HealthMonitor(IpcClient* client, int interval_ms, int ack_timeout_ms)
    : client_(client)
    , interval_ms_(interval_ms)
    , ack_timeout_ms_(ack_timeout_ms)
{}

HealthMonitor::~HealthMonitor()
{
    stop();
}

void HealthMonitor::start()
{
    running_ = true;
    beat_thread_ = std::thread([this]() { beat_loop(); });
}

void HealthMonitor::stop()
{
    running_ = false;
    if (beat_thread_.joinable())
        beat_thread_.join();
}

void HealthMonitor::beat_loop()
{
    uint64_t seq = 0;

    while (running_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms_));
        if (!running_.load()) break;

        IpcClient::ToolCallRequest hb_req;
        hb_req.qualified_name = "__heartbeat__";   // sentinel handled by IpcClient
        hb_req.args_json      = "{\"seq\":" + std::to_string(seq++) + "}";
        hb_req.is_heartbeat   = true;

        auto resp = client_->call_tool(hb_req, ack_timeout_ms_);
        if (!resp.ok) {
            spdlog::warn("[HealthMonitor] heartbeat missed (seq={})", seq - 1);
            healthy_ = false;
            if (on_unhealthy_) on_unhealthy_();
        } else {
            healthy_ = true;
            spdlog::debug("[HealthMonitor] heartbeat ack seq={}", seq - 1);
        }
    }
}

} // namespace truellm
