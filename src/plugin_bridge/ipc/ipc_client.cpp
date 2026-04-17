// ---------------------------------------------------------------------------
// ipc_client.cpp
// ---------------------------------------------------------------------------

#include "ipc_client.h"

#include <spdlog/spdlog.h>

namespace truellm {

using json = nlohmann::json;

// ---------------------------------------------------------------------------
IpcClient::IpcClient(IpcTransport* transport)
    : transport_(transport)
{}

// ---------------------------------------------------------------------------
void IpcClient::start()
{
    transport_->start_recv_loop([this](std::vector<uint8_t> raw) {
        on_message(std::move(raw));
    });
}

// ---------------------------------------------------------------------------
// on_message — decode JSON envelope and resolve pending futures.
// ---------------------------------------------------------------------------
void IpcClient::on_message(std::vector<uint8_t> raw)
{
    json msg;
    try {
        msg = json::parse(raw.begin(), raw.end());
    } catch (const json::exception& e) {
        spdlog::warn("[IpcClient] JSON parse error: {}", e.what());
        return;
    }

    const std::string type = msg.value("type", "");
    // Sidecars use "cid" as the correlation key (matches Python/JS protocol).
    uint64_t cid = msg.contains("cid")
        ? msg.value("cid", uint64_t{0})
        : msg.value("correlation_id", uint64_t{0}); // legacy fallback

    if (type == "status_report" && status_cb_) {
        status_cb_(cid, msg.value("message", ""));
        return;
    }

    if (type == "plugin_announce") {
        if (announce_cb_) announce_cb_(std::move(msg));
        // Not a correlated reply — do not route to pending_ map.
        return;
    }

    // heartbeat_ack, tool_response, preprocess_response all resolve via cid.
    resolve(cid, msg);
}

void IpcClient::resolve(uint64_t cid, json body)
{
    std::lock_guard<std::mutex> lk(pending_mu_);
    auto it = pending_.find(cid);
    if (it == pending_.end()) {
        spdlog::debug("[IpcClient] unexpected reply for cid={}", cid);
        return;
    }
    it->second.set_value(std::move(body));
    pending_.erase(it);
}

// ---------------------------------------------------------------------------
// wait_for
// ---------------------------------------------------------------------------
std::optional<json> IpcClient::wait_for(std::future<json>& fut, int timeout_ms)
{
    auto status = fut.wait_for(std::chrono::milliseconds(timeout_ms));
    if (status != std::future_status::ready) {
        return std::nullopt;
    }
    return fut.get();
}

// ---------------------------------------------------------------------------
// call_tool
// ---------------------------------------------------------------------------
IpcClient::ToolCallResponse IpcClient::call_tool(
    const ToolCallRequest& req, int timeout_ms)
{
    uint64_t cid = next_id();

    // Register promise before sending to avoid a race.
    std::future<json> fut;
    {
        std::lock_guard<std::mutex> lk(pending_mu_);
        auto [it, ok] = pending_.emplace(cid, std::promise<json>{});
        fut = it->second.get_future();
    }

    // Build and send the request.
    const char* msg_type = req.is_heartbeat ? "heartbeat" : "tool_call";
    json msg = {
        {"type",           msg_type},
        {"cid",            cid},
        {"qualified_name", req.qualified_name},
        {"args_json",      req.args_json},
        {"request_id",     req.request_id},
        {"token_budget",   req.token_budget},
        {"max_tokens",     req.max_tokens},
        {"temperature",    req.temperature},
    };

    std::string serialised = msg.dump();
    if (!transport_->send(
            reinterpret_cast<const uint8_t*>(serialised.data()),
            static_cast<uint32_t>(serialised.size()))) {
        std::lock_guard<std::mutex> lk(pending_mu_);
        pending_.erase(cid);
        return {false, "", "send failed"};
    }

    auto reply = wait_for(fut, timeout_ms);
    if (!reply) {
        {
            std::lock_guard<std::mutex> lk(pending_mu_);
            pending_.erase(cid);
        }
        spdlog::warn("[IpcClient] tool_call timeout cid={}", cid);
        return {false, "", "timeout"};
    }

    bool ok = ((*reply).value("error", 0) == 0);
    return {
        ok,
        (*reply).value("payload",   ""),
        (*reply).value("error_msg", ""),
    };
}

// ---------------------------------------------------------------------------
// run_preprocessor
// ---------------------------------------------------------------------------
IpcClient::PreprocessResponse IpcClient::run_preprocessor(
    const PreprocessRequest& req, int timeout_ms)
{
    uint64_t cid = next_id();

    std::future<json> fut;
    {
        std::lock_guard<std::mutex> lk(pending_mu_);
        auto [it, ok] = pending_.emplace(cid, std::promise<json>{});
        fut = it->second.get_future();
    }

    json msg = {
        {"type",          "preprocess_call"},
        {"cid",           cid},
        {"plugin_id",     req.plugin_id},
        {"messages_json", req.messages_json},
        {"request_id",    req.request_id},
        {"token_budget",  req.token_budget},
        {"max_tokens",    req.max_tokens},
        {"temperature",   req.temperature},
    };

    std::string serialised = msg.dump();
    if (!transport_->send(
            reinterpret_cast<const uint8_t*>(serialised.data()),
            static_cast<uint32_t>(serialised.size()))) {
        std::lock_guard<std::mutex> lk(pending_mu_);
        pending_.erase(cid);
        return {false};
    }

    auto reply = wait_for(fut, timeout_ms);
    if (!reply) {
        {
            std::lock_guard<std::mutex> lk(pending_mu_);
            pending_.erase(cid);
        }
        spdlog::warn("[IpcClient] preprocess timeout cid={}", cid);
        return {false};
    }

    return {
        true,
        (*reply).value("decision", 0),
        (*reply).value("modified_messages_json", ""),
        (*reply).value("response_text", ""),
    };
}

// ---------------------------------------------------------------------------
// abort_call
// ---------------------------------------------------------------------------
void IpcClient::abort_call(uint64_t correlation_id)
{
    json msg = {
        {"type", "abort_call"},
        {"cid",  correlation_id},
    };
    std::string s = msg.dump();
    transport_->send(reinterpret_cast<const uint8_t*>(s.data()),
                     static_cast<uint32_t>(s.size()));
}

} // namespace truellm
