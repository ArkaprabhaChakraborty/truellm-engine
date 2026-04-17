#pragma once
// ---------------------------------------------------------------------------
// ipc_client.h — High-level IPC client used by PluginBridge.
//
// Wraps IpcTransport with:
//   - A correlation-id → std::promise<T> map for call/response pairing.
//   - Typed send/receive helpers for each message type.
//   - A background recv thread that resolves pending promises.
//
// The IPC message format is length-prefixed JSON (same framing as
// IpcTransport).  FlatBuffers codegen is wired in a subsequent step;
// for Phase D the messages use nlohmann::json serialised to UTF-8.
// ---------------------------------------------------------------------------

#include "ipc_transport.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <atomic>
#include <chrono>
#include <future>
#include <mutex>
#include <string>
#include <unordered_map>

namespace truellm {

// ---------------------------------------------------------------------------
// IpcClient
// ---------------------------------------------------------------------------
class IpcClient {
public:
    // Milliseconds to wait for a tool/preprocess/generate reply.
    static constexpr int kDefaultTimeoutMs = 10'000;

    explicit IpcClient(IpcTransport* transport);
    ~IpcClient() = default;

    // Non-copyable.
    IpcClient(const IpcClient&)            = delete;
    IpcClient& operator=(const IpcClient&) = delete;

    // Starts the receive dispatch loop.  Must be called once after the transport
    // has accepted a connection.
    void start();

    // ── Tool call ─────────────────────────────────────────────────────────────
    struct ToolCallRequest {
        std::string  qualified_name;
        std::string  args_json;
        std::string  request_id;
        int64_t      token_budget   = 0;
        int32_t      max_tokens     = 512;
        float        temperature    = 0.7f;
        bool         is_heartbeat   = false; // if true, sends type="heartbeat" not "tool_call"
    };

    struct ToolCallResponse {
        bool        ok = false;
        std::string payload;
        std::string error_msg;
    };

    ToolCallResponse call_tool(const ToolCallRequest& req,
                               int timeout_ms = kDefaultTimeoutMs);

    // ── Preprocessor ─────────────────────────────────────────────────────────
    struct PreprocessRequest {
        std::string plugin_id;
        std::string messages_json;
        std::string request_id;
        int64_t     token_budget = 0;
        int32_t     max_tokens   = 512;
        float       temperature  = 0.7f;
    };

    struct PreprocessResponse {
        bool        ok = false;
        int         decision = 0;  // 0=pass, 1=respond, 2=abort
        std::string modified_messages_json;
        std::string response_text;
    };

    PreprocessResponse run_preprocessor(const PreprocessRequest& req,
                                         int timeout_ms = kDefaultTimeoutMs);

    // ── Status report ─────────────────────────────────────────────────────────
    // Non-blocking; fire-and-forget status notification from plugin to engine.
    // Engine calls this internally to relay report_status() from sidecar plugins.
    using StatusCallback = std::function<void(uint64_t correlation_id, std::string message)>;
    void set_status_callback(StatusCallback cb) { status_cb_ = std::move(cb); }

    // ── Plugin announce ───────────────────────────────────────────────────────
    // Called when the sidecar sends a "plugin_announce" message (unsolicited,
    // cid == 0).  Must be set before start() is called.
    using AnnounceCallback = std::function<void(nlohmann::json announce)>;
    void set_announce_callback(AnnounceCallback cb) { announce_cb_ = std::move(cb); }

    // Sends an Abort message for an in-flight correlation_id.
    void abort_call(uint64_t correlation_id);

    bool is_connected() const { return transport_->is_connected(); }

private:
    IpcTransport* transport_;

    std::atomic<uint64_t> next_correlation_id_{1};

    // Pending call map: correlation_id → promise<nlohmann::json>
    std::mutex                                         pending_mu_;
    std::unordered_map<uint64_t,
        std::promise<nlohmann::json>>                  pending_;

    StatusCallback   status_cb_;
    AnnounceCallback announce_cb_;

    void on_message(std::vector<uint8_t> raw);
    void resolve(uint64_t cid, nlohmann::json body);

    uint64_t next_id() { return next_correlation_id_.fetch_add(1, std::memory_order_relaxed); }

    // Waits for a future with a timeout.  Returns nullopt on timeout.
    static std::optional<nlohmann::json> wait_for(
        std::future<nlohmann::json>& fut,
        int timeout_ms);
};

} // namespace truellm
