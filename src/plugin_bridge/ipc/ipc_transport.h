#pragma once
// ---------------------------------------------------------------------------
// ipc_transport.h — Platform-portable byte-stream transport.
//
// Windows: Named Pipe  (\\.\pipe\truellm-plugins-<runtime>)
// POSIX:   Unix Domain Socket (/tmp/truellm-plugins-<runtime>.sock)
//
// Wire format: [uint32_t little-endian length] [<length> bytes payload]
// The payload is a FlatBuffers Envelope.
// ---------------------------------------------------------------------------

#include <cstdint>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <sys/types.h>
#endif

namespace truellm {

// ---------------------------------------------------------------------------
// IpcTransport
// ---------------------------------------------------------------------------
class IpcTransport {
public:
    // Callback invoked on the receive thread for every complete message.
    using RecvCallback = std::function<void(std::vector<uint8_t> payload)>;

    explicit IpcTransport(std::string runtime_tag); // "py" | "js"
    ~IpcTransport();

    // Non-copyable.
    IpcTransport(const IpcTransport&)            = delete;
    IpcTransport& operator=(const IpcTransport&) = delete;

    // Returns the socket/pipe path the sidecar must connect to.
    std::string socket_path() const { return socket_path_; }

    // Starts listening for a single sidecar connection.
    // Blocks until the sidecar connects, then returns.
    // Returns false if the listen fails.
    bool listen_and_accept();

    // Sends a framed message to the connected sidecar.
    // Thread-safe.  Returns false on write error.
    bool send(const uint8_t* data, uint32_t len);
    bool send(const std::vector<uint8_t>& buf) { return send(buf.data(), static_cast<uint32_t>(buf.size())); }

    // Starts a background receive thread that calls cb for each inbound message.
    // Must be called after listen_and_accept().
    void start_recv_loop(RecvCallback cb);

    // Closes the connection and stops the receive thread.
    void close();

    bool is_connected() const { return connected_; }

private:
    std::string  socket_path_;
    bool         connected_ = false;
    std::thread  recv_thread_;

#ifdef _WIN32
    HANDLE listen_pipe_ = INVALID_HANDLE_VALUE;
    HANDLE conn_pipe_   = INVALID_HANDLE_VALUE;
    bool   write_to_pipe(const uint8_t* data, uint32_t len);
    bool   read_exact(HANDLE h, uint8_t* buf, DWORD n);
#else
    int server_fd_ = -1;
    int conn_fd_   = -1;
    bool write_to_fd(const int fd, const uint8_t* data, uint32_t len);
    bool read_exact(int fd, uint8_t* buf, size_t n);
#endif

    void recv_loop(RecvCallback cb);
};

} // namespace truellm
