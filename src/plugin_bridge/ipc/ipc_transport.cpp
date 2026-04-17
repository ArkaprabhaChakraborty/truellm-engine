// ---------------------------------------------------------------------------
// ipc_transport.cpp — Platform-portable IPC byte-stream transport.
// ---------------------------------------------------------------------------

#include "ipc_transport.h"

#include <spdlog/spdlog.h>

#include <array>
#include <cstring>

#ifdef _WIN32
// Windows: Named Pipes
#  include <sstream>
#  include <process.h>
#else
// POSIX: Unix Domain Sockets
#  include <sys/socket.h>
#  include <sys/un.h>
#  include <unistd.h>
#  include <fcntl.h>
#endif

namespace truellm {

// ---------------------------------------------------------------------------
// Constructor / destructor
// ---------------------------------------------------------------------------
IpcTransport::IpcTransport(std::string runtime_tag)
{
#ifdef _WIN32
    std::ostringstream ss;
    ss << R"(\\.\pipe\truellm-plugins-)" << runtime_tag << "-" << GetCurrentProcessId();
    socket_path_ = ss.str();
#else
    socket_path_ = "/tmp/truellm-plugins-" + runtime_tag + ".sock";
    // Remove stale socket from a previous run.
    ::unlink(socket_path_.c_str());
#endif
}

IpcTransport::~IpcTransport()
{
    close();
}

// ---------------------------------------------------------------------------
// listen_and_accept
// ---------------------------------------------------------------------------
bool IpcTransport::listen_and_accept()
{
#ifdef _WIN32
    listen_pipe_ = CreateNamedPipeA(
        socket_path_.c_str(),
        PIPE_ACCESS_DUPLEX | FILE_FLAG_FIRST_PIPE_INSTANCE,
        PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
        1,          // max instances
        65536,      // out buffer
        65536,      // in buffer
        0,          // default timeout
        nullptr     // security attributes
    );

    if (listen_pipe_ == INVALID_HANDLE_VALUE) {
        spdlog::error("[IpcTransport] CreateNamedPipeA failed: {}", GetLastError());
        return false;
    }

    spdlog::debug("[IpcTransport] waiting for sidecar on {}", socket_path_);

    if (!ConnectNamedPipe(listen_pipe_, nullptr) &&
        GetLastError() != ERROR_PIPE_CONNECTED) {
        spdlog::error("[IpcTransport] ConnectNamedPipe failed: {}", GetLastError());
        CloseHandle(listen_pipe_);
        listen_pipe_ = INVALID_HANDLE_VALUE;
        return false;
    }

    conn_pipe_  = listen_pipe_;
    listen_pipe_ = INVALID_HANDLE_VALUE;
    connected_  = true;
    spdlog::info("[IpcTransport] sidecar connected on {}", socket_path_);
    return true;

#else
    server_fd_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        spdlog::error("[IpcTransport] socket() failed: {}", strerror(errno));
        return false;
    }

    struct sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);
    addr.sun_path[sizeof(addr.sun_path) - 1] = '\0'; // explicit guard: strncpy does not null-terminate on truncation

    if (::bind(server_fd_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        spdlog::error("[IpcTransport] bind() failed: {}", strerror(errno));
        ::close(server_fd_);
        server_fd_ = -1;
        return false;
    }

    if (::listen(server_fd_, 1) < 0) {
        spdlog::error("[IpcTransport] listen() failed: {}", strerror(errno));
        ::close(server_fd_);
        server_fd_ = -1;
        return false;
    }

    spdlog::debug("[IpcTransport] waiting for sidecar on {}", socket_path_);

    conn_fd_ = ::accept(server_fd_, nullptr, nullptr);
    if (conn_fd_ < 0) {
        spdlog::error("[IpcTransport] accept() failed: {}", strerror(errno));
        return false;
    }

    connected_ = true;
    spdlog::info("[IpcTransport] sidecar connected on {}", socket_path_);
    return true;
#endif
}

// ---------------------------------------------------------------------------
// send — length-prefix framing
// ---------------------------------------------------------------------------
bool IpcTransport::send(const uint8_t* data, uint32_t len)
{
    if (!connected_) return false;

    // Little-endian 4-byte length prefix.
    uint8_t hdr[4];
    hdr[0] = static_cast<uint8_t>(len & 0xFF);
    hdr[1] = static_cast<uint8_t>((len >> 8)  & 0xFF);
    hdr[2] = static_cast<uint8_t>((len >> 16) & 0xFF);
    hdr[3] = static_cast<uint8_t>((len >> 24) & 0xFF);

#ifdef _WIN32
    return write_to_pipe(hdr, 4) && write_to_pipe(data, len);
#else
    return write_to_fd(conn_fd_, hdr, 4) && write_to_fd(conn_fd_, data, len);
#endif
}

// ---------------------------------------------------------------------------
// start_recv_loop
// ---------------------------------------------------------------------------
void IpcTransport::start_recv_loop(RecvCallback cb)
{
    recv_thread_ = std::thread([this, cb = std::move(cb)]() mutable {
        recv_loop(std::move(cb));
    });
}

// ---------------------------------------------------------------------------
// close
// ---------------------------------------------------------------------------
void IpcTransport::close()
{
    connected_ = false;

#ifdef _WIN32
    if (conn_pipe_ != INVALID_HANDLE_VALUE) {
        DisconnectNamedPipe(conn_pipe_);
        CloseHandle(conn_pipe_);
        conn_pipe_ = INVALID_HANDLE_VALUE;
    }
    if (listen_pipe_ != INVALID_HANDLE_VALUE) {
        CloseHandle(listen_pipe_);
        listen_pipe_ = INVALID_HANDLE_VALUE;
    }
#else
    if (conn_fd_ >= 0)   { ::close(conn_fd_);   conn_fd_   = -1; }
    if (server_fd_ >= 0) { ::close(server_fd_); server_fd_ = -1; }
    ::unlink(socket_path_.c_str());
#endif

    if (recv_thread_.joinable())
        recv_thread_.join();
}

// ---------------------------------------------------------------------------
// recv_loop (private)
// ---------------------------------------------------------------------------
void IpcTransport::recv_loop(RecvCallback cb)
{
    while (connected_) {
        // Read 4-byte length prefix.
        uint8_t hdr[4] = {};
#ifdef _WIN32
        if (!read_exact(conn_pipe_, hdr, 4)) break;
#else
        if (!read_exact(conn_fd_, hdr, 4)) break;
#endif
        uint32_t len =
            static_cast<uint32_t>(hdr[0])        |
            (static_cast<uint32_t>(hdr[1]) << 8)  |
            (static_cast<uint32_t>(hdr[2]) << 16) |
            (static_cast<uint32_t>(hdr[3]) << 24);

        if (len == 0 || len > 16 * 1024 * 1024) {
            spdlog::warn("[IpcTransport] invalid frame length {}", len);
            break;
        }

        std::vector<uint8_t> payload(len);
#ifdef _WIN32
        if (!read_exact(conn_pipe_, payload.data(), len)) break;
#else
        if (!read_exact(conn_fd_, payload.data(), len)) break;
#endif

        cb(std::move(payload));
    }

    spdlog::info("[IpcTransport] recv_loop exited");
    connected_ = false;
}

// ---------------------------------------------------------------------------
// Platform helpers
// ---------------------------------------------------------------------------
#ifdef _WIN32
bool IpcTransport::write_to_pipe(const uint8_t* data, uint32_t len)
{
    DWORD written = 0;
    DWORD remaining = len;
    const uint8_t* ptr = data;
    while (remaining > 0) {
        if (!WriteFile(conn_pipe_, ptr, remaining, &written, nullptr))
            return false;
        ptr       += written;
        remaining -= written;
    }
    return true;
}

bool IpcTransport::read_exact(HANDLE h, uint8_t* buf, DWORD n)
{
    DWORD done = 0;
    while (done < n) {
        DWORD got = 0;
        if (!ReadFile(h, buf + done, n - done, &got, nullptr) || got == 0)
            return false;
        done += got;
    }
    return true;
}
#else
bool IpcTransport::write_to_fd(const int fd, const uint8_t* data, uint32_t len)
{
    ssize_t remaining = static_cast<ssize_t>(len);
    const uint8_t* ptr = data;
    while (remaining > 0) {
        ssize_t sent = ::write(fd, ptr, static_cast<size_t>(remaining));
        if (sent <= 0) return false;
        ptr       += sent;
        remaining -= sent;
    }
    return true;
}

bool IpcTransport::read_exact(int fd, uint8_t* buf, size_t n)
{
    size_t done = 0;
    while (done < n) {
        ssize_t got = ::read(fd, buf + done, n - done);
        if (got <= 0) return false;
        done += static_cast<size_t>(got);
    }
    return true;
}
#endif

} // namespace truellm
