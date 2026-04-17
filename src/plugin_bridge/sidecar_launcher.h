#pragma once
// ---------------------------------------------------------------------------
// sidecar_launcher.h — Spawns and monitors Python / JS sidecar processes.
// ---------------------------------------------------------------------------

#include <functional>
#include <string>
#include <vector>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <sys/types.h>
#endif

namespace truellm {

enum class SidecarRuntime { Python, JavaScript };

struct SidecarConfig {
    SidecarRuntime runtime     = SidecarRuntime::Python;
    std::string    executable;   // "python3" or "bun" — resolved from PATH if empty
    std::string    module;       // "-m truellm_engine" or path to JS entry
    std::string    socket_path;  // IPC socket path to pass as argument
    std::string    config_path;  // server.toml path forwarded to sidecar
    std::string    plugin_dir;   // directory scanned for Python/JS plugins
    std::vector<std::string> extra_args;
};

// ---------------------------------------------------------------------------
// SidecarLauncher
// ---------------------------------------------------------------------------
class SidecarLauncher {
public:
    using ExitCallback = std::function<void(int exit_code)>;

    SidecarLauncher() = default;
    ~SidecarLauncher();

    // Non-copyable.
    SidecarLauncher(const SidecarLauncher&)            = delete;
    SidecarLauncher& operator=(const SidecarLauncher&) = delete;

    // Spawns the sidecar process.  Returns true on success.
    bool launch(const SidecarConfig& cfg, ExitCallback on_exit = nullptr);

    // Sends SIGTERM (POSIX) / TerminateProcess (Windows) and waits up to
    // timeout_ms for the process to exit.
    void stop(int timeout_ms = 3000);

    // Returns true if the process is still running.
    bool is_running() const;

    // Returns the platform process ID (0 if not launched).
    uint64_t pid() const;

private:
#ifdef _WIN32
    HANDLE process_handle_  = INVALID_HANDLE_VALUE;
    HANDLE monitor_thread_  = INVALID_HANDLE_VALUE;
    DWORD  process_id_      = 0;
#else
    pid_t  process_pid_     = 0;
    int    monitor_pipe_[2] = {-1, -1};
#endif

    ExitCallback on_exit_;
    void start_monitor(ExitCallback cb);
};

} // namespace truellm
