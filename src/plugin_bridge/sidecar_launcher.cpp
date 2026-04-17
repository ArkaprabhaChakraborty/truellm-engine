// ---------------------------------------------------------------------------
// sidecar_launcher.cpp
// ---------------------------------------------------------------------------

#include "sidecar_launcher.h"

#include <spdlog/spdlog.h>

#include <cstring>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#else
#  include <spawn.h>
#  include <sys/wait.h>
#  include <unistd.h>
extern char** environ;
#endif

// Helper: append "KEY=VALUE" to a copy of the current environment.
// Used to inject TRUELLM_SOCKET_PATH so sidecars can find the IPC pipe.

namespace truellm {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
namespace {

#ifdef _WIN32
// Builds a Windows command line string from a vector of arguments.
// Applies minimal quoting for arguments that contain spaces.
std::string build_cmdline(const std::vector<std::string>& args)
{
    std::ostringstream oss;
    for (size_t i = 0; i < args.size(); ++i) {
        if (i > 0) oss << ' ';
        bool needs_quote = args[i].find(' ') != std::string::npos;
        if (needs_quote) oss << '"';
        oss << args[i];
        if (needs_quote) oss << '"';
    }
    return oss.str();
}
#endif

// Resolves the sidecar executable.  Falls back to "python3" / "bun" if empty.
std::string resolve_executable(const SidecarConfig& cfg)
{
    if (!cfg.executable.empty()) return cfg.executable;
    switch (cfg.runtime) {
        case SidecarRuntime::Python:     return "python3";
        case SidecarRuntime::JavaScript: return "bun";
    }
    return "python3";
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// ~SidecarLauncher
// ---------------------------------------------------------------------------
SidecarLauncher::~SidecarLauncher()
{
    stop(3000);
}

// ---------------------------------------------------------------------------
// launch
// ---------------------------------------------------------------------------
bool SidecarLauncher::launch(const SidecarConfig& cfg, ExitCallback on_exit)
{
    on_exit_ = std::move(on_exit);

    std::string exe = resolve_executable(cfg);

    // Build argv: [executable, module_parts..., --config, path, --plugin-dir, dir, extra...]
    // The socket path is passed as TRUELLM_SOCKET_PATH env var (not CLI arg)
    // so that sidecars can read it with os.environ / process.env without
    // requiring a specific argument parser.
    std::vector<std::string> args;
    args.push_back(exe);

    if (!cfg.module.empty()) {
        // Module can be "-m truellm_engine" — split on space.
        std::istringstream iss(cfg.module);
        std::string tok;
        while (iss >> tok) args.push_back(tok);
    }

    args.push_back("--config");  args.push_back(cfg.config_path);
    if (!cfg.plugin_dir.empty()) {
        args.push_back("--plugin-dir"); args.push_back(cfg.plugin_dir);
    }
    for (const auto& a : cfg.extra_args) args.push_back(a);

    spdlog::info("[SidecarLauncher] spawning: {}", [&]() {
        std::string s;
        for (const auto& a : args) { s += a; s += ' '; }
        return s;
    }());

#ifdef _WIN32
    std::string cmdline = build_cmdline(args);
    STARTUPINFOA si{};
    PROCESS_INFORMATION pi{};
    si.cb = sizeof(si);

    // Build an environment block that inherits the current environment and
    // adds TRUELLM_SOCKET_PATH so the sidecar can find the IPC pipe.
    // The block is a sequence of "KEY=VALUE\0" entries terminated by "\0\0".
    std::string env_block;
    {
        // Snapshot existing environment.
        LPCH env_ptr = GetEnvironmentStringsA();
        if (env_ptr) {
            for (LPCH p = env_ptr; *p; p += strlen(p) + 1) {
                // Skip any pre-existing TRUELLM_SOCKET_PATH so we always
                // use the freshly generated path.
                if (strncmp(p, "TRUELLM_SOCKET_PATH=", 20) != 0) {
                    env_block.append(p);
                    env_block.push_back('\0');
                }
            }
            FreeEnvironmentStringsA(env_ptr);
        }
        env_block += "TRUELLM_SOCKET_PATH=" + cfg.socket_path;
        env_block.push_back('\0');
        env_block.push_back('\0'); // double-null terminator
    }

    if (!CreateProcessA(nullptr, cmdline.data(), nullptr, nullptr, FALSE,
                        0, env_block.data(), nullptr, &si, &pi)) {
        spdlog::error("[SidecarLauncher] CreateProcess failed: {}", GetLastError());
        return false;
    }

    CloseHandle(pi.hThread);
    process_handle_ = pi.hProcess;
    process_id_     = pi.dwProcessId;

    spdlog::info("[SidecarLauncher] sidecar PID {}", process_id_);
    start_monitor(on_exit_);
    return true;

#else
    // Build C-string argv for posix_spawnp.
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
    argv.push_back(nullptr);

    // Build an env array that inherits the current environment and injects
    // TRUELLM_SOCKET_PATH so the sidecar can find the Unix socket.
    std::string socket_env_entry = "TRUELLM_SOCKET_PATH=" + cfg.socket_path;
    std::vector<char*> envp;
    for (char** e = environ; *e; ++e) {
        if (strncmp(*e, "TRUELLM_SOCKET_PATH=", 20) != 0)
            envp.push_back(*e);
    }
    envp.push_back(const_cast<char*>(socket_env_entry.c_str()));
    envp.push_back(nullptr);

    pid_t pid = 0;
    int   rc  = posix_spawnp(&pid, exe.c_str(), nullptr, nullptr,
                              argv.data(), envp.data());
    if (rc != 0) {
        spdlog::error("[SidecarLauncher] posix_spawnp failed: {}", strerror(rc));
        return false;
    }

    process_pid_ = pid;
    spdlog::info("[SidecarLauncher] sidecar PID {}", pid);
    start_monitor(on_exit_);
    return true;
#endif
}

// ---------------------------------------------------------------------------
// stop
// ---------------------------------------------------------------------------
void SidecarLauncher::stop(int timeout_ms)
{
#ifdef _WIN32
    if (process_handle_ == INVALID_HANDLE_VALUE) return;
    TerminateProcess(process_handle_, 1);
    WaitForSingleObject(process_handle_, static_cast<DWORD>(timeout_ms));
    CloseHandle(process_handle_);
    process_handle_ = INVALID_HANDLE_VALUE;
    process_id_     = 0;
#else
    if (process_pid_ <= 0) return;
    ::kill(process_pid_, SIGTERM);

    int elapsed = 0;
    while (elapsed < timeout_ms) {
        int status = 0;
        pid_t r = ::waitpid(process_pid_, &status, WNOHANG);
        if (r == process_pid_) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        elapsed += 50;
    }
    // Force-kill if still running.
    ::kill(process_pid_, SIGKILL);
    ::waitpid(process_pid_, nullptr, 0);
    process_pid_ = 0;
#endif
}

// ---------------------------------------------------------------------------
// is_running
// ---------------------------------------------------------------------------
bool SidecarLauncher::is_running() const
{
#ifdef _WIN32
    if (process_handle_ == INVALID_HANDLE_VALUE) return false;
    DWORD code = 0;
    GetExitCodeProcess(process_handle_, &code);
    return code == STILL_ACTIVE;
#else
    if (process_pid_ <= 0) return false;
    return ::kill(process_pid_, 0) == 0;
#endif
}

// ---------------------------------------------------------------------------
// pid
// ---------------------------------------------------------------------------
uint64_t SidecarLauncher::pid() const
{
#ifdef _WIN32
    return static_cast<uint64_t>(process_id_);
#else
    return static_cast<uint64_t>(process_pid_);
#endif
}

// ---------------------------------------------------------------------------
// start_monitor — background thread that waits for process exit.
// ---------------------------------------------------------------------------
void SidecarLauncher::start_monitor(ExitCallback cb)
{
    if (!cb) return;

    std::thread([this, cb = std::move(cb)]() {
#ifdef _WIN32
        WaitForSingleObject(process_handle_, INFINITE);
        DWORD code = 1;
        GetExitCodeProcess(process_handle_, &code);
        spdlog::warn("[SidecarLauncher] sidecar exited with code {}", code);
        cb(static_cast<int>(code));
#else
        int status = 0;
        ::waitpid(process_pid_, &status, 0);
        int code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
        spdlog::warn("[SidecarLauncher] sidecar exited with code {}", code);
        cb(code);
#endif
    }).detach();
}

} // namespace truellm
