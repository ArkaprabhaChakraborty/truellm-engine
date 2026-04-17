#pragma once
// ---------------------------------------------------------------------------
// application.h — Top-level orchestrator
//
// Owns the engine and HTTP server lifetime.  main.cpp creates one of these,
// calls init(), then run() which blocks until request_shutdown() is called
// (typically from the signal handler).
// ---------------------------------------------------------------------------

#include <truellm/config_schema.h>
#include <truellm/types.h>

#include <memory>

namespace truellm {

class EngineInterface;
class HttpServer;
class PluginBridge;

class Application {
public:
    explicit Application(TrueLLMConfig cfg);
    ~Application();

    // Non-copyable
    Application(const Application&)            = delete;
    Application& operator=(const Application&) = delete;

    // Initialise engine + plugin bridge + HTTP server.  Returns Ok on success.
    ErrorCode init();

    // Blocks until request_shutdown() is called.
    void run();

    // Safe to call from a signal handler (sets an atomic flag).
    void request_shutdown();

private:
    TrueLLMConfig                    cfg_;
    std::unique_ptr<EngineInterface> engine_;
    std::unique_ptr<PluginBridge>    plugin_bridge_;
    std::unique_ptr<HttpServer>      http_;
};

} // namespace truellm
