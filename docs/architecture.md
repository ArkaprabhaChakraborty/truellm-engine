# Architecture

## Why TrueLLM is structured the way it is

Most LLM inference servers conflate concerns: the HTTP layer talks directly
to CUDA, the model loader is tangled with the sampling loop, and plugins
are afterthoughts bolted onto the outside. This works fine for a single use
case but falls apart when you need to swap backends, add rate limiting, or
extend behavior without recompiling the core.

TrueLLM separates concerns into four independent subsystems that communicate
through stable, narrow interfaces. The payoff: you can swap the inference
backend without touching the server layer, add a plugin without knowing
anything about attention kernels, and enforce request policies without
modifying routers.

```
                           HTTP (cpp-httplib)
                                  |
                         +--------v--------+
                         |   src/server/   |
                         |  HttpServer     |  <-- middleware/ (auth, logging)
                         |  OpenAIRouter   |      <-- src/policy/ (rate limiter)
                         |  NativeRouter   |
                         +---+--------+----+
                             |        |
              +--------------v--+  +--v--------------+
              | src/inference/  |  | src/plugin_bridge/|
              |  EngineInterface|  |  PluginBridge    |
              |  GgmlEngine(CPU)|  |  ToolRegistry    |
              |  CudaEngine(GPU)|  |  PreprocessorChain|
              +-----------------+  |  SidecarLauncher |
                                   +-----------------+
                                          |
                              Native .dll/.so, Python, JS
```

---

## Module Boundaries

### `src/inference/` -- the engine abstraction

**The problem it solves:** CUDA code cannot live in the same translation
unit as HTTP handling code without forcing every developer to install the
CUDA toolkit and turning simple changes into 10-minute rebuilds. More
importantly, CPU and GPU inference have fundamentally different memory
models: the CPU backend uses llama.cpp's standard KV cache while the GPU
backend uses a vLLM-style paged block allocator.

**What it provides:** A single pure-abstract interface, `EngineInterface`,
that makes the rest of the codebase unaware of which backend is running.
The interface exposes five operations: `load_model`, `generate`, `tokenize`,
`get_gpu_stats`, `get_kv_cache_stats`. Nothing above this layer includes a
CUDA header.

**Key components:**

- `EngineInterface` -- abstract base; the only type the server and plugin
  layers ever hold
- `GgmlEngine` -- CPU backend via llama.cpp/ggml; also handles CPU-side
  GPU layer offload when a CUDA GPU is present but the paged engine is off
- `CudaEngine` -- paged-attention CUDA backend; vLLM-style block allocator,
  separate prefill/decode/copy CUDA stream pool, optional FP8 KV cache on
  sm_89+ (Ada Lovelace, Hopper)
- `BatchScheduler` -- iteration-level continuous batching coordinator;
  manages the queue of pending sequences and decides which to decode next
- `GpuDevice` -- VRAM query helper; stubs to zero in CPU-only builds so
  the rest of the code never needs an `#ifdef`
- `engine_factory.cpp` -- reads `InferenceConfig` and calls `new GgmlEngine`
  or `new CudaEngine`; the only place where backend selection happens

**Design rule:** Backend selection is a one-time factory decision at
startup. Everything above `engine_factory.cpp` uses `EngineInterface*`.

---

### `src/plugin_bridge/` -- the extension point

**The problem it solves:** An LLM server is most useful when it can do
things the base model cannot: fetch live data, query databases, call
internal APIs. Hard-coding these capabilities means a recompile for every
new tool. Plugins solve this by letting external code register tools and
preprocessors at runtime without touching the core.

**What it provides:** A safe, versioned C ABI (defined in
`include/truellm/plugin.h`) that native, Python, and JavaScript plugins
all target. The bridge translates between the ABI and the internal
`ToolRegistry`/`PreprocessorChain` types.

**Plugin types and how they work:**

- **Native (C ABI):** A `.dll`/`.so` loaded via `NativeLoader`.
  Must export `truellm_plugin_init(host, tool_out, prep_out, gen_out)`.
  The engine passes a `truellm_host_api_t*` vtable at load time; the
  plugin uses it to log, read config, and allocate result memory.
  Memory ownership is explicit: the plugin allocates result payloads,
  the engine frees them via the `free_fn` pointer in the result struct.

- **Python sidecar:** A subprocess spawned by `SidecarLauncher` from a
  directory containing `manifest.toml + main.py`. Communicates over a
  named pipe (Windows) or Unix socket (POSIX) using length-prefixed JSON.
  The engine sends `tool_call` messages; the sidecar replies with
  `tool_response` messages. A correlation ID (`uint64_t`) ties each
  request to its reply.

- **JavaScript sidecar:** Same IPC protocol as Python. The `runtime`
  field in `manifest.toml` selects `"node"` or `"bun"`.

**Key internal components:**

- `PluginBridge` -- top-level coordinator; owns all plugin state,
  starts sidecars, wires the NativeLoader, and exposes the combined
  tool/preprocessor view to the server
- `ToolRegistry` -- thread-safe store of tool entries (both native
  `truellm_tool_provider_t*` and remote IPC dispatch lambdas);
  uses `std::shared_mutex` so concurrent tool calls share the read lock
- `PreprocessorChain` -- ordered chain of preprocessor hooks; each hook
  runs in priority order and can pass, modify, respond, or abort
- `IpcClient` -- correlation-ID based request/response map over
  `IpcTransport`; each in-flight call parks a promise in `pending_`
  until the matching response arrives
- `HealthMonitor` -- sends periodic heartbeats to sidecars; calls
  `launcher->stop()` on timeout, triggering exponential-backoff restart
  in `PluginBridge::on_sidecar_exit()`

**Adding your own plugin:** See [plugin-development.md](plugin-development.md).
No changes to the engine are required for new native plugins or sidecars.

---

### `src/server/` -- the HTTP layer

**The problem it solves:** The HTTP layer needs to stay thin. Routers
that import inference headers create a dependency web that makes testing
and refactoring painful. The server layer achieves separation by holding
only `EngineInterface*` and `PluginBridge*` pointers -- it never reaches
into backend internals.

**What it provides:** An OpenAI-compatible REST API that any OpenAI client
library can target, plus a native status/plugin API. Middleware (auth, logging,
rate limiting) is applied uniformly before any request reaches a router.

**Key components:**

- `HttpServer` -- owns the cpp-httplib server instance; installs the
  pre-routing handler (timestamp injection + rate limiting), registers
  routes, and holds the `RateLimiter` and middleware references
- `OpenAIRouter` -- handles `/v1/models`, `/v1/completions`,
  `/v1/chat/completions`; runs the preprocessor chain then the tool call
  loop (up to 5 rounds) before the final generation pass
- `NativeRouter` -- handles `/truellm/v1/status` and `/truellm/v1/plugins`;
  returns engine metadata and plugin state as JSON
- `middleware/auth_middleware` -- checks the `Authorization: Bearer` header
  against `server.api_key`; returns 401 if the key is wrong; passes through
  when `api_key` is empty
- `middleware/logging_middleware` -- injects `X-Request-Start-Us` headers
  and logs method, path, source IP, status, and elapsed time at debug level

---

### `src/policy/` -- request enforcement

**The problem it solves:** A server that accepts unbounded requests will
OOM or get saturated quickly. Rate limiting and token caps need to be
consistent across all routes and composable with other middleware. Putting
this logic in individual routers leads to duplication and inconsistency.

**What it provides:** Two stateless policy objects that are configured once
at startup from the guardrail preset and then called from the HTTP layer.

- `RateLimiter` -- token bucket per client IP; the bucket size and refill
  rate are derived from `hardware.guardrails.preset`:

  | Preset     | Rate (req/s) | Burst |
  |------------|-------------|-------|
  | `strict`   | 5           | 10    |
  | `balanced` | 20          | 40    |
  | `relaxed`  | disabled    | --    |
  | `off`      | disabled    | --    |

  `try_acquire(ip)` is called in the pre-routing handler before auth, so
  rate-limited requests never reach the inference layer.

- `GuardrailPolicy` -- maps preset names to `max_tokens` and
  `max_ctx_tokens` limits; `check_request()` validates a request before
  inference starts; `clamp_max_tokens()` caps generation length so a
  single request cannot consume the entire context.

---

### `src/host/` -- application lifecycle glue

**The problem it solves:** Something needs to construct the subsystems in
the right order, hand them their dependencies, and tear them down cleanly
on SIGINT/SIGTERM. That logic does not belong in `main.cpp` and does not
belong in any of the subsystems themselves.

**What it provides:**

- `Application` -- constructs `EngineInterface` (via factory), then
  `PluginBridge`, then `HttpServer`; wires them together and handles
  clean shutdown
- `config/config.cpp` -- TOML parser that populates `TrueLLMConfig` using
  toml11 v4
- `config/profile.cpp` -- named preset defaults applied before the TOML
  file overrides; presets are `local-cpu`, `workstation`, `server`
- `config/secrets.cpp` -- `${ENV:VAR}` substitution in string config
  values; runs at parse time so secrets never appear in logs

---

## Request Flow (chat completions)

This is the full path of a `POST /v1/chat/completions` request:

```
POST /v1/chat/completions
  |
  +--> Pre-routing handler (HttpServer)
  |      +--> Inject X-Request-Start-Us header
  |      +--> RateLimiter::try_acquire(client_ip)  -- 429 if exhausted
  |
  +--> HttpServer::check_auth()                    -- 401 if key wrong
  |
  +--> OpenAIRouter::handle_chat_completions()
         |
         +--> GuardrailPolicy::check_request()     -- 400 if over limit
         |
         +--> PreprocessorChain::run()
         |      Each preprocessor hook runs in priority order.
         |      Decision: pass (continue), respond (bypass inference),
         |                abort (return error)
         |
         +--> apply_chat_template()                -- messages[] -> prompt string
         +--> engine_->tokenize()                  -- text -> token IDs
         |
         +--> Tool call loop  (up to 5 rounds):
         |      +--> engine_->generate()           -- with tool schemas in system prompt
         |      +--> parse tool calls from generated text
         |      +--> ToolRegistry::dispatch()      -- routes to native or IPC handler
         |      +--> append tool result to prompt
         |
         +--> engine_->generate()                  -- final answer generation
         +--> logging_middleware                    -- log status + elapsed
         +--> return JSON response
```

---

## Config Loading Order

```
compiled-in defaults     (config_schema.h field initializers)
  |
  v
profile preset           (profile.cpp  apply_profile())
  |
  v
TOML file fields         (config.cpp   load())
  |
  v
CLI flags                (main.cpp     override specific fields)
```

Later steps override earlier ones. A field absent from the TOML file
retains its profile/default value. CLI flags are the highest priority
and always win.

---

## Thread Safety

| Component | Mechanism | Notes |
|---|---|---|
| `EngineInterface::generate()` | mutex in router | one inference at a time |
| `ToolRegistry` | `std::shared_mutex` | concurrent reads share; writes are exclusive |
| `PluginBridge::loaded_plugins_` | `plugins_mu_` | protects plugin list during load/unload |
| `IpcClient::pending_` | `pending_mu_` | per-client correlation ID map |
| `RateLimiter::buckets_` | `mu_` | per-IP bucket map |

---

## CUDA Paged-Attention Architecture

When `backend = "cuda"` and `TRUELLM_BUILD_CUDA_ENGINE=ON`, the
`CudaEngine` uses a vLLM-style paged KV cache:

- KV memory is divided into fixed-size blocks (`paged_kv_block_size`
  tokens each), managed by `CudaAllocator`
- Each active sequence holds a slot map: logical block index -> physical
  block index; this is the paging mechanism that allows sequences of
  varying length to share a fixed memory pool without fragmentation
- Three CUDA streams run concurrently: Prefill (prompt evaluation),
  Decode (autoregressive generation), and Copy (host-to-device weight
  transfers); the scheduler pipelines work across them
- CUDA graphs can be captured for fixed-batch decode steps, eliminating
  kernel launch overhead on repeated decode calls with the same batch size
- FP8 KV cache halves KV memory vs F16 at negligible quality cost;
  available on sm_89+ (Ada Lovelace RTX 4xxx, Hopper H100, Blackwell)

---

## Plugin IPC Protocol

The sidecar IPC uses length-prefixed JSON over a named pipe (Windows) or
Unix domain socket (POSIX). The framing is the same for all message types:

```
[ uint32_t little-endian payload_length ][ payload_length bytes UTF-8 JSON ]
```

Message types (direction: sidecar -> engine or engine -> sidecar):

| Type | Direction | Purpose |
|---|---|---|
| `plugin_announce` | sidecar -> engine | Announces tool/preprocessor capabilities at connect time. Uses `cid=0`. |
| `tool_call` | engine -> sidecar | Invokes a named tool with `args_json`. |
| `tool_response` | sidecar -> engine | Returns the tool result for the matching `cid`. |
| `preprocess` | engine -> sidecar | Sends the current `messages_json` to a preprocessor hook. |
| `preprocess_response` | sidecar -> engine | Returns decision (0=pass, 1=respond, 2=abort) and optionally modified messages. |
| `status_report` | sidecar -> engine | Unsolicited health/status update from the sidecar. |
| `abort` | engine -> sidecar | Cancels an in-flight call by `cid`. |

Correlation IDs (`uint64_t`) pair each request with its response.
`plugin_announce` uses `cid=0` and is routed directly to the `announce_cb_`
callback before any correlation-ID lookup occurs.
