# Plugin Development

## What plugins are for

Out of the box, TrueLLM can generate text from a model. What it cannot do
is act on the world: fetch a live web page, query a database, call an
internal API, or apply a content policy that lives outside the engine.
Plugins are how you add those capabilities without modifying or recompiling
the engine.

A plugin registers one or more hooks at load time. The engine calls those
hooks at defined points in the request pipeline. The plugin author writes
only the hook logic; the engine handles scheduling, error recovery, IPC
framing, and memory cleanup.

There are three hook types in use today:

| Hook type | Constant | When the engine calls it |
|---|---|---|
| Tool provider | `TRUELLM_HOOK_TOOL_PROVIDER` | When the model generates a function call during generation |
| Preprocessor | `TRUELLM_HOOK_PREPROCESSOR` | Before tokenization, on every request |
| Generator | `TRUELLM_HOOK_GENERATOR` | Replaces the generation loop entirely (reserved; not yet dispatched) |

A single plugin can implement any combination of these hooks. Set the
output pointers you do not use to `nullptr` in `truellm_plugin_init`.

---

## The Plugin ABI

The C ABI (`include/truellm/plugin.h`) is the stable contract between the
engine and every plugin, regardless of language. It will not change without
a version bump to `TRUELLM_ABI_VERSION`. Always fill in `abi_version` and
`struct_size` in every provider struct -- the engine checks these before
calling any callbacks.

**Memory ownership rule:** Strings the plugin returns to the engine are
allocated by the plugin (or via `host->alloc`) and freed by the engine
through the `free_fn` / `free_arg` pair in the result struct. Strings the
engine passes to the plugin (tool name, args_json, messages_json, etc.)
are engine-owned and valid only for the duration of that function call.
Copy them if you need to retain them.

The engine provides `host->alloc` / `host->dealloc` so that plugin result
payloads can be freed safely even across DLL boundaries (important on
Windows where each DLL can have a separate CRT heap).

---

## Native Plugin (C++)

A native plugin is a shared library (`.dll` on Windows, `.so` on Linux)
that the engine loads with `LoadLibrary`/`dlopen`. It must export one
symbol: `truellm_plugin_init`.

### When to choose native

- Maximum performance -- no IPC round-trip; tool call dispatch is a
  direct function pointer call
- Access to C/C++ libraries that are hard to wrap in Python or JS
- The tool_web_fetch plugin shipped with TrueLLM is a native plugin for
  exactly this reason: it uses cpp-httplib directly

### Minimal example

```cpp
// my_plugin.cpp
#include <truellm/plugin.h>
#include <truellm/host_api.h>

#include <cstring>
#include <string>

static const truellm_host_api_t* g_host = nullptr;

// ── Tool definition ───────────────────────────────────────────────────────
// The engine reads this struct once during registration.
// All pointers must remain valid until destroy() is called.
static const truellm_tool_def_t s_my_tool_def = {
    /*name=*/            "my_tool",
    /*description=*/     "Returns the answer to everything.",
    /*parameters_json=*/ R"json({"type":"object","properties":{}})json"
};

static const truellm_tool_def_t* s_tools[] = { &s_my_tool_def, nullptr };

static const truellm_tool_def_t* const* get_tools(void* /*state*/)
{
    return s_tools;
}

// ── Tool call handler ─────────────────────────────────────────────────────
// Called by the engine when the model generates a call to "my_plugin.my_tool".
// args_json is engine-owned and valid only during this call.
static truellm_tool_result_t call_tool(void*                    /*state*/,
                                       const truellm_context_t* /*ctx*/,
                                       const char*              /*tool_name*/,
                                       const char*              /*args_json*/)
{
    const char* json = R"json({"answer":42})json";
    size_t len = strlen(json);

    // Allocate via the engine heap so the engine can free the payload safely.
    char* buf = static_cast<char*>(g_host->alloc(len + 1));
    if (!buf) return { TRUELLM_ERR_INTERNAL, nullptr, nullptr, nullptr };
    memcpy(buf, json, len + 1);

    truellm_tool_result_t result{};
    result.error    = TRUELLM_OK;
    result.payload  = buf;
    result.free_fn  = g_host->dealloc;
    result.free_arg = buf;
    return result;
}

static void destroy_provider(void* /*state*/) {}

static truellm_tool_provider_t s_provider = {
    /*abi_version=*/    TRUELLM_ABI_VERSION,
    /*abi_minor=*/      TRUELLM_ABI_MINOR,
    /*struct_size=*/    sizeof(truellm_tool_provider_t),
    /*plugin_id=*/      "my_plugin",
    /*plugin_version=*/ "1.0.0",
    /*get_tools=*/      get_tools,
    /*call_tool=*/      call_tool,
    /*state=*/          nullptr,
    /*destroy=*/        destroy_provider
};

// ── Entry point ───────────────────────────────────────────────────────────
// Called once when the engine loads the DLL.  Return TRUELLM_OK to confirm
// successful init; any other code aborts the load.
extern "C" truellm_error_t truellm_plugin_init(
    const truellm_host_api_t* host,
    truellm_tool_provider_t** tool_out,
    truellm_preprocessor_t**  prep_out,
    truellm_generator_t**     gen_out)
{
    g_host    = host;
    *tool_out = &s_provider;
    *prep_out = nullptr;
    *gen_out  = nullptr;
    return TRUELLM_OK;
}
```

### Returning errors from a tool call

If your tool encounters an error, set `error` to a non-OK code and put a
human-readable description in `payload`. The engine will include the error
in the message passed back to the model so it can handle it gracefully.

```cpp
truellm_tool_result_t call_tool(...)
{
    // something went wrong
    const char* msg = "upstream service unavailable";
    size_t len = strlen(msg);
    char* buf = static_cast<char*>(g_host->alloc(len + 1));
    memcpy(buf, msg, len + 1);
    return { TRUELLM_ERR_INTERNAL, buf, g_host->dealloc, buf };
}
```

### Reading plugin config

Values from `[plugins.settings]` in the server TOML are accessible via
`host->get_config`. The lookup tries `"plugin_id.key"` first, then `"key"`.

```cpp
// In server TOML:
// [plugins.settings]
// my_plugin.api_url = "https://api.example.com"
// my_plugin.timeout_ms = "3000"

const char* url = g_host->get_config("my_plugin", "api_url");
if (url) { /* use url */ }

const char* timeout_str = g_host->get_config("my_plugin", "timeout_ms");
int timeout_ms = timeout_str ? atoi(timeout_str) : 5000;
```

The return value is engine-owned and valid only until the next call to
`get_config`. Copy the string if you need to retain it past the call.

### Accessing host model metadata

The host API exposes model metadata that is valid after model load:

```cpp
const char* arch = g_host->model_arch();        // e.g. "llama", "qwen2"
int64_t ctx_len  = g_host->model_context_len(); // maximum context tokens
```

### Logging from a plugin

```cpp
g_host->log_info ("my_plugin", "tool call started");
g_host->log_warn ("my_plugin", "retrying after transient error");
g_host->log_error("my_plugin", "fatal: cannot connect to upstream");
g_host->log_debug("my_plugin", "parsed args OK");
```

Log lines appear in the server log at the corresponding level and are
prefixed with the plugin ID.

### Building the native plugin

```cmake
# In CMakeLists.txt at the project root, or in a separate CMakeLists.txt
add_library(my_plugin SHARED my_plugin.cpp)

target_include_directories(my_plugin PRIVATE
    path/to/truellm/include
)

# Output into plugins/native/ to keep it separate from the server binary
set_target_properties(my_plugin PROPERTIES
    PREFIX ""
    OUTPUT_NAME "my_plugin"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins/native"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins/native"
)
```

### Registering the built plugin

In your TOML config, point `plugin_dirs` at the output directory:

```toml
[plugins]
enabled      = true
plugin_dirs  = ["build/windows-cuda/plugins/native"]
```

The engine scans each directory for `.dll` / `.so` files, loads them, calls
`truellm_plugin_init`, and registers the resulting providers. All tools
registered by native plugins are immediately available to the model.

---

## Python Sidecar

Python sidecars run as separate subprocesses. The engine spawns the process,
connects to it over a named pipe or Unix socket, and exchanges JSON messages.
This means your Python code runs in its own interpreter with no GIL sharing
or ABI concerns.

### When to choose Python

- You need third-party Python libraries (requests, boto3, psycopg2, etc.)
- The tool logic is complex enough that C++ would be slower to write
- You want to iterate quickly without a compile step

### Directory layout

```
plugins/python/my_plugin/
  manifest.toml   -- declares plugin identity, runtime, and hooks
  main.py         -- the entry point the engine spawns
```

### manifest.toml

```toml
id           = "my_plugin"
display_name = "My Python Plugin"
version      = "1.0.0"
runtime      = "python"
entry        = "main.py"

[[hooks]]
type = "tool_provider"

[[tools]]
name        = "my_plugin.search"
description = "Search for information."
parameters  = '''
{
  "type": "object",
  "properties": {
    "query": {"type": "string", "description": "Search query"}
  },
  "required": ["query"]
}
'''
```

### main.py

```python
import json
import os
import socket

SOCKET_PATH = os.environ.get("TRUELLM_SOCKET_PATH", "")

def handle_announce():
    return {
        "type": "plugin_announce",
        "cid": 0,
        "plugins": [{
            "plugin_id":    "my_plugin",
            "display_name": "My Python Plugin",
            "version":      "1.0.0",
            "hook_types":   ["tool_provider"],
            "tool_defs":    [{
                "name":           "search",
                "qualified_name": "my_plugin.search",
                "description":    "Search for information.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            }]
        }]
    }

def handle_tool_call(msg):
    args  = json.loads(msg.get("args_json", "{}"))
    query = args.get("query", "")
    # Your logic here
    result = {"results": [f"Result for: {query}"]}
    return {
        "type":    "tool_response",
        "cid":     msg["cid"],
        "ok":      True,
        "payload": json.dumps(result)
    }

def send_message(conn, msg):
    data = json.dumps(msg).encode("utf-8")
    conn.sendall(len(data).to_bytes(4, "little") + data)

def recv_message(conn):
    header = conn.recv(4)
    if len(header) < 4:
        return None
    length = int.from_bytes(header, "little")
    data = b""
    while len(data) < length:
        chunk = conn.recv(length - len(data))
        if not chunk:
            return None
        data += chunk
    return json.loads(data.decode("utf-8"))

def main():
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.connect(SOCKET_PATH)
    send_message(conn, handle_announce())  # announce immediately after connect
    while True:
        msg = recv_message(conn)
        if msg is None:
            break
        msg_type = msg.get("type")
        if msg_type == "tool_call":
            send_message(conn, handle_tool_call(msg))
        elif msg_type == "heartbeat":
            # Ack the heartbeat so the engine knows the sidecar is alive.
            send_message(conn, {"type": "heartbeat_ack", "cid": msg.get("cid", 0)})

if __name__ == "__main__":
    main()
```

### Config

```toml
[plugins]
enabled     = true
python_path = "python3"
plugin_dirs = ["./plugins/python/my_plugin"]

[plugins.settings]
my_plugin.api_key = "${ENV:MY_PLUGIN_API_KEY}"
```

The engine passes the socket path to the sidecar process via the
`TRUELLM_SOCKET_PATH` environment variable.

---

## JavaScript / TypeScript Sidecar

Same IPC protocol as Python. Set `runtime = "node"` or `runtime = "bun"`
in `manifest.toml`.

### main.ts (Node.js)

```typescript
import * as net from "net";

const SOCKET_PATH = process.env.TRUELLM_SOCKET_PATH ?? "";

function sendMessage(sock: net.Socket, msg: object) {
    const data   = Buffer.from(JSON.stringify(msg), "utf-8");
    const header = Buffer.alloc(4);
    header.writeUInt32LE(data.length);
    sock.write(Buffer.concat([header, data]));
}

const sock = net.createConnection(SOCKET_PATH, () => {
    sendMessage(sock, {
        type: "plugin_announce", cid: 0,
        plugins: [{
            plugin_id:    "my_js_plugin",
            display_name: "My JS Plugin",
            version:      "1.0.0",
            hook_types:   ["tool_provider"],
            tool_defs:    [{
                name:           "greet",
                qualified_name: "my_js_plugin.greet",
                description:    "Return a greeting.",
                parameters: {
                    type: "object",
                    properties: { name: { type: "string" } },
                    required: ["name"]
                }
            }]
        }]
    });
});

let buf = Buffer.alloc(0);
sock.on("data", (chunk: Buffer) => {
    buf = Buffer.concat([buf, chunk]);
    while (buf.length >= 4) {
        const len = buf.readUInt32LE(0);
        if (buf.length < 4 + len) break;
        const msg = JSON.parse(buf.subarray(4, 4 + len).toString("utf-8"));
        buf = buf.subarray(4 + len);

        if (msg.type === "tool_call") {
            const args = JSON.parse(msg.args_json ?? "{}");
            sendMessage(sock, {
                type:    "tool_response",
                cid:     msg.cid,
                ok:      true,
                payload: JSON.stringify({ greeting: `Hello, ${args.name}!` })
            });
        } else if (msg.type === "heartbeat") {
            sendMessage(sock, { type: "heartbeat_ack", cid: msg.cid });
        }
    }
});
```

---

## Preprocessor Hook

A preprocessor runs on every request before tokenization. It can inspect
or modify the message array, short-circuit with a direct reply, or abort
the request entirely -- all without touching the inference engine.

This is useful for:
- Content filtering before the model ever sees a request
- Request enrichment (appending system context, injecting RAG results)
- Access control based on message content

Decision values: `0` = pass, `1` = respond (use `response_text`), `2` = abort.

### Python preprocessor example

```python
def handle_preprocess(msg):
    messages = json.loads(msg.get("messages_json", "[]"))
    cid      = msg["cid"]

    # Reject messages containing disallowed keywords
    for m in messages:
        if "forbidden" in m.get("content", "").lower():
            return {
                "type":          "preprocess_response",
                "cid":           cid,
                "decision":      2,   # abort
                "response_text": "Request contains disallowed content."
            }

    # Append a RAG result to the system message
    for m in messages:
        if m.get("role") == "system":
            m["content"] += "\n\nContext: [retrieved document here]"

    return {
        "type":                   "preprocess_response",
        "cid":                    cid,
        "decision":               0,   # pass through with (optionally modified) messages
        "modified_messages_json": json.dumps(messages)
    }
```

In your `main.py` message loop, dispatch on `msg.get("type") == "preprocess"`.

### Native preprocessor example (C++)

Implement `truellm_preproc_result_t preprocess(state, ctx, messages_json)` and set
`*prep_out` in `truellm_plugin_init`. Leave `*tool_out = nullptr` if the plugin
is preprocessor-only.

---

## Sidecar Restart

If a sidecar process crashes or exits with a non-zero code, the engine
restarts it automatically when `restart_on_crash = true` in `[plugins]`.

Restarts use exponential backoff so a repeatedly crashing plugin does not
spin-loop:

| Attempt | Delay |
|---|---|
| 1st | 1 s |
| 2nd | 2 s |
| 3rd | 4 s |
| 4th+ | capped at 30 s |

After `max_restart_attempts` consecutive restarts (default 3), the engine
disables the plugin and logs an error. It remains disabled until the server
is restarted.

---

## Plugin Config Bridge

Every plugin can read config values from `[plugins.settings]` in the
server TOML without parsing the TOML itself. The engine resolves environment
variable substitutions (`${ENV:...}`) before handing values to plugins, so
plugins always receive the final string.

TOML:
```toml
[plugins.settings]
my_plugin.timeout_ms  = "5000"
my_plugin.endpoint    = "https://api.example.com"
my_plugin.api_key     = "${ENV:MY_PLUGIN_API_KEY}"
global_log_level      = "debug"
```

In a native plugin:
```cpp
// Lookup order: "my_plugin.timeout_ms" -> "timeout_ms"
const char* timeout = host->get_config("my_plugin", "timeout_ms");
// Returns "5000"

const char* level = host->get_config("my_plugin", "global_log_level");
// Returns "debug" (global fallback, no plugin prefix)
```

In a Python sidecar, the engine does not currently forward `[plugins.settings]`
over IPC. Read per-plugin values from environment variables set via the
`${ENV:...}` syntax in the TOML:

```toml
[plugins.settings]
my_plugin.api_key = "${ENV:MY_PLUGIN_API_KEY}"
```

```python
import os
api_key = os.environ.get("MY_PLUGIN_API_KEY", "")
```

---

## Testing Your Plugin

The fastest way to verify a native plugin compiles and loads is:

```bash
# 1. Build
cmake --build build/windows-cuda --parallel

# 2. Check the DLL exists in the isolated output directory
ls build/windows-cuda/plugins/native/

# 3. Start server with the web-fetch scenario config
.\build\windows-cuda\bin\truellm-server.exe --config configs\scenario-web-fetch.toml

# 4. Verify plugin is loaded
curl http://127.0.0.1:9099/truellm/v1/plugins

# 5. Run the integration test
python tests\integration\python\test_web_fetch_plugin.py
```

The integration test (`test_web_fetch_plugin.py`) performs four checks:
1. Server health
2. Plugin loaded (DLL found and registered)
3. fetch_url tool advertised in the tool schema
4. End-to-end tool dispatch (fetches the server's own `/health` endpoint)
