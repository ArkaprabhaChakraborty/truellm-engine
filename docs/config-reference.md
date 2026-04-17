# TrueLLM Server Configuration Reference

TrueLLM reads its configuration from a TOML file at startup. Almost every
aspect of server behavior -- bind address, model path, inference backend,
GPU memory allocation, plugins, rate limits, logging -- is expressed in a
single file rather than a mixture of CLI flags, environment variables, and
code-level constants.

The intent is that you should be able to reproduce any running configuration
by capturing its TOML file and the model path. There are no hidden defaults
that differ between environments. A field absent from the TOML file uses
the same compiled-in default on every machine.

This document describes every configuration section and field supported by
`server.toml` (and any TOML file passed via `--config`).

## Table of Contents

1. [How Configuration Works](#how-configuration-works)
2. [CLI Flags](#cli-flags)
3. [[server]](#server)
4. [[model]](#model)
5. [[inference]](#inference)
   - [[inference.kv_cache]](#inferencekv_cache)
   - [[inference.cuda]](#inferencecuda)
   - [[inference.cuda_engine]](#inferencecuda_engine)
   - [[inference.batching]](#inferencebatching)
6. [[sampling]](#sampling)
7. [[logging]](#logging)
8. [[hardware]](#hardware)
   - [[hardware.gpu]](#hardwaregpu)
   - [[hardware.offload]](#hardwareoffload)
   - [[hardware.guardrails]](#hardwareguardrails)
9. [[runtime]](#runtime)
   - [[runtime.harmony]](#runtimeharmony)
   - [[[runtime.extensions]]](#runtimeextensions)
10. [[plugins]](#plugins)
    - [[plugins.settings]](#pluginssettings)
11. [Profiles](#profiles)
12. [Environment Variable Substitution](#environment-variable-substitution)
13. [Scenario Config Index](#scenario-config-index)

---

## How Configuration Works

Configuration is applied in layers, each overriding the previous:

1. **Compiled-in defaults** -- field initializers in `config_schema.h`; what you get with an empty config file
2. **Profile preset** -- if `profile = "..."` is set at the top of the file or via `--profile`, a named batch of defaults is applied before the file is parsed
3. **TOML file fields** -- the values in your `.toml` file
4. **CLI flags** -- `--host`, `--port`, `--backend`, `--model` override the corresponding config values at launch time

Any field absent from the TOML file retains the value from the layer below it.
This means you can create a minimal config with just `[model]` and `[inference]`
and get a fully functional server -- every other section picks up its defaults.

Profiles are useful when you maintain configs for multiple environments (laptop,
workstation, production server) and want to express only the deltas rather than
repeating common settings. See the [Profiles](#profiles) section.

---

## CLI Flags

These override the corresponding config values at runtime:

| Flag             | Type    | Overrides                      |
|------------------|---------|--------------------------------|
| `--config PATH`  | string  | Path to the TOML config file   |
| `--profile NAME` | string  | `profile` key                  |
| `--backend NAME` | string  | `inference.backend`            |
| `--model PATH`   | string  | `model.path`                   |
| `--host ADDR`    | string  | `server.host`                  |
| `--port PORT`    | integer | `server.port`                  |

---

## [server]

Controls the HTTP server. This is where you decide who can reach the server
and how requests are accepted. For local development, the defaults (`127.0.0.1:8080`,
no API key) are fine. For any deployment that is reachable from a network, set
`api_key` to a strong random token and set `host = "0.0.0.0"` only if you intend
to accept connections from other machines.

| Key                   | Type     | Default      | Description |
|-----------------------|----------|--------------|-------------|
| `host`                | string   | `"127.0.0.1"`| Bind address. Use `"0.0.0.0"` to accept connections on all interfaces. |
| `port`                | integer  | `8080`       | TCP port to listen on. Range: 1-65535. |
| `api_key`             | string   | `""`         | Bearer token required in `Authorization: Bearer <key>`. Empty = no auth. Supports `${ENV:VAR}`. |
| `cors_origins`        | string[] | `["*"]`      | Allowed CORS origins. `["*"]` = accept all. Empty array = deny cross-origin requests. |
| `request_timeout_ms`  | integer  | `30000`      | Per-request hard timeout in milliseconds. Set higher for CPU inference with long prompts. |
| `max_concurrent`      | integer  | `4`          | Maximum number of parallel HTTP handler goroutines. Set to match expected concurrency. |
| `tls_enabled`         | bool     | `false`      | Enable HTTPS. Requires `tls_cert_file` and `tls_key_file`. |
| `tls_cert_file`       | string   | `""`         | Path to PEM-encoded TLS certificate. |
| `tls_key_file`        | string   | `""`         | Path to PEM-encoded TLS private key. |

Example:

```toml
[server]
host              = "0.0.0.0"
port              = 8080
api_key           = "${ENV:TRUELLM_API_KEY}"
cors_origins      = ["https://app.example.com"]
max_concurrent    = 16
request_timeout_ms = 60000
```

---

## [model]

Specifies which model to load at startup. TrueLLM loads GGUF files (the format
produced by llama.cpp's conversion tools). Set `path` to the absolute path of
the file; the server will refuse to start if the file does not exist rather than
starting in a degraded state.

The `chat_template` field matters: different model families use different special
tokens to separate user and assistant turns. If the template is wrong the model
will produce garbled output because the prompt is malformed. When in doubt, check
the model card on HuggingFace to see which template the model expects.

| Key                | Type   | Default  | Description |
|--------------------|--------|----------|-------------|
| `path`             | string | `""`     | Absolute or relative path to a GGUF model file. If empty, the engine starts without a model (all inference endpoints return 503). |
| `hf_repo`          | string | `""`     | HuggingFace Hub repository ID (e.g. `"bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"`). Reserved; model downloading is not yet implemented. |
| `hf_file`          | string | `""`     | Specific filename within the HF repo (e.g. `"Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"`). |
| `alias`            | string | `""`     | Model name shown in `/v1/models` and response `model` fields. Defaults to the filename stem if empty. |
| `chat_template`    | string | `"auto"` | Chat template to apply to message arrays. Valid values: `"auto"`, `"chatml"`, `"llama3"`, `"gemma"`, `"phi3"`. `"auto"` falls back to `"chatml"`. |
| `trust_remote_code`| bool   | `false`  | Allow custom tokenizer code from HuggingFace. Reserved; not enforced in the current version. |

Chat template syntax reference:

| Value      | Format                                           | Use for                          |
|------------|--------------------------------------------------|----------------------------------|
| `chatml`   | `<\|im_start\|>role\ncontent<\|im_end\|>\n`     | Mistral, Qwen, many instruction models |
| `llama3`   | `<\|start_header_id\|>role<\|end_header_id\|>\n\ncontent<\|eot_id\|>` | Meta Llama 3.x |
| `gemma`    | `<start_of_turn>role\ncontent<end_of_turn>\n`    | Google Gemma 2.x |
| `phi3`     | `<\|role\|>\ncontent<\|end\|>\n`                 | Microsoft Phi-3 |

---

## [inference]

Core inference engine settings.

| Key               | Type    | Default  | Description |
|-------------------|---------|----------|-------------|
| `backend`         | string  | `"cpu"`  | Inference backend. Valid: `"cpu"` (llama.cpp CPU), `"ggml"` (alias for `"cpu"`), `"cuda"` (NVIDIA GPU). |
| `context_size`    | integer | `4096`   | Maximum context window in tokens. Set to 0 to use the model's built-in default. Larger values use proportionally more KV cache memory. |
| `batch_size`      | integer | `512`    | Prompt batch size for tokenized input processing (prompt evaluation). |
| `ubatch_size`     | integer | `512`    | Micro-batch size for continuous batching decode steps. Typically equal to `batch_size`. |
| `threads`         | integer | `0`      | CPU threads for inference. 0 = auto-detect logical core count. |
| `flash_attention` | bool    | `true`   | Enable flash attention in the CPU backend when available. Reduces memory bandwidth and can improve throughput. |
| `mmap`            | bool    | `true`   | Memory-map the GGUF file. Enables fast startup and allows the OS to page out unused model weights. Disable if the model is on a network filesystem. |
| `mlock`           | bool    | `false`  | Lock model pages into RAM to prevent swapping. Requires `CAP_IPC_LOCK` on Linux or Administrator on Windows. |
| `numa`            | string  | `"disabled"` | NUMA memory policy. Valid: `"disabled"`, `"distribute"` (spread across NUMA nodes), `"isolate"` (restrict to a single node). |

---

## [inference.kv_cache]

Controls the Key-Value attention cache precision.

| Key      | Type    | Default | Description |
|----------|---------|---------|-------------|
| `type_k` | string  | `"f16"` | Precision of the key cache. Valid: `"f16"` (full precision), `"q8_0"` (8-bit quantized), `"q4_0"` (4-bit quantized). |
| `type_v` | string  | `"f16"` | Precision of the value cache. Same valid values as `type_k`. Key and value can differ. |
| `size_mb`| integer | `0`     | Hard cap on KV cache size in MiB. 0 = let the backend compute an appropriate size from `context_size`. |

KV quantization trade-offs:

| Type   | VRAM per token (8B, 32 layers) | Quality impact      |
|--------|-------------------------------|---------------------|
| `f16`  | ~4 bytes                      | None (reference)    |
| `q8_0` | ~1 byte                       | Negligible (<0.2%)  |
| `q4_0` | ~0.5 bytes                    | Noticeable at long ctx |

---

## [inference.cuda]

Settings for the llama.cpp GPU offload path. Active only when `backend = "cuda"` and the
build does not have the paged-attention CudaEngine enabled.

| Key                        | Type     | Default  | Description |
|----------------------------|----------|----------|-------------|
| `gpu_layers`               | integer  | `-1`     | Number of transformer layers to offload to GPU. -1 = fill all available VRAM. 0 = CPU-only. |
| `main_gpu`                 | integer  | `0`      | Primary CUDA device index (matches `nvidia-smi` output). |
| `tensor_split`             | float[]  | `[]`     | VRAM split fractions for multi-GPU setups, e.g. `[0.6, 0.4]`. One entry per GPU. Empty = equal split. |
| `gpu_memory_utilization`   | float    | `0.90`   | Maximum fraction of GPU VRAM to use. Range: 0.0-1.0. |
| `use_flash_attention_cuda` | bool     | `true`   | Enable CUDA flash attention in the llama.cpp backend. |
| `compute_capability_major` | integer  | `0`      | Override detected CUDA compute capability major version. 0 = auto-detect. |
| `compute_capability_minor` | integer  | `0`      | Override detected CUDA compute capability minor version. |

---

## [inference.cuda_engine]

Settings for the vLLM-style paged-attention CudaEngine. Active only when
`backend = "cuda"` and the engine is built with the CudaEngine (Phase 6).

| Key                        | Type    | Default    | Description |
|----------------------------|---------|------------|-------------|
| `paged_kv_block_size`      | integer | `16`       | Number of tokens per KV page block. Must be a power of 2. Smaller = finer-grained eviction; larger = less metadata overhead. |
| `max_num_blocks`           | integer | `0`        | Total KV blocks to allocate. 0 = automatically computed from `gpu_memory_utilization` and free VRAM at startup. |
| `max_num_seqs`             | integer | `256`      | Maximum number of concurrently live sequences (across decode steps). |
| `num_cuda_streams`         | integer | `3`        | CUDA stream pool size. Minimum 3: one each for prefill, decode, and host-to-device copy. |
| `use_fp8_kv`               | bool    | `false`    | Store KV cache in FP8 format. Requires compute capability sm_89+ (Ada, Hopper). Halves KV memory vs F16 at marginal quality cost. |
| `gpu_memory_utilization`   | float   | `0.90`     | Fraction of free VRAM to reserve for the KV pool. 0.0-1.0. |
| `tensor_parallel_size`     | integer | `1`        | Number of GPUs for tensor parallelism. Values >1 require an NCCL-enabled build and multiple GPUs. |
| `enforce_eager`            | bool    | `false`    | Disable CUDA graph capture. Graphs speed up decode at the cost of setup time. Enable to reduce startup VRAM. |
| `cuda_graph_max_batch_size`| integer | `8`        | Maximum batch size to capture CUDA graphs for. |
| `attention_layout`         | string  | `"paged"`  | Internal attention kernel layout. Valid: `"paged"`, `"flash_paged"`. |

---

## [inference.batching]

Continuous (vLLM-style) batching for the CPU backend.

| Key          | Type    | Default | Description |
|--------------|---------|---------|-------------|
| `enabled`    | bool    | `false` | Enable continuous batching. When disabled, requests are processed one at a time. |
| `max_seqs`   | integer | `8`     | Maximum sequences decoded together in one decode step. Increasing this improves throughput at the cost of per-request latency. |
| `max_queue`  | integer | `256`   | Maximum number of requests waiting in the queue. Requests beyond this limit receive HTTP 503. |
| `kv_unified` | bool    | `true`  | When true, each sequence may use up to the full `context_size` tokens. When false, context is partitioned equally among `max_seqs` sequences (uses less memory but limits individual sequence length). |

---

## [sampling]

Default generation parameters. All values can be overridden per request via the
OpenAI API `temperature`, `top_p`, `max_tokens`, etc. fields.

| Key                 | Type     | Default | Description |
|---------------------|----------|---------|-------------|
| `temperature`       | float    | `0.7`   | Sampling temperature. 0.0 = deterministic greedy decoding. >1.0 = more random output. |
| `top_p`             | float    | `0.95`  | Nucleus sampling: keep only the top tokens whose cumulative probability exceeds `top_p`. 0.0 = disabled (use all tokens). |
| `top_k`             | integer  | `40`    | Top-K sampling: keep only the K highest probability tokens. 0 = disabled. |
| `min_p`             | float    | `0.05`  | Min-P filtering: remove tokens whose probability is less than `min_p * max_token_prob`. 0.0 = disabled. |
| `repeat_penalty`    | float    | `1.1`   | Penalize tokens that have already appeared in the context. 1.0 = no penalty. Values above 1.0 reduce repetition. |
| `frequency_penalty` | float    | `0.0`   | OpenAI-style frequency penalty applied per token occurrence count. 0.0 = disabled. |
| `presence_penalty`  | float    | `0.0`   | OpenAI-style presence penalty applied if a token has appeared at all. 0.0 = disabled. |
| `max_tokens`        | integer  | `2048`  | Default maximum generated tokens. Requests without an explicit `max_tokens` field use this value. |
| `seed`              | integer  | `-1`    | RNG seed for reproducible output. -1 = random seed per request. |
| `stop`              | string[] | `[]`    | Default stop strings. Generation halts if any of these appear in the output. Per-request `stop` is merged with this list. |

---

## [logging]

| Key               | Type   | Default  | Description |
|-------------------|--------|----------|-------------|
| `level`           | string | `"info"` | Minimum log level. Valid: `"trace"`, `"debug"`, `"info"`, `"warn"`, `"error"`. |
| `format`          | string | `"text"` | Log output format. `"text"` = human-readable with color. `"json"` = newline-delimited JSON for log aggregation (Splunk, Loki, CloudWatch). |
| `file`            | string | `""`     | Additional log file path. Logs are always written to stdout; this duplicates them to a file. Empty = no file. |
| `timestamps`      | bool   | `true`   | Include ISO-8601 timestamps in log lines. |
| `source_location` | bool   | `false`  | Include file:line in every log entry. Verbose; use only for debugging. |

---

## [hardware]

Host resource configuration.

| Key                            | Type     | Default | Description |
|--------------------------------|----------|---------|-------------|
| `cpu_threads`                  | integer  | `0`     | Threads available to the engine. 0 = auto-detect. |
| `cpu_affinity`                 | integer[]| `[]`    | Pin the engine process to specific CPU cores, e.g. `[0, 1, 2, 3]`. Empty = no pinning. |
| `max_ram_gb`                   | float    | `0.0`   | Hard cap on system RAM usage in GiB. 0.0 = no limit. |
| `resource_monitor_interval_ms` | integer  | `1000`  | Polling interval for the CPU/RAM/VRAM metrics subsystem in milliseconds. |

---

## [hardware.gpu]

| Key               | Type    | Default    | Description |
|-------------------|---------|------------|-------------|
| `enabled`         | bool    | `true`     | Whether GPU acceleration is enabled at all. Set to false for CPU-only operation. |
| `device_id`       | integer | `0`        | CUDA or Vulkan device index. Matches the index shown by `nvidia-smi`. |
| `name_hint`       | string  | `""`       | Optional display name for the device. Used for logging only. |
| `vram_capacity_gb`| float   | `0.0`      | Override the auto-detected VRAM capacity in GiB. 0.0 = auto-detect. |
| `backend_api`     | string  | `"cuda"`   | GPU compute API. Valid: `"cuda"` (NVIDIA CUDA 12.x), `"vulkan"` (cross-vendor Vulkan compute). |

---

## [hardware.offload]

Controls how model weights are distributed between GPU VRAM and system RAM.

| Key                   | Type    | Default | Description |
|-----------------------|---------|---------|-------------|
| `use_gpu_vram`        | bool    | `true`  | Offload weight tensors to dedicated GPU VRAM. |
| `use_system_ram`      | bool    | `true`  | Spill weights that do not fit in VRAM into system RAM. |
| `use_shared_memory`   | bool    | `false` | Allow use of shared/UMA GPU memory (integrated GPU overflow). |
| `gpu_layers`          | integer | `-1`    | Number of model layers to keep on GPU. -1 = fill all available VRAM first. 0 = no GPU offload. Takes precedence over `inference.cuda.gpu_layers`. |
| `vram_budget_fraction`| float   | `1.0`   | Fraction of detected VRAM to allocate for model weights. Range: 0.0-1.0. Use <1.0 to reserve VRAM for other applications. |

---

## [hardware.guardrails]

Guardrails protect against two distinct failure modes: loading a model so large
it exhausts RAM/VRAM and OOMs the process at load time, and accepting requests
with token counts so large they exhaust the KV cache during inference.

The `preset` field is the primary control. It sets both a memory load limit and
a per-request token limit simultaneously, so you do not need to tune individual
numbers unless your workload is unusual. `balanced` is appropriate for most
deployments. `strict` is for production servers where you want to refuse oversized
models rather than risk instability. `off` disables all checks, which is useful
for testing or when you are certain the model fits.

Rate limiting is also driven by this preset -- see [src/policy/](../src/policy/)
for how the guardrail preset maps to per-IP request rates.

| Key                  | Type    | Default       | Description |
|----------------------|---------|---------------|-------------|
| `preset`             | string  | `"balanced"`  | Enforcement level. Valid values: |
|                      |         |               | `"off"` - No protections. Load whatever is asked. |
|                      |         |               | `"relaxed"` - Warn if model exceeds 50% of available RAM+VRAM. |
|                      |         |               | `"balanced"` - Warn and cap at 75% of available RAM+VRAM (default). |
|                      |         |               | `"strict"` - Refuse load if model would exceed 90% of available RAM+VRAM. |
|                      |         |               | `"custom"` - Use `max_model_size_gb` and the `*_fraction_limit` fields below. |
| `max_model_size_gb`  | float   | `0.0`         | Used only when `preset = "custom"`. Maximum model size in GiB. 0.0 = unlimited. |
| `allow_overcommit`   | bool    | `false`       | Allow loading a model larger than the guardrail threshold (with a warning). |
| `ram_fraction_limit` | float   | `0.0`         | Override the RAM fraction cap. 0.0 = use the preset default. |
| `vram_fraction_limit`| float   | `0.0`         | Override the VRAM fraction cap. 0.0 = use the preset default. |

---

## [runtime]

Selects the inference engine runtime pack loaded at startup.

| Key              | Type   | Default             | Description |
|------------------|--------|---------------------|-------------|
| `engine`         | string | `"cpu-llama-cpp"`   | Runtime pack. Valid: `"cpu-llama-cpp"`, `"cuda12-llama-cpp"`, `"cuda-llama-cpp"`, `"vulkan-llama-cpp"`. |
| `engine_version` | string | `"latest"`          | Pin to a specific engine version string, e.g. `"v2.8.0"`. `"latest"` = use the most recent available. |
| `auto_update`    | bool   | `false`             | Automatically upgrade the runtime pack on startup. |
| `update_channel` | string | `"stable"`          | Update channel. Valid: `"stable"`, `"nightly"`. |

Runtime pack reference:

| Value               | Requires                      | Use when                              |
|---------------------|-------------------------------|---------------------------------------|
| `cpu-llama-cpp`     | Nothing                       | CPU-only inference (always available) |
| `cuda12-llama-cpp`  | CUDA Toolkit 12.x runtime     | NVIDIA GPU, CUDA 12                   |
| `cuda-llama-cpp`    | CUDA 11 or 12 runtime         | NVIDIA GPU, auto CUDA version         |
| `vulkan-llama-cpp`  | Vulkan 1.3 runtime + drivers  | AMD/Intel/NVIDIA GPU via Vulkan       |

---

## [runtime.harmony]

Harmony is the chat history renderer and OpenAI conversation parser.

| Key       | Type   | Default    | Description |
|-----------|--------|------------|-------------|
| `enabled` | bool   | `false`    | Enable the Harmony subsystem. |
| `version` | string | `"latest"` | Pinned Harmony version, e.g. `"v0.3.6"`. |

---

## [[runtime.extensions]]

Array of additional runtime extension packs. Each entry is a TOML array table.

```toml
[[runtime.extensions]]
name    = "cuda12-llama-cpp"
enabled = true
version = "v2.8.0"

[[runtime.extensions]]
name    = "vulkan-llama-cpp"
enabled = false
version = "latest"
```

| Key       | Type   | Default    | Description |
|-----------|--------|------------|-------------|
| `name`    | string | `""`       | Extension pack name. Same valid values as `runtime.engine`. |
| `enabled` | bool   | `false`    | Whether to load this extension. |
| `version` | string | `"latest"` | Pinned version or `"latest"`. |

---

## [plugins]

Plugin system configuration. Plugins are how you extend TrueLLM with capabilities
that go beyond text generation: fetching live web pages, querying databases,
calling internal APIs, enforcing content policies. The plugin system is designed so
that adding a new tool requires zero changes to the engine -- you write a plugin,
point `plugin_dirs` at its location, and restart the server.

The `enabled` field is a master switch. When it is `false` (the default), no
plugin directories are scanned and no sidecars are spawned, even if `plugin_dirs`
is populated. This makes it easy to disable the plugin layer entirely for a
deployment that does not need it without changing the rest of the config.

See [plugin-development.md](plugin-development.md) for how to write your own
native, Python, or JavaScript plugin.

| Key                   | Type     | Default          | Description |
|-----------------------|----------|------------------|-------------|
| `enabled`             | bool     | `false`          | Master switch. No plugins are loaded when false regardless of `plugin_dirs`. |
| `python_path`         | string   | `"python3"`      | Python interpreter used to launch Python sidecar plugins. Must be Python 3.9+. |
| `plugin_dirs`         | string[] | `[]`             | Directories to scan for plugin packages. Each directory is examined at startup. A directory containing `manifest.toml + main.py` is treated as a Python sidecar. A directory containing `manifest.toml + main.ts` or `main.js` is treated as a JavaScript sidecar. Any other directory is scanned for `.dll` / `.so` native plugins. |
| `socket_path`         | string   | `""`             | Override the IPC socket path. Empty = platform default: `/tmp/truellm-<pid>-<runtime>.sock` on POSIX, `\\.\pipe\truellm-<pid>-<runtime>` on Windows. |
| `sidecar_timeout_ms`  | integer  | `5000`           | Time in milliseconds to wait for a sidecar process to connect and send its announce message after being spawned. |
| `restart_on_crash`    | bool     | `true`           | Automatically restart a sidecar that exits with a non-zero code. |
| `max_restart_attempts`| integer  | `3`              | Maximum consecutive restarts before giving up. Uses exponential backoff: 1s, 2s, 4s, 8s, ..., max 30s between attempts. |

### Plugin Directory Detection

The engine classifies each entry in `plugin_dirs` as follows:

```
plugin_dir/
  manifest.toml + main.py     ->  Python sidecar (truellm_engine launched)
  manifest.toml + main.ts/js  ->  JavaScript sidecar (Bun runtime launched)
  *.dll / *.so                ->  Native C++ plugin (dlopen/LoadLibrary)
```

---

## [plugins.settings]

A flat key-value table of runtime settings passed to plugins via the host API.
Plugins call `host_api.get_config(plugin_id, key)` to read these values.

Lookup order:
1. `"plugin_id.key"` (per-plugin override)
2. `"key"` (global fallback)

```toml
[plugins.settings]
# Per-plugin setting (only visible to tool_web_fetch)
tool_web_fetch.proxy_url = "http://proxy.corp.example.com:3128"
tool_web_fetch.max_bytes = "131072"

# Global setting (visible to all plugins via bare key lookup)
log_level = "debug"
```

Values must be strings, integers, booleans, or floats. They are always returned
to plugins as null-terminated C strings (`const char*`).

---

## Profiles

Profiles are named presets that set a batch of default values before the TOML
file is applied. Set with `profile = "..."` at the top of the file or `--profile`
on the CLI.

| Profile       | Description |
|---------------|-------------|
| `local-cpu`   | Single-user laptop. Conservative defaults: 2 threads, 2 concurrent requests, relaxed guardrails, debug logging. |
| `workstation` | Power-user desktop. Larger context (8192), auto-thread count, 8 concurrent requests, balanced guardrails. |
| `server`      | Production daemon. JSON logging, strict guardrails, 16 concurrent requests, no CORS, info log level. |

---

## Environment Variable Substitution

Any string field can reference environment variables using the syntax:

```toml
api_key = "${ENV:TRUELLM_API_KEY}"
model.path = "${ENV:MODEL_DIR}/llama-3.1-8b.gguf"
```

The substitution happens at config load time. If the environment variable is not
set, the field becomes an empty string.

---

## Scenario Config Index

Ready-to-use scenario configs in `configs/`:

| File                                | Use case |
|-------------------------------------|----------|
| `profiles/local-cpu.toml`           | Laptop or single-user desktop, CPU-only, mmap on, conservative context |
| `profiles/workstation.toml`         | Power-user desktop, larger context, auto thread count, GPU offload |
| `profiles/server.toml`              | Production daemon, JSON logs, strict guardrails, API key required |
| `scenario-web-fetch.toml`           | End-to-end test for the tool_web_fetch native plugin; pairs with `tests/integration/python/test_web_fetch_plugin.py` |

---

## Quick Reference: Minimal Working Config

```toml
[model]
path = "/path/to/your/model.gguf"

[inference]
backend = "cpu"
```

All other fields use their defaults. The server listens on `127.0.0.1:8080` with
no authentication and CPU inference.
