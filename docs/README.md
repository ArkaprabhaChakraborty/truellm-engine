# TrueLLM Documentation

## Documents

| File | Contents |
|---|---|
| [getting-started.md](getting-started.md) | First-run walkthrough — build, configure, run a model |
| [build-instructions.md](build-instructions.md) | Full CMake build options, CUDA setup, vcpkg |
| [architecture.md](architecture.md) | Internal architecture: engine, server, plugin system |
| [config-reference.md](config-reference.md) | Every config section and field explained |
| [api-reference.md](api-reference.md) | HTTP API endpoints (OpenAI-compatible + native) |
| [plugin-development.md](plugin-development.md) | Writing native, Python, and JavaScript plugins |

## Quick Start

```bash
# 1. Build (CPU-only)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# 2. Download a model
mkdir -p models
# place llama-3.2-3b-instruct-q4_k_m.gguf in models/

# 3. Run
./build/bin/truellm-server -c configs/scenario-development.toml

# 4. Test
curl http://127.0.0.1:9099/health
curl http://127.0.0.1:9099/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama-3.2-3b-dev","messages":[{"role":"user","content":"Hello"}]}'
```

## Config Files

| Path | Purpose |
|---|---|
| `configs/server.toml` | Full-featured template with all sections and comments |
| `configs/profiles/local-cpu.toml` | Copy-paste base for CPU-only development |
| `configs/profiles/workstation.toml` | Copy-paste base for GPU workstation |
| `configs/profiles/server.toml` | Copy-paste base for production server |
| `configs/scenario-*.toml` | Scenario configs for specific use cases |
| `configs/schemas/config.schema.json` | JSON Schema for IDE validation |
