# Getting Started

## Prerequisites

- CMake 3.21+
- C++17 compiler (MSVC 2022, GCC 12+, Clang 15+)
- vcpkg (for dependencies)
- CUDA 12.x toolkit (optional, for GPU inference)

## 1. Clone and Initialize

```bash
git clone https://github.com/ArkaprabhaChakraborty/truellm.git
cd truellm/truellm-engine
git submodule update --init --recursive
```

## 2. Install Dependencies via vcpkg

```bash
# Bootstrap vcpkg if not already installed
git clone https://github.com/microsoft/vcpkg.git
./vcpkg/bootstrap-vcpkg.sh

# Install dependencies listed in vcpkg.json
./vcpkg/vcpkg install
```

## 3. Build

### CPU-only build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_TOOLCHAIN_FILE=path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build --parallel
```

### CUDA build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_TOOLCHAIN_FILE=path/to/vcpkg/scripts/buildsystems/vcpkg.cmake \
      -DTRUELLM_BACKEND=CUDA \
      -DTRUELLM_BUILD_CUDA_ENGINE=ON
cmake --build build --parallel
```

The server binary is placed at `build/bin/truellm-server`.

## 4. Download a Model

TrueLLM loads GGUF format models from llama.cpp.  A good starting point:

- `llama-3.2-3b-instruct-q4_k_m.gguf` (~1.9 GB) — for CPU / development
- `llama-3.1-8b-instruct-q4_k_m.gguf` (~4.7 GB) — for GPU workstation

Place the file in `models/` relative to the working directory.

## 5. Choose a Config

Copy a profile to start from:

```bash
cp configs/profiles/local-cpu.toml my.toml
# Edit my.toml and set model.path
```

Or use an existing scenario config:

| Scenario | Config file | Requirements |
|---|---|---|
| CPU development | `configs/scenario-development.toml` | CPU only |
| GPU workstation | `configs/profiles/workstation.toml` | CUDA GPU |
| Production server | `configs/profiles/server.toml` | CUDA + API key |
| Plugin testing | `configs/scenario-plugins.toml` | CPU + Python/Node |

## 6. Run

```bash
./build/bin/truellm-server -c configs/profiles/local-cpu.toml
```

Flags:

| Flag | Effect |
|---|---|
| `-c PATH` | Config file path (default: `configs/server.toml`) |
| `--profile NAME` | Override profile preset |
| `--model PATH` | Override model path |
| `--host ADDR` | Override bind address |
| `--port PORT` | Override port |
| `--backend NAME` | Override inference backend |
| `-v` / `--verbose` | Enable debug logging |

## 7. Test the Server

```bash
# Health check
curl http://127.0.0.1:8080/health

# Server status (engine info, GPU stats, KV cache)
curl http://127.0.0.1:8080/truellm/v1/status

# List models
curl http://127.0.0.1:8080/v1/models

# Chat completion
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.2-3b-dev",
    "messages": [{"role": "user", "content": "What is 2 + 2?"}],
    "max_tokens": 64
  }'
```

## 8. Run Tests

```bash
# Build with tests
cmake -B build -DTRUELLM_BUILD_TESTS=ON ...
cmake --build build --parallel

# Run unit tests
ctest --test-dir build --output-on-failure
```

For integration tests against a running server, see
[tests/integration/README.md](../tests/integration/README.md).

## Next Steps

- [Config Reference](config-reference.md) — all configuration options
- [Architecture](architecture.md) — how the engine works internally
- [Plugin Development](plugin-development.md) — extending TrueLLM with plugins
- [API Reference](api-reference.md) — HTTP API documentation
