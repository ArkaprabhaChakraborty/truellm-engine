# Build Instructions

## CMake Options

These options are set with `-DOPTION=VALUE` at configure time.

| Option | Default | Description |
|---|---|---|
| `TRUELLM_BACKEND` | `CPU` | Inference backend: `CPU` or `CUDA` |
| `TRUELLM_BUILD_CUDA_ENGINE` | `OFF` | Build the vLLM-style paged-attention CudaEngine (requires CUDA) |
| `TRUELLM_BUILD_TESTS` | `ON` | Build GoogleTest unit tests |
| `TRUELLM_BUILD_BENCHMARKS` | `OFF` | Build Google Benchmark throughput suite |
| `TRUELLM_SANITIZERS` | `""` | Semicolon-separated sanitizer list: `address`, `undefined`, `thread` (GCC/Clang only) |
| `TRUELLM_ENABLE_CLANG_TIDY` | `OFF` | Run clang-tidy on every compiled translation unit |
| `TRUELLM_ENABLE_CPPCHECK` | `OFF` | Run cppcheck on every compiled translation unit |
| `TRUELLM_CODESIGN` | `OFF` | Sign built binaries with signtool.exe (Windows only) |
| `TRUELLM_CODESIGN_CERT_SHA1` | `""` | SHA1 thumbprint of the code-signing certificate |
| `TRUELLM_CODESIGN_CERT_NAME` | `""` | Certificate CN subject (used when CERT_SHA1 is not set) |
| `CMAKE_BUILD_TYPE` | -- | `Release`, `Debug`, `RelWithDebInfo` |
| `CMAKE_CUDA_ARCHITECTURES` | -- | CUDA arch list, e.g. `75;80;86;89;90;120` |

---

## Build Presets

`CMakePresets.json` provides ready-made configure + build presets.

```bash
# List available presets
cmake --list-presets

# Windows CUDA Release (the primary development preset)
cmake --preset windows-cuda
cmake --build --preset windows-cuda --parallel

# Windows CPU Debug
cmake --preset windows-debug
cmake --build --preset windows-debug --parallel

# Linux CUDA Release
cmake --preset linux-cuda
cmake --build --preset linux-cuda --parallel

# Linux CPU Release
cmake --preset linux-release
cmake --build --preset linux-release --parallel

# Linux CPU Debug (with AddressSanitizer + UBSan)
cmake --preset linux-debug
cmake --build --preset linux-debug --parallel
ctest --test-dir build/linux-debug --output-on-failure
```

---

## build.ps1 (Windows shortcut)

The `build.ps1` script in the repository root wraps the full build pipeline.

```powershell
# Standard CUDA Release build
.\build.ps1

# With code signing (requires a cert -- see Code Signing section below)
.\build.ps1 -Sign

# Build then run unit tests immediately
.\build.ps1 -Test

# Build then launch the server with the local-cpu config
.\build.ps1 -Run

# CPU Debug, no tests
.\build.ps1 -Backend CPU -Config Debug -SkipTests

# Full build: sign, benchmarks, clang-tidy, then test
.\build.ps1 -Sign -Benchmarks -ClangTidy -Test
```

---

## vcpkg Dependencies

All C++ dependencies are managed by vcpkg (`vcpkg.json`). vcpkg is invoked
automatically at configure time via the toolchain file; you do not need to
run `vcpkg install` manually.

| Package | Purpose |
|---|---|
| `fmt` | String formatting |
| `spdlog` | Structured logging |
| `nlohmann-json` | JSON serialization |
| `toml11` | TOML config parsing |
| `cli11` | CLI argument parsing |
| `cpp-httplib` | HTTP server and client |
| `gtest` | GoogleTest unit testing framework |
| `benchmark` | Google Benchmark throughput harness |

---

## Windows (MSVC + Ninja)

### Prerequisites

- Visual Studio 2022 Build Tools (MSVC 14.4x)
- CMake 3.21+
- Ninja (included with VS Build Tools or installable via winget)
- vcpkg with `VCPKG_ROOT` environment variable set
- CUDA Toolkit 12.x (for CUDA builds)

### Full CUDA Release build

```powershell
# From a Developer PowerShell for VS 2022 (or any shell with cl.exe on PATH)
cmake -B build\windows-cuda -G Ninja `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" `
  -DCMAKE_BUILD_TYPE=Release `
  -DTRUELLM_BACKEND=CUDA `
  -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90;120" `
  -DTRUELLM_BUILD_CUDA_ENGINE=ON `
  -DTRUELLM_BUILD_TESTS=ON `
  -DTRUELLM_BUILD_BENCHMARKS=ON

cmake --build build\windows-cuda --parallel
```

Output:
- `build\windows-cuda\bin\truellm-server.exe`
- `build\windows-cuda\bin\truellm_unit_tests.exe`
- `build\windows-cuda\plugins\native\tool_web_fetch.dll`

### CPU-only Debug build

```powershell
cmake -B build\windows-debug -G Ninja `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" `
  -DCMAKE_BUILD_TYPE=Debug `
  -DTRUELLM_BACKEND=CPU `
  -DTRUELLM_BUILD_TESTS=ON

cmake --build build\windows-debug --parallel
```

---

## Linux (GCC / Clang)

```bash
cmake -B build/linux-cuda -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DTRUELLM_BACKEND=CUDA \
  -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90" \
  -DTRUELLM_BUILD_CUDA_ENGINE=ON \
  -DTRUELLM_BUILD_TESTS=ON
cmake --build build/linux-cuda --parallel
```

CPU + ASan + UBSan (for development):
```bash
cmake -B build/linux-debug -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake \
  -DCMAKE_BUILD_TYPE=Debug \
  -DTRUELLM_BACKEND=CPU \
  -DTRUELLM_SANITIZERS="address;undefined" \
  -DTRUELLM_BUILD_TESTS=ON
cmake --build build/linux-debug --parallel
```

---

## macOS (Apple Silicon)

CPU build only (CUDA not available on macOS):

```bash
cmake -B build/macos -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DTRUELLM_BACKEND=CPU
cmake --build build/macos --parallel
```

---

## CUDA Setup

Requires CUDA Toolkit 12.x from NVIDIA.

1. Download and install CUDA Toolkit 12.x
2. Verify: `nvcc --version`
3. Configure with `-DTRUELLM_BACKEND=CUDA -DTRUELLM_BUILD_CUDA_ENGINE=ON`

For FP8 KV cache (halves VRAM usage for the KV cache):
- Requires sm_89+ (Ada Lovelace RTX 4xxx, Hopper H100, Blackwell)
- Set `use_fp8_kv = true` in `[inference.cuda_engine]` in your config

`CMAKE_CUDA_ARCHITECTURES` controls which GPU generations are compiled for.
Compiling for only your local GPU (e.g. `89` for RTX 4090) cuts build time
significantly compared to the full list `75;80;86;89;90;120`.

---

## llama.cpp Submodule

The llama.cpp library is included as a git submodule in `third_party/llama.cpp`.

```bash
# Initialize after clone
git submodule update --init --recursive

# Update to a newer llama.cpp version
git submodule update --remote third_party/llama.cpp
```

---

## Code Signing (Windows)

Windows Smart App Control flags unsigned binaries from unknown publishers.
TrueLLM provides a signing workflow via CMake and a PowerShell helper.

### One-time setup

```powershell
# Run once as Administrator -- creates a dev cert and installs it in Trusted Root
.\scripts\windows\create-dev-cert.ps1
# Copy the thumbprint printed at the end
```

### Configure with signing

```powershell
cmake -B build\windows-cuda ... `
  -DTRUELLM_CODESIGN=ON `
  -DTRUELLM_CODESIGN_CERT_SHA1=<thumbprint>
```

Every subsequent build runs `signtool.exe` automatically as a post-build
step on `truellm-server.exe` and `tool_web_fetch.dll`.

The signing module (`cmake/CodeSigning.cmake`) locates `signtool.exe` by
globbing across all installed Windows SDK versions, so it survives SDK
updates without manual path maintenance.

---

## Static Analysis

Both clang-tidy and cppcheck are integrated into the build system. The
tools are optional -- the build proceeds normally if they are not installed.

```powershell
# Enable clang-tidy (runs on every compiled TU during build)
cmake --preset windows-cuda -DTRUELLM_ENABLE_CLANG_TIDY=ON
cmake --build --preset windows-cuda --parallel

# Enable cppcheck
cmake --preset windows-cuda -DTRUELLM_ENABLE_CPPCHECK=ON
cmake --build --preset windows-cuda --parallel
```

The check set and suppressions are configured in
`cmake/StaticAnalysis.cmake`. Clang-tidy warnings do not fail the build
by default; to make them errors, edit the `--warnings-as-errors=` flag in
that file.

---

## Building and Running Tests

```bash
# Build with tests (ON by default)
cmake --preset windows-cuda -DTRUELLM_BUILD_TESTS=ON
cmake --build --preset windows-cuda --parallel

# Run all unit tests via CTest
ctest --test-dir build/windows-cuda --output-on-failure --parallel 4

# Filter to a specific suite
ctest --test-dir build/windows-cuda -R "RateLimiter" --output-on-failure

# Run the test binary directly (faster output, GTest format)
.\build\windows-cuda\bin\truellm_unit_tests.exe
.\build\windows-cuda\bin\truellm_unit_tests.exe --gtest_filter="ToolRegistry*"
.\build\windows-cuda\bin\truellm_unit_tests.exe --gtest_list_tests
```

Inference tests that require a real model:
```powershell
$env:TRUELLM_TEST_MODEL_PATH = "C:\models\your-model.gguf"
.\build\windows-cuda\bin\truellm_unit_tests.exe --gtest_filter="GgmlEngineIntegration*"
```

Integration tests against a running server:
```powershell
# Terminal 1 -- start server
.\build\windows-cuda\bin\truellm-server.exe --config configs\scenario-web-fetch.toml

# Terminal 2 -- run integration test
python tests\integration\python\test_web_fetch_plugin.py
python tests\integration\python\test_batch_concurrent.py
```

---

## Building Benchmarks

```bash
cmake --preset windows-cuda -DTRUELLM_BUILD_BENCHMARKS=ON
cmake --build --preset windows-cuda --parallel

# Run with a model
$env:TRUELLM_TEST_MODEL_PATH = "C:\models\your-model.gguf"
.\build\windows-cuda\bin\truellm_benchmarks.exe
.\build\windows-cuda\bin\truellm_benchmarks.exe --benchmark_filter=BM_Tokenize
```

---

## Native Plugin (tool_web_fetch)

The `tool_web_fetch` plugin is built automatically as part of the main
build. Its output lands in an isolated directory separate from the server
binary so it can be signed independently and pointed to by `plugin_dirs`.

```
build\windows-cuda\plugins\native\tool_web_fetch.dll   (Windows)
build/linux-cuda/plugins/native/tool_web_fetch.so      (Linux)
```

Reference it in a config:
```toml
[plugins]
enabled     = true
plugin_dirs = ["build/windows-cuda/plugins/native"]
```

For HTTPS support in the plugin, build cpp-httplib with OpenSSL:
```bash
# Add to vcpkg.json: "cpp-httplib[ssl]"
cmake ... -DCPPHTTPLIB_OPENSSL_SUPPORT=ON
```

---

## Output Directory Summary

| Artifact | Location |
|---|---|
| Server binary | `build/<preset>/bin/truellm-server[.exe]` |
| Unit tests | `build/<preset>/bin/truellm_unit_tests[.exe]` |
| Benchmarks | `build/<preset>/bin/truellm_benchmarks[.exe]` |
| Native plugins | `build/<preset>/plugins/native/*.dll` / `*.so` |
| Static libraries | `build/<preset>/lib/` |
