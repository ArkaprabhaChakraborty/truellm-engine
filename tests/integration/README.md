# Integration Tests

Integration tests require a running TrueLLM server instance and cannot be
run as part of the CTest unit test suite.

## Python Test Scripts

Located in `tests/integration/python/`.

### Prerequisites

```
pip install -r tests/integration/python/requirements.txt
```

### Running

Start the server first, then run the tests against it.

#### Smoke test (Nemotron-3-Nano-4B)

```bash
# Start server
./build/bin/truellm-server -c configs/nemotron-cuda.toml -v

# In another terminal
python tests/integration/python/test_nemotron.py
```

Validates: `/health`, `/truellm/v1/status`, `/v1/models`,
`/v1/chat/completions` (non-streaming, multi-turn, streaming).

#### Concurrent batching stress test

```bash
# Requires a server with batching enabled
./build/bin/truellm-server -c configs/scenario-continuous-batching.toml

# Burst mode (16 parallel requests)
python tests/integration/python/test_batch_concurrent.py --mode burst --n 16

# Ramp mode (finds throughput saturation)
python tests/integration/python/test_batch_concurrent.py --mode ramp --n 16

# Sustained mode (60-second continuous load)
python tests/integration/python/test_batch_concurrent.py --mode sustained --n 8 --duration 60

# GPU KV-cache stress (long prompts)
python tests/integration/python/test_batch_concurrent.py --mode gpu-ctx --n 4 --tokens 512
```

## CTest Integration

Integration tests can optionally be registered with CTest by passing a server
URL via the `TRUELLM_TEST_SERVER` environment variable.  If the variable is
not set, the tests are skipped automatically.

```bash
TRUELLM_TEST_SERVER=http://127.0.0.1:9099 ctest -R integration --verbose
```

## Available Test Configurations

| Config file | Pairs with |
|---|---|
| `configs/test-batching-stress.toml` | `test_batch_concurrent.py` burst/ramp modes |
| `configs/test-gpu-stress.toml` | `test_batch_concurrent.py` sustained/gpu-ctx modes |
| `configs/test-cuda-engine.toml` | `test_batch_concurrent.py` baseline |
| `configs/nemotron-cuda.toml` | `test_nemotron.py` |
| `configs/scenario-continuous-batching.toml` | `test_batch_concurrent.py` all modes |
