# API Reference

TrueLLM exposes two API groups:

- **OpenAI-compatible** (`/v1/*`) — drop-in replacement for the OpenAI API
- **Native** (`/truellm/v1/*`) — server-specific status and plugin management

## Authentication

If `server.api_key` is set, all requests must include:

```
Authorization: Bearer <api_key>
```

The health endpoint (`/health`) does not require authentication.

---

## OpenAI-Compatible Endpoints

### GET /v1/models

Returns the list of loaded models.

**Response**

```json
{
  "object": "list",
  "data": [
    {
      "id": "llama-3.1-8b-cuda",
      "object": "model",
      "created": 1720000000,
      "owned_by": "truellm"
    }
  ]
}
```

---

### POST /v1/chat/completions

OpenAI-compatible chat completions endpoint.

**Request body**

```json
{
  "model": "llama-3.1-8b-cuda",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "What is 2 + 2?"}
  ],
  "max_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.95,
  "top_k": 40,
  "stream": false,
  "stop": ["\n\n"]
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | string | required | Model alias (from `model.alias` config) |
| `messages` | array | required | Chat history. Roles: `system`, `user`, `assistant` |
| `max_tokens` | integer | config default | Maximum tokens to generate |
| `temperature` | float | 0.7 | Sampling temperature (0 = greedy) |
| `top_p` | float | 0.95 | Nucleus sampling cutoff |
| `top_k` | integer | 40 | Top-K sampling (0 = disabled) |
| `stream` | boolean | false | Enable SSE streaming |
| `stop` | array | [] | Stop sequences |
| `seed` | integer | -1 | Random seed (-1 = random) |
| `presence_penalty` | float | 0.0 | Penalize repeated topics |
| `frequency_penalty` | float | 0.0 | Penalize repeated tokens |

**Response (non-streaming)**

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1720000000,
  "model": "llama-3.1-8b-cuda",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "2 + 2 = 4."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 24,
    "completion_tokens": 8,
    "total_tokens": 32
  }
}
```

**Response (streaming, `stream: true`)**

Server-Sent Events, one JSON delta per line:

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"delta":{"content":"2"},"index":0}]}
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"delta":{"content":" + "},"index":0}]}
...
data: [DONE]
```

**Error responses**

| Status | Type | Cause |
|---|---|---|
| 400 | `invalid_request_error` | Missing model, empty messages, invalid parameters |
| 401 | `authentication_error` | Missing or invalid API key |
| 429 | `rate_limit_error` | Rate limit exceeded (from guardrail preset) |
| 503 | `server_error` | No model loaded, or inference queue full |

---

### POST /v1/completions

Text completions (non-chat).  Accepts a plain `prompt` string rather than a
`messages` array.

**Request body**

```json
{
  "model": "llama-3.1-8b-cuda",
  "prompt": "The capital of France is",
  "max_tokens": 64,
  "temperature": 0.0
}
```

Response format mirrors `/v1/chat/completions` with `text` instead of
`message.content` in the choice object.

---

## Native Endpoints

### GET /health

Liveness probe.  Always returns 200.  Does not require authentication.

```json
{"status": "ok"}
```

---

### GET /truellm/v1/status

Full server status: engine info, loaded model, GPU stats, KV cache, plugins.

**Response**

```json
{
  "status": "ready",
  "model": "llama-3.1-8b-cuda",
  "backend": "cuda",
  "context_max": 8192,
  "gpu": {
    "device_id": 0,
    "device_name": "NVIDIA GeForce RTX 5090",
    "vram_total_mb": 32768,
    "vram_free_mb": 18432,
    "layers_on_gpu": 32
  },
  "kv_cache": {
    "block_size_tokens": 16,
    "blocks_total": 2048,
    "blocks_used": 156,
    "blocks_free": 1892,
    "utilization_pct": 7.6,
    "fp8_enabled": false
  },
  "plugins": [
    {
      "id": "tool_web_fetch",
      "type": "native",
      "has_tool_provider": true,
      "tools": ["tool_web_fetch.fetch_url"],
      "status": "healthy"
    }
  ]
}
```

When no model is loaded, `status` is `"no_model"` and the GPU/KV sections
are absent.

---

### GET /truellm/v1/plugins

Lists all registered plugins and their tools.

**Response**

```json
{
  "plugins": [
    {
      "id": "tool_web_fetch",
      "type": "native",
      "tools": [
        {
          "name": "tool_web_fetch.fetch_url",
          "description": "Fetch the content of a URL via HTTP GET"
        }
      ]
    }
  ]
}
```

---

## Rate Limiting

Rate limiting is enforced per client IP by the policy layer.  Limits are set
by the `hardware.guardrails.preset`:

| Preset | Requests/sec | Burst |
|---|---|---|
| strict | 5 | 10 |
| balanced | 20 | 40 |
| relaxed | unlimited | unlimited |
| off | unlimited | unlimited |

When a request is rate-limited, the server returns:

```
HTTP 429 Too Many Requests
Retry-After: 1
{"error": {"message": "Too Many Requests — rate limit exceeded", "type": "rate_limit_error"}}
```

---

## OpenAI Client Compatibility

TrueLLM is designed as an OpenAI API drop-in replacement.  Use any OpenAI
client library by pointing it at `http://your-server:port`:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="your-key-or-empty"
)

response = client.chat.completions.create(
    model="llama-3.1-8b-cuda",
    messages=[{"role": "user", "content": "Hello"}]
)
print(response.choices[0].message.content)
```
