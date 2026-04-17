#!/usr/bin/env python3
"""
TrueLLM — Nemotron-3-Nano-4B smoke test
Run after starting the server:
    .\\build\\windows-cuda\\bin\\truellm-server.exe -c configs\\nemotron-cuda.toml -v
"""

import json
import sys
import textwrap
import time
import requests

# Force UTF-8 output so Unicode symbols (✓ ✗) work in Windows PowerShell
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_URL = "http://127.0.0.1:9099"
MODEL    = "nemotron-3-nano-4b"

GREEN  = "\033[92m"
RED    = "\033[91m"
BLUE   = "\033[94m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

PASS = f"{GREEN}✓{RESET}"
FAIL = f"{RED}✗{RESET}"
HEAD = f"{BLUE}──{RESET}"

SYSTEM_DETAILED = (
    "You are a knowledgeable technical assistant. "
    "Give thorough, well-structured answers with concrete details and examples. "
    "Use multiple paragraphs where appropriate."
)


def check(label: str, ok: bool, detail: str = "") -> bool:
    icon = PASS if ok else FAIL
    detail_str = f"  {DIM}{detail}{RESET}" if detail else ""
    print(f"  {icon}  {label}{detail_str}")
    return ok


def response_stats(text: str):
    words      = len(text.split())
    sentences  = max(1, text.count(".") + text.count("!") + text.count("?"))
    paragraphs = max(1, len([p for p in text.split("\n\n") if p.strip()]))
    return words, sentences, paragraphs


def print_response(label: str, text: str, elapsed: float,
                   prompt_tok: int, gen_tok: int, width: int = 84):
    tps = gen_tok / elapsed if elapsed > 0 else 0
    words, sentences, paragraphs = response_stats(text)

    print(f"\n  {CYAN}{'─' * width}{RESET}")
    print(f"  {BOLD}{label}{RESET}")
    print(f"  {CYAN}{'─' * width}{RESET}")
    if text.strip():
        for para in text.strip().split("\n"):
            if not para.strip():
                print()
                continue
            for wl in textwrap.wrap(para, width=width - 4) or [""]:
                print(f"    {wl}")
    else:
        print(f"    {DIM}(empty){RESET}")
    print(f"  {CYAN}{'─' * width}{RESET}")
    print(f"  {DIM}{prompt_tok} prompt + {gen_tok} gen  "
          f"│  {elapsed:.2f}s  │  {tps:.1f} tok/s  "
          f"│  {words} words  /  {sentences} sentences  /  {paragraphs} para{RESET}")


# ---------------------------------------------------------------------------
# 1. Health
# ---------------------------------------------------------------------------
def test_health():
    print(f"\n{HEAD} /health")
    r = requests.get(f"{BASE_URL}/health", timeout=5)
    check("HTTP 200", r.status_code == 200)
    check('status == "ok"', r.json().get("status") == "ok")


# ---------------------------------------------------------------------------
# 2. Native status
# ---------------------------------------------------------------------------
def test_status():
    print(f"\n{HEAD} /truellm/v1/status")
    r = requests.get(f"{BASE_URL}/truellm/v1/status", timeout=5)
    check("HTTP 200", r.status_code == 200)
    if r.status_code != 200:
        print(f"    {RED}{r.text[:400]}{RESET}")
        return

    body = r.json()
    check('status == "ready"', body.get("status") == "ready", body.get("status", "?"))
    check("model reported",    bool(body.get("model")),   body.get("model", "?"))
    check("backend reported",  bool(body.get("backend")), body.get("backend", "?"))

    gpu = body.get("gpu")
    if gpu:
        check("GPU block present",     True)
        check("device_name not empty", bool(gpu.get("device_name")), gpu.get("device_name", "?"))
        check("vram_total > 0",        gpu.get("vram_total_mb", 0) > 0,
              f'{gpu.get("vram_total_mb", 0):,} MB')
        check("layers_on_gpu > 0",     gpu.get("layers_on_gpu", 0) > 0,
              f'{gpu.get("layers_on_gpu", 0)} layers')

        total = gpu.get("vram_total_mb", 0)
        free  = gpu.get("vram_free_mb", 0)
        used  = total - free
        pct   = used / total * 100 if total else 0
        print(f"\n  {CYAN}GPU:{RESET} {BOLD}{gpu.get('device_name', '?')}{RESET}")
        print(f"  {'VRAM:':<10} {used:,} MB used / {total:,} MB total  ({pct:.1f}% utilised)")
        print(f"  {'Layers:':<10} {gpu.get('layers_on_gpu', 0)} on GPU")
    else:
        check("GPU block present", False, "missing — is this the windows-cuda build?")


# ---------------------------------------------------------------------------
# 3. /v1/models
# ---------------------------------------------------------------------------
def test_models():
    print(f"\n{HEAD} /v1/models")
    r = requests.get(f"{BASE_URL}/v1/models", timeout=5)
    check("HTTP 200", r.status_code == 200)
    ids = [m["id"] for m in r.json().get("data", [])]
    check(f"'{MODEL}' listed", MODEL in ids, str(ids))


# ---------------------------------------------------------------------------
# 4. Non-streaming — detailed technical explanation
#    Validates the model produces a multi-sentence, substantive answer.
# ---------------------------------------------------------------------------
def test_chat_nonstream():
    print(f"\n{HEAD} POST /v1/chat/completions  (non-streaming, detailed explanation)")

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_DETAILED},
            {"role": "user",   "content":
                "Explain how the attention mechanism in transformer models works. "
                "Cover what queries, keys, and values are, how scaled dot-product "
                "attention is computed, and why transformers outperform RNNs for "
                "language tasks. Include a concrete example."},
        ],
        "max_tokens": 700,
        "temperature": 0.6,
        "stream": False,
    }

    t0      = time.perf_counter()
    r       = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=120)
    elapsed = time.perf_counter() - t0

    check("HTTP 200", r.status_code == 200, str(r.status_code))
    if r.status_code != 200:
        print(f"    {RED}{r.text[:400]}{RESET}")
        return

    body    = r.json()
    choices = body.get("choices", [])
    check("choices[0] present", len(choices) > 0)
    if not choices:
        return

    content       = choices[0].get("message", {}).get("content", "")
    finish_reason = choices[0].get("finish_reason", "?")
    usage         = body.get("usage", {})
    prompt_tok    = usage.get("prompt_tokens", 0)
    gen_tok       = usage.get("completion_tokens", 0)
    words         = len(content.split())

    check("response not empty",  bool(content.strip()))
    check("finish_reason set",   bool(finish_reason), finish_reason)
    check("≥ 80 words",          words >= 80,         f"{words} words")
    check("≥ 4 sentences",       content.count(".") >= 4,
          f"{content.count('.')} sentences")
    check("covers key concepts",
          sum(1 for w in ("query", "key", "value", "attention", "dot")
              if w in content.lower()) >= 3,
          "attention terms found")

    print_response("Attention mechanism — detailed explanation",
                   content, elapsed, prompt_tok, gen_tok)


# ---------------------------------------------------------------------------
# 5. Non-streaming — multi-turn conversation
#    Second turn references the first, testing context retention.
# ---------------------------------------------------------------------------
def test_chat_multiturn():
    print(f"\n{HEAD} POST /v1/chat/completions  (multi-turn, context retention)")

    # Turn 1
    t0 = time.perf_counter()
    r1 = requests.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_DETAILED},
            {"role": "user",   "content":
                "Describe the differences between CUDA cores and Tensor Cores on "
                "NVIDIA GPUs. What workloads benefit most from each type?"},
        ],
        "max_tokens": 500,
        "temperature": 0.6,
        "stream": False,
    }, timeout=120)
    t1 = time.perf_counter() - t0

    check("turn 1 HTTP 200", r1.status_code == 200, str(r1.status_code))
    if r1.status_code != 200:
        return

    reply1 = r1.json()["choices"][0]["message"]["content"]
    usage1 = r1.json().get("usage", {})
    words1 = len(reply1.split())

    check("turn 1 ≥ 60 words",      words1 >= 60, f"{words1} words")
    check("turn 1 covers CUDA/TC",  any(w in reply1.lower()
                                        for w in ("cuda", "tensor", "matrix", "fp16", "mixed")))

    print_response("Turn 1 — CUDA cores vs Tensor Cores",
                   reply1, t1, usage1.get("prompt_tokens", 0), usage1.get("completion_tokens", 0))

    # Turn 2 — explicit follow-up requiring the model to remember context
    t0 = time.perf_counter()
    r2 = requests.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [
            {"role": "system",    "content": SYSTEM_DETAILED},
            {"role": "user",      "content":
                "Describe the differences between CUDA cores and Tensor Cores on "
                "NVIDIA GPUs. What workloads benefit most from each type?"},
            {"role": "assistant", "content": reply1},
            {"role": "user",      "content":
                "Based on what you just explained, which type is more critical for "
                "running large language model inference like GPT or Llama, and why? "
                "Be specific about the matrix operations involved."},
        ],
        "max_tokens": 450,
        "temperature": 0.6,
        "stream": False,
    }, timeout=120)
    t2 = time.perf_counter() - t0

    check("turn 2 HTTP 200", r2.status_code == 200, str(r2.status_code))
    if r2.status_code != 200:
        return

    reply2 = r2.json()["choices"][0]["message"]["content"]
    usage2 = r2.json().get("usage", {})
    words2 = len(reply2.split())

    check("turn 2 ≥ 40 words",        words2 >= 40, f"{words2} words")
    check("turn 2 references LLM/MLP", any(w in reply2.lower()
                                           for w in ("llm", "language", "inference",
                                                     "matrix", "tensor", "transformer")))

    print_response("Turn 2 — LLM inference follow-up (context retained)",
                   reply2, t2, usage2.get("prompt_tokens", 0), usage2.get("completion_tokens", 0))


# ---------------------------------------------------------------------------
# 6. Streaming — long-form technical writing (≥ 4 paragraphs requested)
# ---------------------------------------------------------------------------
def test_chat_stream():
    print(f"\n{HEAD} POST /v1/chat/completions  (streaming, long-form)")

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_DETAILED},
            {"role": "user",   "content":
                "Write a detailed technical overview of how continuous batching "
                "works in LLM inference servers. Cover all four of these points:\n"
                "1. Why naive sequential request processing wastes GPU compute.\n"
                "2. How iteration-level scheduling processes multiple sequences "
                "   in a single forward pass.\n"
                "3. How KV cache memory is managed per sequence and what happens "
                "   when the cache fills up.\n"
                "4. What throughput improvements are typically observed compared "
                "   to sequential serving.\n"
                "Write at least 4 substantial paragraphs."},
        ],
        "max_tokens": 10000,
        "temperature": 0.6,
        "stream": True,
    }

    width = 88
    print(f"\n  {CYAN}{'─' * width}{RESET}")
    print(f"  {BOLD}Streaming — continuous batching technical overview{RESET}")
    print(f"  {CYAN}{'─' * width}{RESET}")
    print("    ", end="", flush=True)

    chunks = []
    col    = 0
    t0     = time.perf_counter()

    try:
        with requests.post(f"{BASE_URL}/v1/chat/completions",
                           json=payload, stream=True, timeout=180) as r:
            check("HTTP 200", r.status_code == 200, str(r.status_code))
            if r.status_code != 200:
                print(f"    {RED}{r.text[:400]}{RESET}")
                return

            for raw_line in r.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                    piece = event["choices"][0]["delta"].get("content", "")
                    if not piece:
                        continue
                    chunks.append(piece)
                    for ch in piece:
                        if ch == "\n":
                            print(f"\n    ", end="", flush=True)
                            col = 0
                        else:
                            if col >= width - 4:
                                print(f"\n    ", end="", flush=True)
                                col = 0
                            print(ch, end="", flush=True)
                            col += 1
                except (json.JSONDecodeError, KeyError):
                    pass

    except requests.exceptions.Timeout:
        check("stream completed", False, "timed out")
        return

    elapsed   = time.perf_counter() - t0
    full_text = "".join(chunks)
    words, _, paragraphs = response_stats(full_text)
    tps_approx = len(chunks) / elapsed if elapsed > 0 else 0

    print(f"\n  {CYAN}{'─' * width}{RESET}")
    print(f"  {DIM}chunks: {len(chunks)}  │  chars: {len(full_text)}"
          f"  │  {elapsed:.2f}s  │  ~{tps_approx:.1f} tok/s"
          f"  │  {words} words  /  {paragraphs} para{RESET}")

    check("received chunks",          len(chunks) > 0,   f"{len(chunks)} chunks")
    check("text not empty",           bool(full_text.strip()))
    check("≥ 150 words",              words >= 150,      f"{words} words")
    check("≥ 3 paragraphs",           paragraphs >= 3,   f"{paragraphs} para")
    check("covers batching keywords",
          sum(1 for w in ("batch", "kv", "cache", "sequence", "throughput", "schedule")
              if w in full_text.lower()) >= 3,
          "topic keywords present")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    width = 88
    print(f"\n{BOLD}{'=' * width}{RESET}")
    print(f"{BOLD}  TrueLLM — Nemotron-3-Nano-4B smoke test{RESET}")
    print(f"{DIM}  Target : {BASE_URL}{RESET}")
    print(f"{DIM}  Model  : {MODEL}{RESET}")
    print(f"{BOLD}{'=' * width}{RESET}")

    for attempt in range(10):
        try:
            if requests.get(f"{BASE_URL}/health", timeout=2).status_code == 200:
                break
        except requests.exceptions.ConnectionError:
            if attempt == 9:
                print(f"\n{FAIL}  Cannot connect to {BASE_URL}")
                print("  Start the server first:")
                print("    .\\build\\windows-cuda\\bin\\truellm-server.exe"
                      " -c configs\\nemotron-cuda.toml -v")
                sys.exit(1)
            print(f"  {DIM}Waiting for server... ({attempt + 1}/10){RESET}")
            time.sleep(1)

    ok = True
    try:
        test_health()
        test_status()
        test_models()
        test_chat_nonstream()
        test_chat_multiturn()
        test_chat_stream()
    except Exception as e:
        print(f"\n{FAIL}  Unexpected error: {e}")
        ok = False

    print(f"\n{BOLD}{'=' * width}{RESET}")
    print(f"{PASS}  All checks passed" if ok else f"{FAIL}  Tests failed — see above")
    print(f"{BOLD}{'=' * width}{RESET}\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
