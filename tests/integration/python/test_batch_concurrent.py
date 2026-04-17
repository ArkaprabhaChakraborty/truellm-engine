#!/usr/bin/env python3
"""
TrueLLM — Stress test suite

Modes:
  burst     — N requests fired simultaneously (default)
  ramp      — concurrency 1→2→4→...→N, find throughput saturation point
  sustained — N workers running continuously for --duration seconds
  gpu-ctx   — burst with long prompts to stress the paged KV cache

Run examples:
  python test_batch_concurrent.py
  python test_batch_concurrent.py --mode burst     --n 16 --tokens 128
  python test_batch_concurrent.py --mode ramp      --n 16
  python test_batch_concurrent.py --mode sustained --n 8  --duration 60
  python test_batch_concurrent.py --mode gpu-ctx   --n 4  --tokens 512

GPU configs to pair with:
  test-batching-stress.toml  — batching on, max_seqs=8   (quick GPU burst)
  test-gpu-stress.toml       — batching on, max_seqs=16, ctx=8192 (heavy)
  test-cuda-engine.toml      — CudaEngine paged-attention baseline
  test-cuda-paged-kv.toml    — per-layer KV layout validation
  test-cuda-long-ctx.toml    — long context / v2 paged-attention path
"""

import argparse
import sys
import threading
import time

import requests

# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

# ── Prompts ────────────────────────────────────────────────────────────────────
SHORT_PROMPTS = [
    "Name the capital of France in one sentence.",
    "What is 17 × 13? Answer with just the number.",
    "Name one famous physicist.",
    "What colour is the sky? One word.",
    "Name one programming language.",
    "What is the boiling point of water in Celsius?",
    "Name one planet in our solar system.",
    "What is 100 divided by 4?",
    "Name one continent.",
    "What is the speed of light approximately?",
    "Name one chemical element.",
    "What year did World War II end?",
    "Who wrote Hamlet?",
    "What is the chemical symbol for gold?",
    "What is the square root of 144?",
    "Name one ocean.",
]

# Long prompts for gpu-ctx mode — force large KV allocation per sequence.
LONG_PROMPTS = [
    "Explain in detail how an HTTP request travels from a browser through DNS "
    "resolution, TCP handshake, TLS negotiation, request transmission, and back "
    "to the client. Cover each step thoroughly.",

    "Walk through exactly how a transformer attention layer computes its output: "
    "QKV projections, scaled dot-product attention with masking, softmax, the "
    "output projection, and how this plugs into the residual stream.",

    "Describe how a GPU executes a CUDA kernel: grid and block dimensions, warp "
    "scheduling, shared memory, register files, memory coalescing, and how bank "
    "conflicts degrade performance. Give a concrete matrix-multiply example.",

    "Explain how paged virtual memory works in a modern operating system: page "
    "tables, TLB hits and misses, page fault handling, demand paging, copy-on-write "
    "fork semantics, and how swapping interacts with the page replacement policy.",

    "Walk through how a C++ compiler converts source code to a binary: lexing, "
    "parsing, AST construction, semantic analysis, IR lowering, optimisation passes "
    "(inlining, loop vectorisation, dead-code elimination), register allocation, "
    "and final code generation.",

    "Describe how TCP congestion control works end-to-end: slow start, congestion "
    "avoidance, fast retransmit, fast recovery, the CUBIC algorithm, and how "
    "BBR differs from loss-based approaches.",

    "Explain how a relational database executes a SELECT with multiple JOINs: "
    "query parsing, logical planning, cost-based physical plan selection, index "
    "scans, hash joins, merge joins, and result streaming to the client.",

    "Describe the full pipeline of LLM inference from user prompt to final text: "
    "tokenisation, embedding lookup, positional encoding, all transformer layers "
    "(attention + FFN), KV cache management across decode steps, sampling "
    "strategies, and detokenisation.",
]


# ── Request sender ─────────────────────────────────────────────────────────────
def send_request(idx: int, prompt: str, max_tokens: int,
                 results: list, errors: list,
                 base_url: str, model: str):
    """Fire one chat/completions request. Appends to results or errors."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "stream": False,
    }
    try:
        t0 = time.perf_counter()
        r  = requests.post(f"{base_url}/v1/chat/completions",
                           json=payload, timeout=300)
        elapsed = time.perf_counter() - t0

        if r.status_code != 200:
            errors.append((idx, f"HTTP {r.status_code}: {r.text[:200]}"))
            return

        body       = r.json()
        text       = body["choices"][0]["message"]["content"]
        usage      = body.get("usage", {})
        gen_tok    = usage.get("completion_tokens", 0)
        prompt_tok = usage.get("prompt_tokens", 0)
        results.append((idx, elapsed, gen_tok, prompt_tok, text, prompt))

    except Exception as exc:
        errors.append((idx, str(exc)))


# ── Statistics helpers ─────────────────────────────────────────────────────────
def percentile(sorted_vals: list, p: float) -> float:
    if not sorted_vals:
        return 0.0
    k  = (len(sorted_vals) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def print_results(results: list, errors: list, wall: float, width: int = 72):
    """Print per-request table with full responses, then aggregate stats."""
    results.sort(key=lambda x: x[0])
    total_gen = 0

    for idx, elapsed, gen_tok, prompt_tok, text, prompt in results:
        total_gen += gen_tok
        tps = gen_tok / elapsed if elapsed > 0 else 0.0
        print(f"  [{idx:3d}] {elapsed:6.2f}s  "
              f"prompt={prompt_tok:4d} tok  gen={gen_tok:4d} tok  {tps:5.1f} t/s")
        # Print prompt
        prompt_short = prompt if len(prompt) <= 80 else prompt[:77] + "…"
        print(f"         {DIM}Q: {prompt_short}{RESET}")
        # Print full response
        for line in text.strip().splitlines():
            print(f"         {line}")
        print()

    for idx, msg in errors:
        print(f"  {RED}[{idx:3d}] ERR {msg}{RESET}")

    print(f"  {CYAN}{'─' * width}{RESET}")

    if not results:
        print(f"  {RED}No successful results.{RESET}")
        return

    latencies = sorted(r[1] for r in results)
    n_ok  = len(results)
    n_err = len(errors)
    p50   = percentile(latencies, 50)
    p95   = percentile(latencies, 95)
    p99   = percentile(latencies, 99)
    agg_tps = total_gen / wall if wall > 0 else 0

    print(f"\n  {BOLD}Summary{RESET}")
    print(f"  {'Completed:':<22} {n_ok} ok  {n_err} errors")
    print(f"  {'Wall time:':<22} {wall:.2f}s")
    print(f"  {'Latency p50:':<22} {p50:.2f}s")
    print(f"  {'Latency p95:':<22} {p95:.2f}s")
    print(f"  {'Latency p99:':<22} {p99:.2f}s")
    print(f"  {'Total tokens generated:':<22} {total_gen}")
    print(f"  {'Aggregate tok/s:':<22} {BOLD}{agg_tps:.1f}{RESET}  "
          f"{DIM}(total gen tokens / wall time){RESET}")

    if n_err == 0:
        print(f"\n{GREEN}{BOLD}  All {n_ok} requests completed successfully.{RESET}")
    else:
        print(f"\n{YELLOW}{BOLD}  {n_err} request(s) failed — check server logs.{RESET}")


# ── Burst mode ─────────────────────────────────────────────────────────────────
def run_burst(n: int, max_tokens: int, prompts: list,
              base_url: str, model: str) -> tuple:
    """Fire n requests simultaneously and wait for all."""
    prompt_list = [prompts[i % len(prompts)] for i in range(n)]
    results, errors = [], []

    threads = [
        threading.Thread(
            target=send_request,
            args=(i, prompt_list[i], max_tokens, results, errors, base_url, model),
            daemon=True,
        )
        for i in range(n)
    ]

    wall_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results, errors, time.perf_counter() - wall_start


# ── Ramp mode ──────────────────────────────────────────────────────────────────
def run_ramp(max_n: int, max_tokens: int, base_url: str, model: str):
    """Step concurrency from 1 → max_n (doubling), report throughput at each level."""
    print(f"\n  {BOLD}Ramp mode — concurrency 1 → {max_n}{RESET}")
    print(f"  {DIM}Fires a burst at each concurrency level; shows throughput plateau.{RESET}\n")
    print(f"  {'Concurrency':>12}  {'Wall(s)':>8}  {'Agg tok/s':>10}  "
          f"{'p50(s)':>7}  {'p95(s)':>7}  {'Errors':>6}")
    print(f"  {'─' * 58}")

    levels = []
    n = 1
    while True:
        results, errors, wall = run_burst(n, max_tokens, SHORT_PROMPTS, base_url, model)
        total_gen = sum(r[2] for r in results)
        agg_tps   = total_gen / wall if wall > 0 else 0
        latencies = sorted(r[1] for r in results)
        p50 = percentile(latencies, 50)
        p95 = percentile(latencies, 95)
        n_err = len(errors)

        improving = levels and agg_tps > levels[-1][3]
        marker    = f"  {GREEN}▲{RESET}" if improving else "   "
        color     = GREEN if improving else RESET
        print(f"{marker} {n:>12}  {wall:>8.2f}  "
              f"{color}{agg_tps:>10.1f}{RESET}  "
              f"{p50:>7.2f}  {p95:>7.2f}  "
              f"{RED if n_err else ''}{n_err:>6}{RESET if n_err else ''}")

        levels.append((n, wall, total_gen, agg_tps))

        if n >= max_n:
            break
        n = min(n * 2, max_n)

    best = max(levels, key=lambda x: x[3])
    print(f"\n  {BOLD}Peak: {best[3]:.1f} tok/s at concurrency {best[0]}{RESET}")


# ── Sustained mode ─────────────────────────────────────────────────────────────
def run_sustained(n_workers: int, max_tokens: int, duration_s: int,
                  base_url: str, model: str) -> tuple:
    """Keep n_workers continuously firing requests for duration_s seconds."""
    results, errors = [], []
    lock       = threading.Lock()
    stop_event = threading.Event()
    counter    = [0]

    def worker():
        while not stop_event.is_set():
            with lock:
                idx = counter[0]
                counter[0] += 1
            prompt = SHORT_PROMPTS[idx % len(SHORT_PROMPTS)]
            r, e = [], []
            send_request(idx, prompt, max_tokens, r, e, base_url, model)
            with lock:
                results.extend(r)
                errors.extend(e)
                done = len(results)
            sys.stdout.write(f"\r  {done:4d} completed, {len(errors):3d} errors…  ")
            sys.stdout.flush()

    threads = [threading.Thread(target=worker, daemon=True)
               for _ in range(n_workers)]

    wall_start = time.perf_counter()
    for t in threads:
        t.start()
    time.sleep(duration_s)
    stop_event.set()
    for t in threads:
        t.join(timeout=60)
    print()  # newline after progress line
    return results, errors, time.perf_counter() - wall_start


# ── GPU context stress mode ────────────────────────────────────────────────────
def run_gpu_ctx(n: int, max_tokens: int, base_url: str, model: str) -> tuple:
    """Burst with long prompts — forces large KV block allocation per sequence."""
    return run_burst(n, max_tokens, LONG_PROMPTS, base_url, model)


# ── Server helpers ─────────────────────────────────────────────────────────────
def check_server(base_url: str):
    try:
        r = requests.get(f"{base_url}/health", timeout=3)
        if r.status_code != 200:
            print(f"{RED}Server not healthy (HTTP {r.status_code}){RESET}")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print(f"{RED}Cannot connect to {base_url}{RESET}")
        print("  Start the server first, then re-run.")
        sys.exit(1)


def print_server_info(base_url: str):
    try:
        r = requests.get(f"{base_url}/truellm/v1/status", timeout=3)
        if r.status_code != 200:
            return
        b   = r.json()
        kv  = b.get("kv_cache", {})
        gpu = b.get("gpu", {})
        print(f"  {DIM}model={b.get('model','?')}  "
              f"backend={b.get('backend','?')}  "
              f"ctx={b.get('context_max',0)}")
        if gpu.get("device_name"):
            free_mb  = gpu.get("vram_free_bytes",  0) // (1024 * 1024)
            total_mb = gpu.get("vram_total_bytes", 0) // (1024 * 1024)
            layers   = gpu.get("layers_on_gpu", 0)
            print(f"  GPU : {gpu['device_name']}  "
                  f"VRAM {free_mb} MB free / {total_mb} MB total  "
                  f"layers_on_gpu={layers}")
        if kv.get("blocks_total", 0) > 0:
            print(f"  KV  : {kv['blocks_used']}/{kv['blocks_total']} blocks used  "
                  f"({kv.get('utilization_pct', 0):.1f}%)  "
                  f"block_size={kv.get('block_size_tokens', 0)} tok  "
                  f"fp8={kv.get('fp8_enabled', False)}")
        print(RESET, end="")
    except Exception:
        pass


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="TrueLLM stress test suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
modes:
  burst      N requests simultaneously, print all responses + stats  (default)
  ramp       Ramp concurrency 1→N, find throughput saturation
  sustained  N workers run continuously for --duration seconds
  gpu-ctx    Burst with long prompts to stress paged KV cache

GPU configs:
  test-batching-stress.toml  max_seqs=8,  ctx=4096  quick concurrent GPU test
  test-gpu-stress.toml       max_seqs=16, ctx=8192  heavy sustained GPU test
  test-cuda-engine.toml      CudaEngine paged-attention baseline
  test-cuda-long-ctx.toml    long context, paged_attention_v2 path
        """
    )
    parser.add_argument("--mode",     default="burst",
                        choices=["burst", "ramp", "sustained", "gpu-ctx"],
                        help="Test mode (default: burst)")
    parser.add_argument("--n",        type=int, default=8,
                        help="Concurrent requests / workers (default: 8)")
    parser.add_argument("--tokens",   type=int, default=128,
                        help="max_tokens per request (default: 128)")
    parser.add_argument("--duration", type=int, default=30,
                        help="Duration in seconds for sustained mode (default: 30)")
    parser.add_argument("--url",      default="http://127.0.0.1:9099",
                        help="Server base URL (default: http://127.0.0.1:9099)")
    parser.add_argument("--model",    default="nemotron-3-nano-4b",
                        help="Model alias (default: nemotron-3-nano-4b)")
    args = parser.parse_args()

    width = 72
    print(f"\n{BOLD}{'=' * width}{RESET}")
    print(f"{BOLD}  TrueLLM Stress Test  [{args.mode.upper()}]{RESET}")
    print(f"{DIM}  Target : {args.url}{RESET}")
    print(f"{DIM}  Model  : {args.model}{RESET}")
    print(f"{BOLD}{'=' * width}{RESET}\n")

    check_server(args.url)
    print_server_info(args.url)

    ok = True

    if args.mode == "burst":
        print(f"\n  Firing {BOLD}{args.n}{RESET} requests simultaneously  "
              f"({args.tokens} tok each)…\n")
        print(f"  {CYAN}{'─' * width}{RESET}\n")
        results, errors, wall = run_burst(
            args.n, args.tokens, SHORT_PROMPTS, args.url, args.model)
        print_results(results, errors, wall, width)
        ok = len(errors) == 0

    elif args.mode == "ramp":
        run_ramp(args.n, args.tokens, args.url, args.model)
        ok = True

    elif args.mode == "sustained":
        print(f"\n  {args.n} workers × {args.duration}s  "
              f"({args.tokens} tok each)…")
        print(f"  {DIM}Requests fire back-to-back per worker until time is up.{RESET}\n")
        results, errors, wall = run_sustained(
            args.n, args.tokens, args.duration, args.url, args.model)
        print(f"\n  {CYAN}{'─' * width}{RESET}\n")
        print_results(results, errors, wall, width)

        # Sustained-specific: steady-state throughput
        total_gen = sum(r[2] for r in results)
        rps = len(results) / wall if wall > 0 else 0
        print(f"\n  {BOLD}Steady-state:{RESET}")
        print(f"  {'Requests/s:':<22} {rps:.2f}")
        print(f"  {'Tokens/s (total):':<22} {total_gen / wall if wall > 0 else 0:.1f}")
        ok = len(errors) == 0

    elif args.mode == "gpu-ctx":
        print(f"\n  GPU context stress: {BOLD}{args.n}{RESET} long-prompt requests  "
              f"({args.tokens} tok each)…")
        print(f"  {DIM}Long prompts force large KV block allocation per sequence.")
        print(f"  Use test-gpu-stress.toml (ctx=8192, max_seqs=16) for maximum pressure.{RESET}\n")
        print(f"  {CYAN}{'─' * width}{RESET}\n")
        results, errors, wall = run_gpu_ctx(
            args.n, args.tokens, args.url, args.model)
        print_results(results, errors, wall, width)
        ok = len(errors) == 0

    print(f"\n  {DIM}─ Config guide ─────────────────────────────────────────────")
    print(f"  burst/ramp/sustained → test-batching-stress.toml  (GgmlEngine, max_seqs=8)")
    print(f"  gpu-ctx              → test-gpu-stress.toml        (ctx=8192, max_seqs=16)")
    print(f"  CudaEngine           → test-cuda-engine.toml       (paged-attention){RESET}")
    print(f"\n{BOLD}{'=' * width}{RESET}\n")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
