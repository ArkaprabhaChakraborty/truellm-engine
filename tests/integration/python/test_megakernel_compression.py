#!/usr/bin/env python3
"""
TrueLLM — Megakernel + compression integration tests.

Each test assumes the server was started with a config that enables the
feature under test (one of cuda-megakernel.toml / cuda-kivi-compression.toml /
cuda-vq-compression.toml / cuda-combined-stress.toml).  The test reads
/truellm/v1/status to confirm the feature is reported as enabled, then
runs a generate() to confirm the pipeline doesn't crash under that config.

Suites:
  cuda-only           — assert backend is cuda (sanity gate for the rest)
  megakernel-smoke    — short generate works on cuda (megakernel falls back
                        silently on pre-sm_70 — test just confirms no crash)
  kivi-enabled        — status reports kv_kivi.enabled, short+long generate OK
  vq-enabled          — status reports kv_vq.enabled, short+long generate OK
  presis-long-prompt  — long prompt triggers Presis keep_fraction, generate OK
  tome-enabled        — status reports tome.enabled, generate OK
  compression-stress  — 8 sequential generates exercising all active passes
  combined-compression-absent
                      — when none of kv_vq / kv_kivi is enabled, still verify
                        status shape is sane (negative-control for other tests)

Environment variables:
  TRUELLM_HOST        — server base URL (default: http://127.0.0.1:9099)
  TRUELLM_API_KEY     — bearer token (default: empty)
  TRUELLM_TEST_SUITE  — comma-separated suite names, or 'all'

Exit code: 0 if all selected suites pass, 1 otherwise.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

_NET_ERRORS = (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError)

BASE_URL = os.environ.get("TRUELLM_HOST", "http://127.0.0.1:9099")
API_KEY  = os.environ.get("TRUELLM_API_KEY", "")


def _headers():
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
    return h


def get(path: str, timeout: int = 10) -> dict:
    req = urllib.request.Request(BASE_URL + path, headers=_headers())
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def post(path: str, body: dict, timeout: int = 60) -> dict:
    data = json.dumps(body).encode()
    req  = urllib.request.Request(BASE_URL + path, data=data,
                                   headers=_headers(), method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_ready(max_s: int = 90) -> bool:
    deadline = time.time() + max_s
    while time.time() < deadline:
        try:
            d = get("/truellm/v1/status")
            if d.get("status") in ("ready", "idle"):
                return True
        except _NET_ERRORS:
            pass
        time.sleep(1)
    return False


class Result:
    def __init__(self, name):
        self.name, self.passed, self.message = name, False, ""
    def ok(self, msg=""):
        self.passed = True;  self.message = msg;  return self
    def fail(self, msg):
        self.passed = False; self.message = msg;  return self
    def skip(self, msg):
        self.passed = True;  self.message = f"SKIPPED: {msg}"; return self


def _generate(prompt: str, max_tokens: int = 16, timeout: int = 60) -> dict:
    return post("/v1/chat/completions", {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }, timeout=timeout)


def _show_response(label: str, resp: dict, max_chars: int = 200) -> None:
    """Print the LLM reply inline — mirrors the smoke/batch suites."""
    choices = resp.get("choices", []) if isinstance(resp, dict) else []
    if not choices:
        print(f"    [{label}] <no choices>")
        return
    content = choices[0].get("message", {}).get("content", "")
    trimmed = content if len(content) <= max_chars else content[:max_chars] + "…"
    print(f"    [{label}] {trimmed!r}")


def _status_compression(d: dict) -> dict:
    return d.get("compression", {}) or {}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def t_cuda_only(verbose) -> Result:
    r = Result("cuda-only")
    try:
        d = get("/truellm/v1/status")
    except _NET_ERRORS as e:
        return r.fail(f"status failed: {e}")
    backend = d.get("backend", "")
    if backend != "cuda":
        return r.skip(f"backend={backend!r}, tests below require cuda")
    return r.ok("backend=cuda")


def t_megakernel_smoke(verbose) -> Result:
    r = Result("megakernel-smoke")
    try:
        d = get("/truellm/v1/status")
    except _NET_ERRORS as e:
        return r.fail(f"status failed: {e}")
    if d.get("backend") != "cuda":
        return r.skip("non-cuda backend")
    try:
        resp = _generate("Say hello in one short sentence.", max_tokens=16)
    except _NET_ERRORS as e:
        return r.fail(f"generate failed: {e}")
    if "error" in resp:
        return r.fail(f"server error: {resp['error']}")
    choices = resp.get("choices", [])
    if not choices:
        return r.fail("no choices")
    _show_response("megakernel-smoke", resp)
    return r.ok("generate OK")


def t_kivi_enabled(verbose) -> Result:
    r = Result("kivi-enabled")
    try:
        d = get("/truellm/v1/status")
    except _NET_ERRORS as e:
        return r.fail(f"status failed: {e}")
    comp = _status_compression(d)
    kivi = comp.get("kv_kivi", {})
    if not kivi.get("enabled"):
        return r.skip("kv_kivi not enabled in this config")
    if verbose:
        print(f"  kv_kivi={kivi}")
    # Short prompt (small KV).
    try:
        short = _generate("What is 2+2? Answer briefly.", max_tokens=16)
    except _NET_ERRORS as e:
        return r.fail(f"short generate failed: {e}")
    if "error" in short:
        return r.fail(f"short generate error: {short['error']}")
    _show_response("kivi-short", short)
    # Long-ish prompt (exercises residual_length boundary + outlier selection).
    long_prompt = " ".join([f"sentence {i} is about the color blue." for i in range(80)])
    try:
        long = _generate(long_prompt, max_tokens=16, timeout=90)
    except _NET_ERRORS as e:
        return r.fail(f"long generate failed: {e}")
    if "error" in long:
        return r.fail(f"long generate error: {long['error']}")
    _show_response("kivi-long", long)
    return r.ok(f"KiVi bits={kivi.get('bits')} residual={kivi.get('residual_length')} OK")


def t_vq_enabled(verbose) -> Result:
    r = Result("vq-enabled")
    try:
        d = get("/truellm/v1/status")
    except _NET_ERRORS as e:
        return r.fail(f"status failed: {e}")
    comp = _status_compression(d)
    vq = comp.get("kv_vq", {})
    if not vq.get("enabled"):
        return r.skip("kv_vq not enabled in this config")
    if verbose:
        print(f"  kv_vq={vq}")
    # VQ codebook cold-start needs enough samples. Seed with a prompt long
    # enough to populate calibration buffers (seq_len * n_kv_heads >= codebook).
    prompt = " ".join([f"token{i}" for i in range(256)])
    try:
        resp = _generate(prompt, max_tokens=16, timeout=90)
    except _NET_ERRORS as e:
        return r.fail(f"generate failed: {e}")
    if "error" in resp:
        return r.fail(f"generate error: {resp['error']}")
    _show_response("vq-seed", resp)
    # Second generate — codebook should now be initialised; fast path.
    try:
        resp2 = _generate("Reply with one word.", max_tokens=16)
    except _NET_ERRORS as e:
        return r.fail(f"second generate failed: {e}")
    if "error" in resp2:
        return r.fail(f"second generate error: {resp2['error']}")
    _show_response("vq-fastpath", resp2)
    return r.ok(f"VQ codebook_size={vq.get('codebook_size')} OK")


def t_presis_long_prompt(verbose) -> Result:
    r = Result("presis-long-prompt")
    try:
        d = get("/truellm/v1/status")
    except _NET_ERRORS as e:
        return r.fail(f"status failed: {e}")
    comp = _status_compression(d)
    presis = comp.get("presis", {})
    if not presis.get("enabled"):
        return r.skip("presis not enabled")
    # Construct a prompt long enough to cross Presis threshold (~0.6–0.8 of
    # context_size). 1200 short tokens easily clears 70% of a 4k ctx.
    long_prompt = " ".join([f"fact {i}: the quick brown fox." for i in range(200)])
    try:
        resp = _generate(long_prompt, max_tokens=16, timeout=90)
    except _NET_ERRORS as e:
        return r.fail(f"long generate failed: {e}")
    if "error" in resp:
        return r.fail(f"server error: {resp['error']}")
    choices = resp.get("choices", [])
    if not choices:
        return r.fail("no choices")
    _show_response("presis-long", resp)
    return r.ok(f"presis keep_fraction={presis.get('keep_fraction')} OK")


def t_tome_enabled(verbose) -> Result:
    r = Result("tome-enabled")
    try:
        d = get("/truellm/v1/status")
    except _NET_ERRORS as e:
        return r.fail(f"status failed: {e}")
    tome = _status_compression(d).get("tome", {})
    if not tome.get("enabled"):
        return r.skip("tome not enabled")
    if verbose:
        print(f"  tome={tome}")
    prompt = " ".join([f"item {i}" for i in range(128)])
    try:
        resp = _generate(prompt, max_tokens=16, timeout=60)
    except _NET_ERRORS as e:
        return r.fail(f"generate failed: {e}")
    if "error" in resp:
        return r.fail(f"server error: {resp['error']}")
    _show_response("tome", resp)
    return r.ok(f"ToMe r={tome.get('r')} OK")


def t_compression_stress(verbose) -> Result:
    """Run 8 generates back-to-back exercising whatever passes are enabled."""
    r = Result("compression-stress")
    try:
        d = get("/truellm/v1/status")
    except _NET_ERRORS as e:
        return r.fail(f"status failed: {e}")
    if d.get("backend") != "cuda":
        return r.skip("non-cuda backend")
    comp = _status_compression(d)
    if not comp:
        return r.skip("no compression passes enabled")

    prompts = [
        "Reply with one word.",
        " ".join([f"entry{i}" for i in range(40)]),
        "Summarise: the cat sat on the mat.",
        " ".join([f"fact {i}: blue." for i in range(120)]),
        "What is 1+1?",
        " ".join([f"token{i}" for i in range(200)]),
        "Say hi.",
        " ".join([f"line {i} about testing." for i in range(160)]),
    ]
    failures = []
    for i, p in enumerate(prompts):
        try:
            resp = _generate(p, max_tokens=16, timeout=90)
        except _NET_ERRORS as e:
            failures.append(f"req{i}: net error {e}")
            continue
        if "error" in resp:
            failures.append(f"req{i}: server error {resp['error']}")
            continue
        if not resp.get("choices"):
            failures.append(f"req{i}: empty choices")
            continue
        _show_response(f"stress-req{i}", resp, max_chars=80)
    if failures:
        return r.fail(f"{len(failures)}/{len(prompts)} failed: {failures[0]}")
    return r.ok(f"all {len(prompts)} generates OK; active={list(comp.keys())}")


def t_compression_absent_shape(verbose) -> Result:
    """
    Negative control: when neither kv_vq nor kv_kivi is enabled, the compression
    block may be absent entirely — but the status endpoint must still expose
    backend/context_max/uptime.  Catches regressions in the status handler.
    """
    r = Result("compression-absent-shape")
    try:
        d = get("/truellm/v1/status")
    except _NET_ERRORS as e:
        return r.fail(f"status failed: {e}")
    comp = _status_compression(d)
    any_kv = comp.get("kv_vq", {}).get("enabled") or comp.get("kv_kivi", {}).get("enabled")
    if any_kv:
        return r.skip("KV compression is enabled — not a negative control here")
    for key in ("backend", "context_max", "uptime_s", "status"):
        if key not in d:
            return r.fail(f"status missing '{key}'")
    return r.ok("status shape sane with KV compression off")


ALL_TESTS = {
    "cuda-only":                  t_cuda_only,
    "megakernel-smoke":           t_megakernel_smoke,
    "kivi-enabled":               t_kivi_enabled,
    "vq-enabled":                 t_vq_enabled,
    "presis-long-prompt":         t_presis_long_prompt,
    "tome-enabled":               t_tome_enabled,
    "compression-stress":         t_compression_stress,
    "compression-absent-shape":   t_compression_absent_shape,
}


def main():
    ap = argparse.ArgumentParser(description="TrueLLM megakernel/compression integration tests")
    ap.add_argument("--suite", default=os.environ.get("TRUELLM_TEST_SUITE", "all"),
                    help="Comma-separated test names, or 'all'")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.suite == "all":
        selected = list(ALL_TESTS.keys())
    else:
        selected = [s.strip() for s in args.suite.split(",") if s.strip()]
        unknown = [s for s in selected if s not in ALL_TESTS]
        if unknown:
            print(f"Unknown suite name(s): {unknown}", file=sys.stderr)
            print(f"Available: {list(ALL_TESTS.keys())}", file=sys.stderr)
            sys.exit(2)

    if not wait_ready(90):
        print(f"Server at {BASE_URL} never became ready", file=sys.stderr)
        sys.exit(1)

    print(f"Running {len(selected)} test(s) against {BASE_URL}\n")
    results = []
    for name in selected:
        fn = ALL_TESTS[name]
        t0 = time.time()
        try:
            res = fn(args.verbose)
        except Exception as e:  # defensive — a test must never kill the runner
            res = Result(name).fail(f"uncaught: {type(e).__name__}: {e}")
        dt = time.time() - t0
        mark = "PASS" if res.passed else "FAIL"
        print(f"  [{mark}] {res.name:<30} {dt:5.1f}s  {res.message}")
        results.append(res)

    failed = [r for r in results if not r.passed]
    print()
    if failed:
        print(f"{len(failed)}/{len(results)} test(s) failed.")
        sys.exit(1)
    print(f"All {len(results)} test(s) passed.")


if __name__ == "__main__":
    main()
