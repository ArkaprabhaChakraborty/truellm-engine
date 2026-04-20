#!/usr/bin/env python3
"""
TrueLLM — Compression pipeline and RLM integration tests.

Tests:
  status-fields   — verify /truellm/v1/status reports compression and rlm fields
  presis-reduces  — confirm Presis token reduction (engine-internal) produces a response
  fastv-metadata  — confirm FastV engine-internal pass doesn't break generation
  rlm-recursive   — confirm RLM generates a response (rlm_generator plugin)
  kv-compress-vq  — verify KvCompressionPass (engine-internal VQ/KiVi) + generate OK
  megakernel      — basic generate() works with use_megakernel=true

Note: compression_preprocessor and kv_compression_kernel are no longer plugins.
Presis, ToMe, VQ, and KiVi run as engine-internal passes (CompressionScoringPass
and KvCompressionPass inside CudaEngine).  Tests verify behaviour via the status
endpoint and generate() responses rather than plugin list membership.

Environment variables:
  TRUELLM_HOST       — server base URL (default: http://127.0.0.1:8080)
  TRUELLM_API_KEY    — bearer token (default: empty)
  TRUELLM_TEST_SUITE — comma-separated test names to run (default: all)

Usage:
  python test_compression_rlm.py
  python test_compression_rlm.py --suite status-fields,presis-reduces
  python test_compression_rlm.py --suite all --verbose
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

_NET_ERRORS = (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
BASE_URL = os.environ.get("TRUELLM_HOST", "http://127.0.0.1:8080")
API_KEY  = os.environ.get("TRUELLM_API_KEY", "")


def _headers():
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
    return h


def get(path: str) -> dict:
    req = urllib.request.Request(BASE_URL + path, headers=_headers())
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def post(path: str, body: dict, timeout: int = 30) -> dict:
    data = json.dumps(body).encode()
    req  = urllib.request.Request(BASE_URL + path, data=data,
                                   headers=_headers(), method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_ready(max_s: int = 60) -> bool:
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
    def __init__(self, name: str):
        self.name    = name
        self.passed  = False
        self.message = ""

    def ok(self, msg: str = ""):
        self.passed  = True
        self.message = msg
        return self

    def fail(self, msg: str):
        self.passed  = False
        self.message = msg
        return self


def _show_response(label: str, resp: dict, max_chars: int = 200) -> None:
    """Print the model's reply inline so the CLI mirrors smoke/batch output."""
    choices = resp.get("choices", []) if isinstance(resp, dict) else []
    if not choices:
        print(f"    [{label}] <no choices>")
        return
    content = choices[0].get("message", {}).get("content", "")
    trimmed = content if len(content) <= max_chars else content[:max_chars] + "…"
    print(f"    [{label}] {trimmed!r}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_status_fields(verbose: bool) -> Result:
    r = Result("status-fields")
    try:
        d = get("/truellm/v1/status")
    except _NET_ERRORS as e:
        return r.fail(f"GET /truellm/v1/status failed: {e}")

    missing = []

    # Compression block — only required when server is configured with
    # at least one compression stage.  We check for presence, not values.
    if "compression" in d:
        comp = d["compression"]
        for stage in ("presis", "tome", "fastv", "pyramid_drop", "kv_vq", "kv_kivi"):
            if stage not in comp:
                missing.append(f"compression.{stage}")
            else:
                if "enabled" not in comp[stage]:
                    missing.append(f"compression.{stage}.enabled")

    # RLM block — only present when rlm.enabled is true.
    if "rlm" in d:
        rlm = d["rlm"]
        for key in ("enabled", "chunk_size", "overlap",
                    "summary_tokens", "max_hierarchy_depth"):
            if key not in rlm:
                missing.append(f"rlm.{key}")

    if missing:
        return r.fail(f"Missing fields: {', '.join(missing)}")

    if verbose:
        print(f"  status keys: {list(d.keys())}")
    return r.ok(f"status={d.get('status')}")


def test_presis_reduces(verbose: bool) -> Result:
    r = Result("presis-reduces")
    # Send a long-ish prompt; if Presis is enabled the server should respond.
    # We cannot directly observe token count reduction from the HTTP API, so
    # we verify that the response is non-empty and no error is returned.
    prompt = " ".join(["hello"] * 512)
    try:
        d = post("/v1/chat/completions", {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8,
            "temperature": 0
        })
    except _NET_ERRORS as e:
        return r.fail(f"POST /v1/chat/completions failed: {e}")

    if "error" in d:
        return r.fail(f"Server returned error: {d['error']}")
    choices = d.get("choices", [])
    if not choices:
        return r.fail("No choices in response")
    _show_response("presis-reduces", d)
    return r.ok(f"got {len(choices)} choice(s)")


def test_fastv_metadata(verbose: bool) -> Result:
    r = Result("fastv-metadata")
    # FastV and PyramidDrop are now engine-internal (no plugin).
    # Verify the status endpoint reports them correctly when enabled,
    # and that a generate() succeeds (engine applies masks internally).
    try:
        d = get("/truellm/v1/status")
    except _NET_ERRORS as e:
        return r.fail(f"GET /truellm/v1/status failed: {e}")

    comp = d.get("compression", {})
    if verbose:
        print(f"  compression config: {comp}")

    # If FastV is reported enabled, run a generate to confirm no crash
    if comp.get("fastv", {}).get("enabled"):
        try:
            d2 = post("/v1/chat/completions", {
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 4,
                "temperature": 0
            })
        except _NET_ERRORS as e:
            return r.fail(f"Generate with FastV enabled failed: {e}")
        if "error" in d2:
            return r.fail(f"Server error with FastV: {d2['error']}")
        return r.ok("FastV engine-internal — generate OK")

    return r.ok("FastV not enabled — engine-internal pass skipped")


def test_rlm_recursive(verbose: bool) -> Result:
    r = Result("rlm-recursive")
    # Send a prompt that would benefit from RLM if enabled.
    # We just verify no error and a non-empty response.
    prompt = " ".join([f"fact {i}: the sky is blue." for i in range(50)])
    try:
        d = post("/v1/chat/completions", {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 16,
            "temperature": 0
        }, timeout=60)
    except _NET_ERRORS as e:
        return r.fail(f"POST failed: {e}")

    if "error" in d:
        return r.fail(f"Server error: {d['error']}")
    choices = d.get("choices", [])
    if not choices:
        return r.fail("No choices")
    _show_response("rlm-recursive", d)
    return r.ok(f"got {len(choices)} choice(s)")


def test_kv_compress_vq(verbose: bool) -> Result:
    r = Result("kv-compress-vq")
    # KV compression (VQ + KiVi) is now engine-internal (KvCompressionPass).
    # No plugin to check for — verify status reports the config and generate works.
    try:
        d = get("/truellm/v1/status")
    except _NET_ERRORS as e:
        return r.fail(f"GET /truellm/v1/status failed: {e}")

    comp  = d.get("compression", {})
    kv_vq = comp.get("kv_vq", {})
    kv_kivi = comp.get("kv_kivi", {})

    if verbose:
        print(f"  kv_vq={kv_vq}  kv_kivi={kv_kivi}")

    if not kv_vq.get("enabled") and not kv_kivi.get("enabled"):
        r.passed = True
        r.message = "KV compression disabled — engine-internal pass inactive, skip"
        return r

    # VQ or KiVi enabled — confirm generation succeeds with compression active
    try:
        d2 = post("/v1/chat/completions", {
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "max_tokens": 8,
            "temperature": 0
        })
    except _NET_ERRORS as e:
        return r.fail(f"Generate with KV compression active failed: {e}")

    if "error" in d2:
        return r.fail(f"Server error with KV compression: {d2['error']}")

    mode = "vq" if kv_vq.get("enabled") else "kivi"
    _show_response(f"kv-compress-{mode}", d2)
    return r.ok(f"KvCompressionPass ({mode}) engine-internal — generate OK")


def test_megakernel(verbose: bool) -> Result:
    r = Result("megakernel")
    # Check if megakernel is reported as enabled.
    try:
        d = get("/truellm/v1/status")
    except _NET_ERRORS as e:
        return r.fail(f"GET /truellm/v1/status failed: {e}")

    backend = d.get("backend", "")
    if backend != "cuda":
        r.passed = True
        r.message = f"backend={backend} — megakernel test skipped (CUDA only)"
        return r

    # Basic generate smoke test.  max_tokens bumped to 16 so the model has
    # room to reach a clean UTF-8 stopping point (max_tokens=4 frequently
    # truncates mid-multibyte on ChatML-family tokenisers).
    try:
        d2 = post("/v1/chat/completions", {
            "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
            "max_tokens": 16,
            "temperature": 0
        })
    except _NET_ERRORS as e:
        return r.fail(f"Generate failed: {e}")

    if "error" in d2:
        return r.fail(f"Server error: {d2['error']}")
    choices = d2.get("choices", [])
    if not choices:
        return r.fail("No choices")
    _show_response("megakernel", d2)
    return r.ok("generate OK on CUDA backend")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
ALL_TESTS = {
    "status-fields":   test_status_fields,
    "presis-reduces":  test_presis_reduces,
    "fastv-metadata":  test_fastv_metadata,
    "rlm-recursive":   test_rlm_recursive,
    "kv-compress-vq":  test_kv_compress_vq,
    "megakernel":      test_megakernel,
}


def main():
    ap = argparse.ArgumentParser(description="TrueLLM compression/RLM integration tests")
    ap.add_argument("--suite", default="all",
                    help="Comma-separated test names, or 'all'")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--wait", type=int, default=60,
                    help="Seconds to wait for server ready (default: 60)")
    args = ap.parse_args()

    suite_env = os.environ.get("TRUELLM_TEST_SUITE", "")
    suite_str = suite_env if suite_env else args.suite

    if suite_str == "all":
        selected = list(ALL_TESTS.keys())
    else:
        selected = [s.strip() for s in suite_str.split(",") if s.strip()]
        unknown  = [s for s in selected if s not in ALL_TESTS]
        if unknown:
            print(f"Unknown tests: {', '.join(unknown)}")
            print(f"Available: {', '.join(ALL_TESTS)}")
            sys.exit(2)

    print(f"Server: {BASE_URL}")
    print(f"Waiting up to {args.wait}s for server ready...")
    if not wait_ready(args.wait):
        print("Server not ready — aborting")
        sys.exit(1)
    print("Server ready.\n")

    results = []
    for name in selected:
        fn = ALL_TESTS[name]
        print(f"  {name} ... ", end="", flush=True)
        try:
            res = fn(args.verbose)
        except _NET_ERRORS as e:
            res = Result(name).fail(f"Unexpected exception: {e}")
        status = "PASS" if res.passed else "FAIL"
        print(f"{status}  {res.message}")
        results.append(res)

    n_pass = sum(1 for r in results if r.passed)
    n_fail = len(results) - n_pass
    print(f"\n{n_pass}/{len(results)} passed", end="")
    if n_fail:
        print(f", {n_fail} FAILED")
        sys.exit(1)
    else:
        print(" — all tests passed")


if __name__ == "__main__":
    main()
