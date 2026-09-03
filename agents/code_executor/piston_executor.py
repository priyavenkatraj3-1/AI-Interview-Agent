"""
Piston-backed real sandbox executor for the coding round.

Sends the candidate's code (plus one test case's JSON-encoded args, as a
command-line argument -- same harness contract as the local executor) to an
external Piston instance (https://github.com/engineer-man/piston) over
HTTP via POST {PISTON_API_URL}/execute, instead of running it as a local
subprocess. This is what makes the coding stage a *real* sandbox: the code
runs in Piston's own isolated container, not on the FastAPI host.

Piston's documented /execute request/response contract (v2):
  request:  {"language": ..., "version": ..., "files": [{"name", "content"}],
             "args": [...], "stdin": "...", "run_timeout": <ms>}
  response: {"language": ..., "version": ..., "run": {"stdout", "stderr",
             "code", "signal", ...}}
"run.signal" is set (and "run.code" is null) when Piston kills the process
itself -- notably on hitting run_timeout -- so a non-null signal is treated
as a timeout here, since the harness script never sends itself a signal.

No source-level denylist is applied here (unlike agents.code_executor.
executor): the isolation boundary is the sandbox container itself, not
string-matching the candidate's code.

Exposes the same run_test_cases(code, function_name, test_cases,
timeout_seconds) -> {"results", "passed_count", "total_count"} contract as
agents.code_executor.executor, so agents.code_executor.dispatcher can swap
between the two backends without any change at the coding_service call
site.
"""
import json

import httpx

from agents.code_executor.base import (
    DEFAULT_TIMEOUT_SECONDS,
    MAX_OUTPUT_CHARS,
    build_harness_script,
    build_result,
    parse_harness_stdout,
    summarize,
)
from agents.config import PISTON_API_URL

PISTON_LANGUAGE = "python"
# "*" asks Piston for whatever Python version the instance has installed,
# per Piston's documented /execute contract -- this project pins no
# specific interpreter version on the sandbox side.
PISTON_VERSION = "*"
SCRIPT_FILENAME = "main.py"

# Piston kills the run at run_timeout (ms) itself; give it a small margin
# above our own timeout_seconds so a genuine sandbox-side kill is
# distinguishable from ordinary network/queueing latency around the HTTP
# call, without leaving a hung submission running far past what the caller
# asked for.
_TIMEOUT_MARGIN_SECONDS = 2.0


class PistonExecutorError(Exception):
    """Raised when the Piston API itself is unreachable, unconfigured, or
    returns a response outside its documented contract -- never used for
    candidate-code failures, which are always reported per-test-case
    instead (see run_test_cases)."""


def _client() -> httpx.Client:
    if not PISTON_API_URL:
        raise PistonExecutorError(
            "PISTON_API_URL is not configured; set it in backend/.env to use "
            "CODE_EXECUTION_PROVIDER=piston."
        )
    return httpx.Client(
        base_url=PISTON_API_URL,
        timeout=DEFAULT_TIMEOUT_SECONDS + _TIMEOUT_MARGIN_SECONDS + 5.0,
    )


def _execute_one(client: httpx.Client, script: str, args: list, expected, timeout_seconds: float) -> dict:
    payload = {
        "language": PISTON_LANGUAGE,
        "version": PISTON_VERSION,
        "files": [{"name": SCRIPT_FILENAME, "content": script}],
        "args": [json.dumps(args)],
        "stdin": "",
        "run_timeout": int((timeout_seconds + _TIMEOUT_MARGIN_SECONDS) * 1000),
    }

    try:
        response = client.post("/execute", json=payload)
        response.raise_for_status()
        body = response.json()
    except httpx.HTTPError as exc:
        return build_result(args, expected, error=f"Sandbox request failed: {exc}"[:MAX_OUTPUT_CHARS])
    except ValueError as exc:  # response body wasn't JSON
        return build_result(
            args, expected, error=f"Sandbox returned an unreadable response: {exc}"[:MAX_OUTPUT_CHARS]
        )

    run = body.get("run")
    if not isinstance(run, dict):
        return build_result(args, expected, error=f"Unexpected sandbox response: {body!r}"[:MAX_OUTPUT_CHARS])

    stdout = run.get("stdout") or ""
    stderr = run.get("stderr") or ""
    signal = run.get("signal")
    code = run.get("code")

    if signal:
        # The harness script never sends itself a signal, so any signal
        # here means the sandbox killed the process -- on this codebase's
        # only configured run_timeout, that means "timed out".
        return build_result(
            args,
            expected,
            error=f"Timed out after {timeout_seconds}s (possible infinite loop)",
            timed_out=True,
            stdout=stdout,
            stderr=stderr,
        )

    if code != 0:
        return build_result(
            args,
            expected,
            error=stderr.strip()[:MAX_OUTPUT_CHARS] or f"Process exited with code {code}",
            stdout=stdout,
            stderr=stderr,
        )

    return parse_harness_stdout(stdout, stderr, args, expected)


def run_test_cases(
    code: str,
    function_name: str,
    test_cases: list[dict],
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    *,
    client: httpx.Client | None = None,
) -> dict:
    """Run candidate `code` against each test case via the Piston API, one
    /execute request per test case (mirroring the local executor's "one
    subprocess per case" granularity).

    `client` is exposed for tests to inject an httpx.Client wired to a
    fake transport (httpx.MockTransport) instead of a real network call;
    production callers should never pass it, so a fresh client (raising
    PistonExecutorError if PISTON_API_URL is unset) is built per call.
    """
    script = build_harness_script(code, function_name)
    results = []
    owns_client = client is None
    http_client = client if client is not None else _client()
    try:
        for case in test_cases:
            results.append(_execute_one(http_client, script, case["args"], case["expected"], timeout_seconds))
    finally:
        if owns_client:
            http_client.close()

    return summarize(results)
