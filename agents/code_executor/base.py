"""
Shared result-shape and harness-generation helpers used by every coding-round
code-execution backend (the local subprocess executor in executor.py and the
real-sandbox Piston executor in piston_executor.py), so both implementations
of run_test_cases() return byte-identical result dicts regardless of *where*
the candidate's code actually ran.

Kept here rather than duplicated so "ran to completion but produced bad
output" (missing marker, unparsable JSON) is handled identically no matter
which backend is active.
"""
import json

DEFAULT_TIMEOUT_SECONDS = 5.0

# Truncate captured stderr/stdout used in error messages so one runaway
# print() can't bloat the response or the StageProgress.details JSON blob.
MAX_OUTPUT_CHARS = 2000

# Marks the boundary between whatever the candidate's own code printed
# (captured and returned as-is for visibility/debugging) and the harness's
# own return-value payload, so a candidate's debug print() calls can't be
# mistaken for -- or corrupt parsing of -- the actual function result.
RESULT_MARKER = "\x00__CODING_ROUND_RESULT__\x00"

HARNESS_TEMPLATE = """\
import json as _json
import sys as _sys

{candidate_code}

_args = _json.loads(_sys.argv[1])
_result = {function_name}(*_args)
_sys.stdout.write({marker!r})
_sys.stdout.write(_json.dumps(_result))
"""


class UnsafeCodeError(Exception):
    """Raised when submitted code trips the local executor's static
    denylist -- rejected before any subprocess is ever started. Not raised
    by the Piston backend: a real sandboxed container is the isolation
    boundary there, not a source-level denylist."""


def build_harness_script(code: str, function_name: str) -> str:
    """The candidate's code plus a small driver that calls `function_name`
    with JSON-decoded argv[1] and writes the JSON-encoded result after
    RESULT_MARKER -- identical for every backend, since every backend just
    runs `python script.py <json-args>` one way or another."""
    return HARNESS_TEMPLATE.format(candidate_code=code, function_name=function_name, marker=RESULT_MARKER)


def build_result(args, expected, *, actual=None, passed=False, error=None, timed_out=False, stdout="", stderr="") -> dict:
    return {
        "args": args,
        "expected": expected,
        "actual": actual,
        "passed": passed,
        "error": error,
        "timed_out": timed_out,
        "stdout": stdout[:MAX_OUTPUT_CHARS],
        "stderr": stderr[:MAX_OUTPUT_CHARS],
    }


def parse_harness_stdout(raw_stdout: str, stderr: str, args, expected) -> dict:
    """Parse a harness process's stdout into a result dict, once the process
    is already known to have exited zero (cleanly) with no timeout/signal.
    Shared by every backend."""
    if RESULT_MARKER in raw_stdout:
        candidate_stdout, _, result_json = raw_stdout.partition(RESULT_MARKER)
    else:
        candidate_stdout, result_json = raw_stdout, ""

    if not result_json:
        return build_result(
            args,
            expected,
            error="No result was returned by the submission (missing expected output marker).",
            stdout=candidate_stdout,
            stderr=stderr,
        )

    try:
        actual = json.loads(result_json)
    except json.JSONDecodeError:
        return build_result(
            args,
            expected,
            error=f"Could not parse output as JSON: {result_json[:MAX_OUTPUT_CHARS]!r}",
            stdout=candidate_stdout,
            stderr=stderr,
        )

    return build_result(
        args,
        expected,
        actual=actual,
        passed=actual == expected,
        stdout=candidate_stdout,
        stderr=stderr,
    )


def summarize(results: list[dict]) -> dict:
    passed_count = sum(1 for r in results if r["passed"])
    return {"results": results, "passed_count": passed_count, "total_count": len(results)}
