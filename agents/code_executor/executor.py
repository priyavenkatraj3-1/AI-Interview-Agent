"""
Local subprocess-based Python code executor for the coding round.

This is the offline fallback backend: used whenever MOCK_MODE is enabled
(the default -- see agents.config) or CODE_EXECUTION_PROVIDER isn't
"piston", via agents.code_executor.dispatcher. It deliberately does NOT
call an external sandbox service and does NOT depend on the Claude API at
all, so it works identically regardless of MOCK_MODE or Anthropic credit
availability -- see agents.code_executor.piston_executor for the real
sandbox backend (Piston) that MOCK_MODE=false + CODE_EXECUTION_PROVIDER=
piston routes to instead.

This is a best-effort local sandbox, not a hardened multi-tenant one:
- each test case runs candidate code in its own short-lived subprocess,
  never in-process, so a crash/exception can't touch the FastAPI process;
- a hard wall-clock timeout kills a hung/looping submission;
- the subprocess gets a minimal, secret-free environment;
- `python -I -S` (isolated mode, no site/user-site processing) restricts
  what the interpreter picks up from the host;
- on POSIX, CPU-time and address-space rlimits add defense in depth
  (unavailable on Windows -- subprocess module doesn't support preexec_fn
  there, so the timeout is the primary protection on that platform);
- a static denylist rejects obviously dangerous constructs (file/network/
  process/import-machinery access) before anything is executed.
None of this makes arbitrary code execution "safe" in an adversarial,
multi-tenant sense -- it's calibrated for a single local candidate testing
their own solution, not for hostile production traffic. The denylist below
is specific to this backend: it exists because this backend's isolation is
weak, not because untrusted code review is otherwise a good idea. The
Piston backend runs each submission in its own real container instead, so
it does not need (and does not apply) this denylist.
"""
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

from agents.code_executor.base import (
    DEFAULT_TIMEOUT_SECONDS,
    MAX_OUTPUT_CHARS,
    UnsafeCodeError,
    build_harness_script,
    build_result,
    parse_harness_stdout,
    summarize,
)

__all__ = ["DEFAULT_TIMEOUT_SECONDS", "UnsafeCodeError", "run_test_cases"]

# Heuristic, best-effort denylist: reject candidate code containing any of
# these substrings before it's ever executed. Not a substitute for the
# process-level isolation above -- just cheap defense in depth against the
# most obvious escape attempts (file/network/process/import-machinery
# access). Deliberately does not block stdlib modules with no such access
# (math, itertools, collections, re, json, random, etc.).
_DENYLIST_PATTERNS = [
    "import os",
    "import sys",
    "import subprocess",
    "import socket",
    "import shutil",
    "import ctypes",
    "import importlib",
    "import pathlib",
    "__import__",
    "open(",
    "eval(",
    "exec(",
    "compile(",
    "os.",
    "subprocess.",
    "socket.",
]


def _check_denylist(code: str) -> None:
    for pattern in _DENYLIST_PATTERNS:
        if pattern in code:
            raise UnsafeCodeError(f"Submitted code contains a disallowed construct: {pattern!r}")


def _minimal_env() -> dict:
    """A secret-free environment for the subprocess. On Windows, the Python
    interpreter itself needs a couple of system variables at startup (e.g.
    CryptoAPI-backed hash randomization looks for SYSTEMROOT) -- those are
    not secrets, just OS plumbing, so they're preserved."""
    env = {"PATH": os.environ.get("PATH", "")}
    if platform.system() == "Windows":
        for key in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP"):
            value = os.environ.get(key)
            if value:
                env[key] = value
    return env


def _posix_resource_limits(timeout_seconds: float):
    """Returns a preexec_fn applying CPU-time/address-space rlimits, or
    None on platforms where that's not supported (Windows)."""
    if platform.system() == "Windows":
        return None

    import resource

    cpu_seconds = max(1, int(timeout_seconds) + 1)
    mem_bytes = 256 * 1024 * 1024

    def _limit():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))

    return _limit


def _run_one(script_path: Path, tmp_dir: str, env: dict, args: list, expected, timeout_seconds: float) -> dict:
    preexec_fn = _posix_resource_limits(timeout_seconds)
    popen_kwargs = {}
    if preexec_fn is not None:
        popen_kwargs["preexec_fn"] = preexec_fn

    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-S", str(script_path), json.dumps(args)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=tmp_dir,
            env=env,
            **popen_kwargs,
        )
    except subprocess.TimeoutExpired as exc:
        # subprocess.run kills the process for us on timeout; whatever it
        # had already written before being killed is still surfaced.
        return build_result(
            args,
            expected,
            error=f"Timed out after {timeout_seconds}s (possible infinite loop)",
            timed_out=True,
            stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
            stderr=(exc.stderr or "") if isinstance(exc.stderr, str) else "",
        )

    stderr = proc.stderr or ""

    if proc.returncode != 0:
        return build_result(
            args,
            expected,
            error=stderr.strip()[:MAX_OUTPUT_CHARS] or f"Process exited with code {proc.returncode}",
            stdout=proc.stdout or "",
            stderr=stderr,
        )

    return parse_harness_stdout(proc.stdout or "", stderr, args, expected)


def run_test_cases(
    code: str,
    function_name: str,
    test_cases: list[dict],
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict:
    """Run candidate `code` (must define `function_name`) against each test
    case (`{"args": [...], "expected": ...}`), one subprocess per case.

    Returns {"results": [...], "passed_count": int, "total_count": int}.
    Raises UnsafeCodeError if the denylist rejects the code outright --
    callers should treat that as a clean 4xx, not a crash.
    """
    _check_denylist(code)

    results = []
    with tempfile.TemporaryDirectory(prefix="coding_round_") as tmp_dir:
        script_path = Path(tmp_dir) / "candidate_submission.py"
        script_path.write_text(build_harness_script(code, function_name), encoding="utf-8")
        env = _minimal_env()

        for case in test_cases:
            results.append(
                _run_one(script_path, tmp_dir, env, case["args"], case["expected"], timeout_seconds)
            )

    return summarize(results)
