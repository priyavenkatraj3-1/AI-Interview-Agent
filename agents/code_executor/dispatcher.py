"""
Coding-round code-execution provider dispatcher.

Selects which run_test_cases() implementation app.services.coding_service
calls, based on MOCK_MODE and CODE_EXECUTION_PROVIDER (agents/config.py):

- MOCK_MODE=true (the default -- no live sandbox credentials assumed
  available) always uses the local subprocess executor, exactly like every
  other Claude-backed agent in this codebase falls back to a deterministic
  mock in that mode. Keeps coding_service, its tests, and CI fully offline.
- MOCK_MODE=false and CODE_EXECUTION_PROVIDER=piston routes candidate code
  through the real Piston sandbox (agents.code_executor.piston_executor) --
  a genuine external sandbox, not a local subprocess.
- MOCK_MODE=false with any other/unset CODE_EXECUTION_PROVIDER falls back
  to the local executor (documented as not a real sandbox -- see
  agents.code_executor.executor's docstring).

Both backends implement the identical
run_test_cases(code, function_name, test_cases, timeout_seconds) ->
{"results", "passed_count", "total_count"} contract, so this module is the
only place in the coding round that branches on the provider; callers
(app.services.coding_service) import run_test_cases/UnsafeCodeError from
here instead of from either backend directly.
"""
from agents.code_executor import executor as local_executor
from agents.code_executor import piston_executor
from agents.code_executor.base import DEFAULT_TIMEOUT_SECONDS, UnsafeCodeError
from agents.config import CODE_EXECUTION_PROVIDER, MOCK_MODE

__all__ = ["UnsafeCodeError", "active_provider", "run_test_cases"]

PISTON_PROVIDER = "piston"
LOCAL_PROVIDER = "local"


def active_provider() -> str:
    """Name of the provider run_test_cases() will actually use, given
    current config -- exposed so tests (and ops) can assert on selection
    without duplicating the branching logic below."""
    if MOCK_MODE:
        return LOCAL_PROVIDER
    if CODE_EXECUTION_PROVIDER == PISTON_PROVIDER:
        return PISTON_PROVIDER
    return LOCAL_PROVIDER


def run_test_cases(
    code: str,
    function_name: str,
    test_cases: list[dict],
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict:
    if active_provider() == PISTON_PROVIDER:
        return piston_executor.run_test_cases(code, function_name, test_cases, timeout_seconds)
    return local_executor.run_test_cases(code, function_name, test_cases, timeout_seconds)
