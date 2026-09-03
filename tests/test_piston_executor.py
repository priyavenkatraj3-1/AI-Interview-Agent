"""
Unit tests for agents/code_executor/piston_executor.py -- the real-sandbox
(Piston) code executor.

Every test injects an httpx.Client wired to httpx.MockTransport (ships with
httpx, already a transitive dependency via anthropic -- no new test
dependency), so these run fully offline and never make a real network call
to a Piston instance. They prove the *request/response contract* is wired
correctly, not that a live Piston service is reachable -- see
test_coding_sandbox_dispatch.py and the task report for what live
verification would require.
"""
import json

import httpx
import pytest

from agents.code_executor.piston_executor import PistonExecutorError, run_test_cases

ADD_CODE = "def add_two(a, b):\n    return a + b\n"
TEST_CASES = [{"args": [2, 3], "expected": 5}, {"args": [-1, 1], "expected": 0}]
RESULT_MARKER = "\x00__CODING_ROUND_RESULT__\x00"


def _mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="https://fake-piston.test/api/v2")


def _piston_response(stdout: str = "", stderr: str = "", code=0, signal=None) -> httpx.Response:
    return httpx.Response(200, json={"language": "python", "version": "3.11.0", "run": {"stdout": stdout, "stderr": stderr, "code": code, "signal": signal}})


def _add_two_handler(request: httpx.Request) -> httpx.Response:
    """Emulates a real Piston instance executing the harness script: reads
    the JSON args from `args[0]` (as the real interpreter would via
    sys.argv[1]) and returns their sum, marker-delimited like the real
    harness's stdout."""
    body = json.loads(request.content)
    args = json.loads(body["args"][0])
    result = args[0] + args[1]
    return _piston_response(stdout=RESULT_MARKER + json.dumps(result))


# --- Request contract ---


def test_request_contains_candidate_code_and_test_case_args():
    captured_bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_bodies.append(json.loads(request.content))
        return _add_two_handler(request)

    summary = run_test_cases(ADD_CODE, "add_two", TEST_CASES, client=_mock_client(handler))

    assert summary["total_count"] == 2
    assert len(captured_bodies) == 2
    for body, case in zip(captured_bodies, TEST_CASES):
        assert body["language"] == "python"
        assert ADD_CODE in body["files"][0]["content"]
        assert "add_two" in body["files"][0]["content"]
        assert json.loads(body["args"][0]) == case["args"]


def test_request_hits_the_execute_endpoint():
    seen_paths = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        return _add_two_handler(request)

    run_test_cases(ADD_CODE, "add_two", [TEST_CASES[0]], client=_mock_client(handler))
    assert seen_paths == ["/api/v2/execute"]


# --- Successful execution ---


def test_correct_solution_passes_all_test_cases_via_piston():
    summary = run_test_cases(ADD_CODE, "add_two", TEST_CASES, client=_mock_client(_add_two_handler))
    assert summary["passed_count"] == 2
    assert summary["total_count"] == 2
    assert all(r["passed"] for r in summary["results"])
    assert all(r["error"] is None for r in summary["results"])


def test_wrong_solution_is_reported_as_failed_not_raised():
    def handler(request: httpx.Request) -> httpx.Response:
        return _piston_response(stdout=RESULT_MARKER + json.dumps(999))

    summary = run_test_cases(ADD_CODE, "add_two", [TEST_CASES[0]], client=_mock_client(handler))
    result = summary["results"][0]
    assert result["passed"] is False
    assert result["actual"] == 999
    assert result["expected"] == 5


# --- Error handling ---


def test_runtime_error_is_mapped_from_nonzero_exit_code():
    def handler(request: httpx.Request) -> httpx.Response:
        return _piston_response(stderr="Traceback (most recent call last):\nNameError: name 'x' is not defined", code=1)

    summary = run_test_cases(ADD_CODE, "add_two", [TEST_CASES[0]], client=_mock_client(handler))
    result = summary["results"][0]
    assert result["passed"] is False
    assert result["timed_out"] is False
    assert "NameError" in result["error"]


def test_syntax_error_is_mapped_from_nonzero_exit_code():
    def handler(request: httpx.Request) -> httpx.Response:
        return _piston_response(stderr="  File \"main.py\", line 1\nSyntaxError: invalid syntax", code=1)

    summary = run_test_cases("def add_two(a, b)\n    return a + b\n", "add_two", [TEST_CASES[0]], client=_mock_client(handler))
    result = summary["results"][0]
    assert result["passed"] is False
    assert "SyntaxError" in result["error"]


# --- Timeout ---


def test_sandbox_kill_signal_is_mapped_to_timed_out():
    def handler(request: httpx.Request) -> httpx.Response:
        # Piston's documented behavior on hitting run_timeout: process is
        # killed, code is null, signal is set.
        return _piston_response(code=None, signal="SIGKILL")

    summary = run_test_cases(ADD_CODE, "add_two", [TEST_CASES[0]], client=_mock_client(handler))
    result = summary["results"][0]
    assert result["timed_out"] is True
    assert result["passed"] is False
    assert "time" in result["error"].lower()


# --- Transport / service-level failures (never raised to the caller) ---


def test_http_error_status_is_reported_per_test_case_not_raised():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal server error")

    summary = run_test_cases(ADD_CODE, "add_two", [TEST_CASES[0]], client=_mock_client(handler))
    result = summary["results"][0]
    assert result["passed"] is False
    assert result["error"]
    assert summary["passed_count"] == 0


def test_unreadable_response_body_is_reported_per_test_case_not_raised():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    summary = run_test_cases(ADD_CODE, "add_two", [TEST_CASES[0]], client=_mock_client(handler))
    result = summary["results"][0]
    assert result["passed"] is False
    assert result["error"]


def test_missing_piston_api_url_raises_a_clear_configuration_error(monkeypatch):
    import agents.code_executor.piston_executor as piston_executor_module

    monkeypatch.setattr(piston_executor_module, "PISTON_API_URL", "")
    with pytest.raises(PistonExecutorError):
        run_test_cases(ADD_CODE, "add_two", TEST_CASES)
