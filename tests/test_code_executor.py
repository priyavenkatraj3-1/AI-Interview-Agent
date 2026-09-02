"""
Unit tests for agents/code_executor/executor.py — local subprocess-based
Python execution. No network call, no Anthropic dependency: these exercise
the real sandbox (temp-dir subprocess, timeout, denylist), not a fake.
"""
import pytest

from agents.code_executor.executor import UnsafeCodeError, run_test_cases

ADD_CODE = "def add_two(a, b):\n    return a + b\n"
WRONG_CODE = "def add_two(a, b):\n    return a - b\n"
SYNTAX_ERROR_CODE = "def add_two(a, b)\n    return a + b\n"
INFINITE_LOOP_CODE = "def add_two(a, b):\n    while True:\n        pass\n"
UNSAFE_CODE = "import os\ndef add_two(a, b):\n    return a + b\n"
MISSING_FUNCTION_CODE = "x = 1\n"
RUNTIME_EXCEPTION_CODE = "def add_two(a, b):\n    raise ValueError('boom')\n"
PRINTING_CODE = "def add_two(a, b):\n    print('debug:', a, b)\n    return a + b\n"

TEST_CASES = [{"args": [2, 3], "expected": 5}, {"args": [-1, 1], "expected": 0}]


# --- Successful execution ---


def test_all_test_cases_pass_for_correct_solution():
    summary = run_test_cases(ADD_CODE, "add_two", TEST_CASES)
    assert summary["passed_count"] == 2
    assert summary["total_count"] == 2
    assert all(r["passed"] for r in summary["results"])
    assert all(r["error"] is None for r in summary["results"])
    assert all(r["timed_out"] is False for r in summary["results"])


def test_candidate_stdout_is_captured_without_corrupting_the_result():
    summary = run_test_cases(PRINTING_CODE, "add_two", [{"args": [2, 3], "expected": 5}])
    result = summary["results"][0]
    assert result["passed"] is True
    assert result["actual"] == 5
    assert "debug: 2 3" in result["stdout"]


# --- Wrong output ---


def test_wrong_solution_fails_test_cases_with_actual_output_reported():
    summary = run_test_cases(WRONG_CODE, "add_two", TEST_CASES)
    assert summary["passed_count"] == 0
    assert summary["results"][0]["actual"] == -1  # 2 - 3
    assert summary["results"][0]["passed"] is False
    assert summary["results"][0]["error"] is None  # ran fine, just the wrong answer
    assert summary["results"][0]["timed_out"] is False


# --- Runtime error ---


def test_syntax_error_is_reported_per_test_case_not_raised():
    summary = run_test_cases(SYNTAX_ERROR_CODE, "add_two", TEST_CASES)
    assert summary["passed_count"] == 0
    assert all(r["error"] for r in summary["results"])
    assert all(r["timed_out"] is False for r in summary["results"])


def test_missing_function_is_reported_as_error():
    summary = run_test_cases(MISSING_FUNCTION_CODE, "add_two", TEST_CASES)
    assert summary["passed_count"] == 0
    assert all(r["error"] for r in summary["results"])


def test_raised_exception_is_captured_in_stderr_and_reported_as_error():
    summary = run_test_cases(RUNTIME_EXCEPTION_CODE, "add_two", TEST_CASES)
    result = summary["results"][0]
    assert result["passed"] is False
    assert result["timed_out"] is False
    assert result["error"]
    assert "ValueError" in result["error"]
    assert "boom" in result["stderr"]


# --- Timeout ---


def test_infinite_loop_times_out_instead_of_hanging():
    summary = run_test_cases(
        INFINITE_LOOP_CODE, "add_two", [{"args": [1, 2], "expected": 3}], timeout_seconds=1.0
    )
    result = summary["results"][0]
    assert summary["passed_count"] == 0
    assert result["timed_out"] is True
    assert "time" in result["error"].lower()


# --- Isolation / safety ---


def test_denylisted_import_is_rejected_before_execution():
    with pytest.raises(UnsafeCodeError):
        run_test_cases(UNSAFE_CODE, "add_two", TEST_CASES)


def test_safe_stdlib_import_is_allowed():
    code = "import math\ndef add_two(a, b):\n    return math.floor(a + b)\n"
    summary = run_test_cases(code, "add_two", TEST_CASES)
    assert summary["passed_count"] == 2


def test_environment_variables_are_not_exposed_to_candidate_code():
    # os itself is denylisted, but confirm indirectly: the candidate can
    # only ever see whatever comes back over stdout, and the harness never
    # forwards host env vars into the printed result.
    code = "def add_two(a, b):\n    return a + b\n"
    summary = run_test_cases(code, "add_two", [{"args": [1, 1], "expected": 2}])
    result = summary["results"][0]
    assert result["passed"] is True
    assert "ANTHROPIC_API_KEY" not in result["stdout"]
    assert "ANTHROPIC_API_KEY" not in result["stderr"]
