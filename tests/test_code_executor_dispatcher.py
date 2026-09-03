"""
Unit tests for agents/code_executor/dispatcher.py -- the coding-round
executor provider selection (MOCK_MODE / CODE_EXECUTION_PROVIDER).

No network call anywhere here: Piston selection is proven by monkeypatching
agents.code_executor.piston_executor.run_test_cases to a spy, not by
hitting a real Piston instance (see test_piston_executor.py for the
Piston request/response contract itself, exercised via httpx.MockTransport).
"""
import agents.code_executor.dispatcher as dispatcher
import agents.code_executor.executor as local_executor
import agents.code_executor.piston_executor as piston_executor

ADD_CODE = "def add_two(a, b):\n    return a + b\n"
TEST_CASES = [{"args": [2, 3], "expected": 5}]


# --- Provider selection ---


def test_local_is_selected_when_mock_mode_is_true_regardless_of_provider(monkeypatch):
    monkeypatch.setattr(dispatcher, "MOCK_MODE", True)
    monkeypatch.setattr(dispatcher, "CODE_EXECUTION_PROVIDER", "piston")
    assert dispatcher.active_provider() == "local"


def test_piston_is_selected_in_live_mode_when_configured(monkeypatch):
    monkeypatch.setattr(dispatcher, "MOCK_MODE", False)
    monkeypatch.setattr(dispatcher, "CODE_EXECUTION_PROVIDER", "piston")
    assert dispatcher.active_provider() == "piston"


def test_local_is_the_fallback_for_any_unrecognized_provider_in_live_mode(monkeypatch):
    monkeypatch.setattr(dispatcher, "MOCK_MODE", False)
    monkeypatch.setattr(dispatcher, "CODE_EXECUTION_PROVIDER", "judge0")
    assert dispatcher.active_provider() == "local"


def test_local_is_the_default_provider_when_unset_in_live_mode(monkeypatch):
    monkeypatch.setattr(dispatcher, "MOCK_MODE", False)
    monkeypatch.setattr(dispatcher, "CODE_EXECUTION_PROVIDER", "")
    assert dispatcher.active_provider() == "local"


# --- Dispatch actually calls the selected backend ---


def test_dispatcher_calls_piston_run_test_cases_when_piston_is_active(monkeypatch):
    monkeypatch.setattr(dispatcher, "MOCK_MODE", False)
    monkeypatch.setattr(dispatcher, "CODE_EXECUTION_PROVIDER", "piston")

    calls = []

    def fake_run_test_cases(code, function_name, test_cases, timeout_seconds=5.0):
        calls.append({"code": code, "function_name": function_name, "test_cases": test_cases})
        return {"results": [], "passed_count": 0, "total_count": 0}

    monkeypatch.setattr(piston_executor, "run_test_cases", fake_run_test_cases)

    dispatcher.run_test_cases(ADD_CODE, "add_two", TEST_CASES)

    assert len(calls) == 1
    assert calls[0]["code"] == ADD_CODE
    assert calls[0]["function_name"] == "add_two"
    assert calls[0]["test_cases"] == TEST_CASES


def test_dispatcher_never_calls_piston_when_mock_mode_is_true(monkeypatch):
    monkeypatch.setattr(dispatcher, "MOCK_MODE", True)
    monkeypatch.setattr(dispatcher, "CODE_EXECUTION_PROVIDER", "piston")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Piston must never be called while MOCK_MODE is true")

    monkeypatch.setattr(piston_executor, "run_test_cases", fail_if_called)

    summary = dispatcher.run_test_cases(ADD_CODE, "add_two", TEST_CASES)
    assert summary["passed_count"] == 1


# --- Local executor remains directly usable (not only via the dispatcher) ---


def test_local_executor_module_is_still_directly_usable_for_mock_mode_and_tests():
    summary = local_executor.run_test_cases(ADD_CODE, "add_two", TEST_CASES)
    assert summary["passed_count"] == 1
    assert summary["total_count"] == 1
