"""
Integration tests for the coding round API (start / current / run / submit /
result), exercised through FastAPI's TestClient against a real (temp)
SQLite DB. The code-problem generator is replaced by the deterministic
`fake_code_problem_generator` fixture (autouse, see conftest.py) so these
run offline; code execution goes through the real local subprocess executor
(agents.code_executor.executor) since it makes no network call and needs no
Anthropic credits either — see test_code_executor.py for direct unit tests
of that sandbox.
"""
from agents.code_problem_generator.taxonomy import MAX_DIFFICULTY, MIN_DIFFICULTY, START_DIFFICULTY, TOTAL_PROBLEMS
from app.services import coding_service

CORRECT_CODE = "def add_two(a, b):\n    return a + b\n"
WRONG_CODE = "def add_two(a, b):\n    return a - b\n"
UNSAFE_CODE = "import os\ndef add_two(a, b):\n    return a + b\n"


def _submit(client, session_id, code=CORRECT_CODE):
    return client.post(f"/api/sessions/{session_id}/coding/submit", json={"code": code})


def _run(client, session_id, code=CORRECT_CODE):
    return client.post(f"/api/sessions/{session_id}/coding/run", json={"code": code})


def test_coding_generation_failure_returns_clean_502_not_a_crash(client, session_id, monkeypatch):
    async def always_fails(**kwargs):
        raise RuntimeError("simulated: mock fixture bank unavailable")

    monkeypatch.setattr(coding_service._code_problem_generator, "run", always_fails)

    response = client.post(f"/api/sessions/{session_id}/coding/start")
    assert response.status_code == 502
    assert "simulated" in response.json()["detail"]


def test_start_returns_first_problem(client, session_id):
    response = client.post(f"/api/sessions/{session_id}/coding/start")
    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == session_id
    assert body["total_problems"] == TOTAL_PROBLEMS
    assert body["current_index"] == 1
    assert body["difficulty"] == START_DIFFICULTY
    assert body["score"] == 0
    assert body["completed"] is False
    assert body["problem"] is not None
    assert body["problem"]["function_name"] == "add_two"
    # hidden/public test cases must never be exposed to the client
    assert "hidden_tests" not in body["problem"]
    assert "public_tests" not in body["problem"]


def test_double_start_is_idempotent(client, session_id, fake_code_problem_generator):
    first = client.post(f"/api/sessions/{session_id}/coding/start").json()
    second = client.post(f"/api/sessions/{session_id}/coding/start").json()
    assert first["problem"]["title"] == second["problem"]["title"]
    assert fake_code_problem_generator.calls == 1


def test_current_before_start_returns_409(client, session_id):
    response = client.get(f"/api/sessions/{session_id}/coding/current")
    assert response.status_code == 409


def test_submit_before_start_returns_409(client, session_id):
    response = _submit(client, session_id)
    assert response.status_code == 409


def test_run_before_start_returns_409(client, session_id):
    response = _run(client, session_id)
    assert response.status_code == 409


def test_result_before_completion_returns_409(client, session_id):
    client.post(f"/api/sessions/{session_id}/coding/start")
    response = client.get(f"/api/sessions/{session_id}/coding/result")
    assert response.status_code == 409


def test_session_persists_across_simulated_refresh(client, session_id):
    started = client.post(f"/api/sessions/{session_id}/coding/start").json()
    resumed = client.get(f"/api/sessions/{session_id}/coding/current").json()

    assert resumed["problem"]["title"] == started["problem"]["title"]
    assert resumed["current_index"] == started["current_index"] == 1
    assert resumed["difficulty"] == started["difficulty"]
    assert resumed["score"] == started["score"]


def test_run_endpoint_checks_sample_tests_without_scoring(client, session_id):
    client.post(f"/api/sessions/{session_id}/coding/start")
    response = _run(client, session_id, CORRECT_CODE)
    assert response.status_code == 200
    body = response.json()
    assert body["total_count"] == 1
    assert body["passed_count"] == 1
    assert body["results"][0]["passed"] is True

    # run doesn't consume a submission attempt or advance any state
    current = client.get(f"/api/sessions/{session_id}/coding/current").json()
    assert current["current_index"] == 1
    assert current["score"] == 0
    assert current["difficulty"] == START_DIFFICULTY


def test_run_with_wrong_code_reports_failure_without_raising(client, session_id):
    client.post(f"/api/sessions/{session_id}/coding/start")
    response = _run(client, session_id, WRONG_CODE)
    assert response.status_code == 200
    body = response.json()
    assert body["passed_count"] == 0
    assert body["results"][0]["passed"] is False


def test_unsafe_code_is_rejected_with_422(client, session_id):
    client.post(f"/api/sessions/{session_id}/coding/start")
    response = _submit(client, session_id, UNSAFE_CODE)
    assert response.status_code == 422


def test_correct_submission_increases_difficulty_scores_and_advances(client, session_id):
    client.post(f"/api/sessions/{session_id}/coding/start")
    body = _submit(client, session_id, CORRECT_CODE).json()

    assert body["is_correct"] is True
    assert body["passed_count"] == body["total_count"] == 3
    assert body["difficulty"] == min(START_DIFFICULTY + 1, MAX_DIFFICULTY)
    assert body["score"] == 1
    assert body["completed"] is False
    assert body["next_problem"] is not None
    # current_index tracks problems answered so far (capped at total),
    # mirroring aptitude_service's AnswerSubmitResponse.current_index.
    assert body["current_index"] == 1


def test_incorrect_submission_decreases_difficulty(client, session_id):
    client.post(f"/api/sessions/{session_id}/coding/start")
    body = _submit(client, session_id, WRONG_CODE).json()

    assert body["is_correct"] is False
    assert body["passed_count"] == 0
    assert body["difficulty"] == max(START_DIFFICULTY - 1, MIN_DIFFICULTY)
    assert body["score"] == 0


def test_full_round_is_exactly_total_problems_and_completes(client, session_id):
    client.post(f"/api/sessions/{session_id}/coding/start")

    last_body = None
    for i in range(TOTAL_PROBLEMS):
        last_body = _submit(client, session_id, CORRECT_CODE).json()
        is_last = i == TOTAL_PROBLEMS - 1
        assert last_body["completed"] is is_last
        if is_last:
            assert last_body["next_problem"] is None
        else:
            assert last_body["next_problem"] is not None

    assert last_body["score"] == TOTAL_PROBLEMS

    result = client.get(f"/api/sessions/{session_id}/coding/result")
    assert result.status_code == 200
    result_body = result.json()
    assert result_body["score"] == TOTAL_PROBLEMS
    assert len(result_body["history"]) == TOTAL_PROBLEMS
    assert result_body["percentage"] == 100.0

    # Stage transition: coding -> technical, via the existing orchestrator.
    session = client.get(f"/api/sessions/{session_id}").json()
    assert session["current_stage"] == "technical"


def test_submit_after_completion_returns_409(client, session_id):
    client.post(f"/api/sessions/{session_id}/coding/start")
    for _ in range(TOTAL_PROBLEMS):
        _submit(client, session_id, CORRECT_CODE)
    response = _submit(client, session_id, CORRECT_CODE)
    assert response.status_code == 409


def test_history_entries_have_required_timing_and_grading_fields(client, session_id):
    client.post(f"/api/sessions/{session_id}/coding/start")
    for _ in range(TOTAL_PROBLEMS):
        _submit(client, session_id, CORRECT_CODE)

    history = client.get(f"/api/sessions/{session_id}/coding/result").json()["history"]
    for entry in history:
        assert entry["presented_at"]
        assert entry["submitted_at"]
        assert entry["response_time_seconds"] >= 0
        assert entry["submitted_code"] == CORRECT_CODE
        assert entry["passed_count"] == entry["total_count"] == 3
        assert entry["is_correct"] is True
