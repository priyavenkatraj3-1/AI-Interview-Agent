"""
Integration tests for the technical round API (start / current / answer /
result), exercised through FastAPI's TestClient against a real (temp)
SQLite DB. The technical-question generator is replaced by the
deterministic `fake_technical_interviewer` fixture (autouse, see
conftest.py) so these run offline; grading goes through the real
the real two-phase pipeline (see agents/technical_interviewer/grader.py)
whose phase-2 key validator is MockTechnicalKeyValidatorAgent (offline
keyword heuristic, selected automatically since MOCK_MODE defaults to
true) — deterministic given the fake generator's fixed rubric_keywords,
so no further mocking is needed there, mirroring how the coding round's
tests exercise the real local executor. See
tests/test_grader_independence.py for proof that phase 1 never sees the
answer key.
"""
from agents.technical_interviewer.taxonomy import MAX_DIFFICULTY, MIN_DIFFICULTY, START_DIFFICULTY, TOTAL_QUESTIONS
from app.services import technical_service

CORRECT_ANSWER = "A stack is LIFO and a queue is FIFO."
INCORRECT_ANSWER = "I have no idea."


def _answer(client, session_id, answer=CORRECT_ANSWER):
    return client.post(f"/api/sessions/{session_id}/technical/answer", json={"answer": answer})


def test_technical_generation_failure_returns_clean_502_not_a_crash(client, session_id, monkeypatch):
    async def always_fails(**kwargs):
        raise RuntimeError("simulated: mock fixture bank unavailable")

    monkeypatch.setattr(technical_service._technical_interviewer, "run", always_fails)

    response = client.post(f"/api/sessions/{session_id}/technical/start")
    assert response.status_code == 502
    assert "simulated" in response.json()["detail"]


def test_start_returns_first_question(client, session_id):
    response = client.post(f"/api/sessions/{session_id}/technical/start")
    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == session_id
    assert body["target_company"] == "TCS_NQT"
    assert body["total_questions"] == TOTAL_QUESTIONS
    assert body["current_index"] == 1
    assert body["difficulty"] == START_DIFFICULTY
    assert body["score"] == 0
    assert body["completed"] is False
    assert body["question"] is not None
    assert body["question"]["question"]
    # model_answer/rubric_keywords must never be exposed before answering
    assert "model_answer" not in body["question"]
    assert "rubric_keywords" not in body["question"]


def test_unsupported_company_returns_400(client):
    create = client.post("/api/sessions", json={"target_company": "GOOGLE"})
    assert create.status_code == 200
    sid = create.json()["id"]
    response = client.post(f"/api/sessions/{sid}/technical/start")
    assert response.status_code == 400


def test_double_start_is_idempotent(client, session_id, fake_technical_interviewer):
    first = client.post(f"/api/sessions/{session_id}/technical/start").json()
    second = client.post(f"/api/sessions/{session_id}/technical/start").json()
    assert first["question"]["question"] == second["question"]["question"]
    assert fake_technical_interviewer.calls == 1


def test_current_before_start_returns_409(client, session_id):
    response = client.get(f"/api/sessions/{session_id}/technical/current")
    assert response.status_code == 409


def test_answer_before_start_returns_409(client, session_id):
    response = _answer(client, session_id)
    assert response.status_code == 409


def test_result_before_completion_returns_409(client, session_id):
    client.post(f"/api/sessions/{session_id}/technical/start")
    response = client.get(f"/api/sessions/{session_id}/technical/result")
    assert response.status_code == 409


def test_session_persists_across_simulated_refresh(client, session_id):
    started = client.post(f"/api/sessions/{session_id}/technical/start").json()

    # Simulate a browser refresh: a fresh GET should return the SAME
    # pending question, not a newly generated one.
    resumed = client.get(f"/api/sessions/{session_id}/technical/current").json()

    assert resumed["question"]["question"] == started["question"]["question"]
    assert resumed["current_index"] == started["current_index"] == 1
    assert resumed["difficulty"] == started["difficulty"]
    assert resumed["score"] == started["score"]


def test_correct_answer_increases_score_and_difficulty(client, session_id):
    client.post(f"/api/sessions/{session_id}/technical/start")
    body = _answer(client, session_id, CORRECT_ANSWER).json()

    assert body["is_correct"] is True
    assert body["answer_score"] == 100
    assert body["feedback"]
    assert body["difficulty"] == min(START_DIFFICULTY + 1, MAX_DIFFICULTY)
    assert body["score"] == 1
    assert body["completed"] is False
    assert body["next_question"] is not None


def test_incorrect_answer_decreases_difficulty(client, session_id):
    client.post(f"/api/sessions/{session_id}/technical/start")
    body = _answer(client, session_id, INCORRECT_ANSWER).json()

    assert body["is_correct"] is False
    assert body["answer_score"] == 0
    assert body["difficulty"] == max(START_DIFFICULTY - 1, MIN_DIFFICULTY)
    assert body["score"] == 0


def test_full_round_is_exactly_total_questions_and_completes(client, session_id):
    client.post(f"/api/sessions/{session_id}/technical/start")

    last_body = None
    for i in range(TOTAL_QUESTIONS):
        last_body = _answer(client, session_id, CORRECT_ANSWER).json()
        is_last = i == TOTAL_QUESTIONS - 1
        assert last_body["completed"] is is_last
        if is_last:
            assert last_body["next_question"] is None
        else:
            assert last_body["next_question"] is not None

    assert last_body["score"] == TOTAL_QUESTIONS

    result = client.get(f"/api/sessions/{session_id}/technical/result")
    assert result.status_code == 200
    result_body = result.json()
    assert result_body["score"] == TOTAL_QUESTIONS
    assert len(result_body["history"]) == TOTAL_QUESTIONS
    assert result_body["percentage"] == 100.0

    # Stage transition: technical -> hr, via the existing orchestrator.
    session = client.get(f"/api/sessions/{session_id}").json()
    assert session["current_stage"] == "hr"


def test_answer_after_completion_returns_409(client, session_id):
    client.post(f"/api/sessions/{session_id}/technical/start")
    for _ in range(TOTAL_QUESTIONS):
        _answer(client, session_id, CORRECT_ANSWER)
    response = _answer(client, session_id, CORRECT_ANSWER)
    assert response.status_code == 409


def test_history_entries_have_required_fields(client, session_id):
    client.post(f"/api/sessions/{session_id}/technical/start")
    for _ in range(TOTAL_QUESTIONS):
        _answer(client, session_id, CORRECT_ANSWER)

    history = client.get(f"/api/sessions/{session_id}/technical/result").json()["history"]
    for entry in history:
        assert entry["presented_at"]
        assert entry["answered_at"]
        assert entry["response_time_seconds"] >= 0
        assert entry["candidate_answer"] == CORRECT_ANSWER
        assert entry["model_answer"]
        assert entry["is_correct"] is True
        assert entry["answer_score"] == 100
