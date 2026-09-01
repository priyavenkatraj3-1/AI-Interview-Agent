"""
Integration tests for the aptitude round API (start / current / answer /
result), exercised through FastAPI's TestClient against a real (temp)
SQLite DB. The QuestionGeneratorAgent is replaced by the deterministic
`fake_question_generator` fixture (autouse, see conftest.py) so these run
offline — the fake always sets correct_option=0, so submitting
selected_option=0 is "correct" and any other in-range value is "incorrect".
"""
from agents.question_generator.question_generator import QuestionGenerationError
from agents.question_generator.taxonomy import MAX_DIFFICULTY, MIN_DIFFICULTY, START_DIFFICULTY, TOTAL_QUESTIONS
from app.services import aptitude_service

CORRECT = 0
INCORRECT = 1


def _answer(client, session_id, selected_option=CORRECT):
    return client.post(f"/api/sessions/{session_id}/aptitude/answer", json={"selected_option": selected_option})


def test_persistent_generation_failure_returns_clean_502_not_a_crash(client, session_id, monkeypatch):
    # If the generator can never produce a usable question (e.g. Claude
    # keeps returning malformed tool input across every retry), the
    # service must surface a clean 502 — never a raw 500, and never a
    # 200 built from incomplete question data.
    async def always_fails(**kwargs):
        raise QuestionGenerationError("simulated: Claude never returned a valid question")

    monkeypatch.setattr(aptitude_service._question_generator, "run", always_fails)

    response = client.post(f"/api/sessions/{session_id}/aptitude/start")
    assert response.status_code == 502
    assert "simulated" in response.json()["detail"]


def test_start_returns_first_question(client, session_id):
    response = client.post(f"/api/sessions/{session_id}/aptitude/start")
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
    assert len(body["question"]["options"]) == 4
    # correct_option/explanation must never be exposed before answering
    assert "correct_option" not in body["question"]
    assert "explanation" not in body["question"]


def test_unsupported_company_returns_400(client):
    create = client.post("/api/sessions", json={"target_company": "GOOGLE"})
    assert create.status_code == 200
    sid = create.json()["id"]
    response = client.post(f"/api/sessions/{sid}/aptitude/start")
    assert response.status_code == 400


def test_double_start_is_idempotent(client, session_id, fake_question_generator):
    first = client.post(f"/api/sessions/{session_id}/aptitude/start").json()
    second = client.post(f"/api/sessions/{session_id}/aptitude/start").json()
    assert first["question"]["question"] == second["question"]["question"]
    assert fake_question_generator.calls == 1


def test_current_before_start_returns_409(client, session_id):
    response = client.get(f"/api/sessions/{session_id}/aptitude/current")
    assert response.status_code == 409


def test_answer_before_start_returns_409(client, session_id):
    response = _answer(client, session_id)
    assert response.status_code == 409


def test_result_before_completion_returns_409(client, session_id):
    client.post(f"/api/sessions/{session_id}/aptitude/start")
    response = client.get(f"/api/sessions/{session_id}/aptitude/result")
    assert response.status_code == 409


def test_out_of_range_answer_is_rejected(client, session_id):
    client.post(f"/api/sessions/{session_id}/aptitude/start")
    response = _answer(client, session_id, selected_option=4)
    assert response.status_code == 422
    response = _answer(client, session_id, selected_option=-1)
    assert response.status_code == 422


def test_session_persists_across_simulated_refresh(client, session_id):
    started = client.post(f"/api/sessions/{session_id}/aptitude/start").json()

    # Simulate a browser refresh: a fresh GET should return the SAME
    # pending question, not a newly generated one.
    resumed = client.get(f"/api/sessions/{session_id}/aptitude/current").json()

    assert resumed["question"]["question"] == started["question"]["question"]
    assert resumed["current_index"] == started["current_index"] == 1
    assert resumed["difficulty"] == started["difficulty"]
    assert resumed["score"] == started["score"]


def test_correct_answer_increases_difficulty_up_to_max(client, session_id):
    client.post(f"/api/sessions/{session_id}/aptitude/start")
    difficulty = START_DIFFICULTY
    for _ in range(MAX_DIFFICULTY + 2):  # overshoot on purpose to test the cap
        body = _answer(client, session_id, CORRECT).json()
        difficulty = min(difficulty + 1, MAX_DIFFICULTY)
        assert body["is_correct"] is True
        assert body["difficulty"] == difficulty
        assert body["difficulty"] <= MAX_DIFFICULTY


def test_incorrect_answer_decreases_difficulty_down_to_min(client, session_id):
    client.post(f"/api/sessions/{session_id}/aptitude/start")
    difficulty = START_DIFFICULTY
    for _ in range(MIN_DIFFICULTY + 3):  # overshoot on purpose to test the floor
        body = _answer(client, session_id, INCORRECT).json()
        difficulty = max(difficulty - 1, MIN_DIFFICULTY)
        assert body["is_correct"] is False
        assert body["difficulty"] == difficulty
        assert body["difficulty"] >= MIN_DIFFICULTY


def test_response_time_is_tracked_and_nonnegative(client, session_id):
    client.post(f"/api/sessions/{session_id}/aptitude/start")
    body = _answer(client, session_id, CORRECT).json()
    assert isinstance(body["response_time_seconds"], float)
    assert 0 <= body["response_time_seconds"] < 5


def test_full_round_is_exactly_fifteen_questions_and_completes(client, session_id):
    client.post(f"/api/sessions/{session_id}/aptitude/start")

    pattern = [CORRECT, CORRECT, INCORRECT, CORRECT, INCORRECT] * 3  # 15 answers
    assert len(pattern) == TOTAL_QUESTIONS
    expected_score = pattern.count(CORRECT)

    last_body = None
    for i, choice in enumerate(pattern):
        last_body = _answer(client, session_id, choice).json()
        is_last = i == TOTAL_QUESTIONS - 1
        assert last_body["completed"] is is_last
        if is_last:
            assert last_body["next_question"] is None
        else:
            assert last_body["next_question"] is not None

    assert last_body["score"] == expected_score

    result = client.get(f"/api/sessions/{session_id}/aptitude/result")
    assert result.status_code == 200
    result_body = result.json()
    assert result_body["score"] == expected_score
    assert len(result_body["history"]) == TOTAL_QUESTIONS
    assert result_body["percentage"] == round(100 * expected_score / TOTAL_QUESTIONS, 2)
    assert sum(item["is_correct"] for item in result_body["history"]) == expected_score

    # Stage transition: aptitude -> coding, via the existing orchestrator.
    session = client.get(f"/api/sessions/{session_id}").json()
    assert session["current_stage"] == "coding"


def test_answer_after_completion_returns_409(client, session_id):
    client.post(f"/api/sessions/{session_id}/aptitude/start")
    for _ in range(TOTAL_QUESTIONS):
        _answer(client, session_id, CORRECT)
    response = _answer(client, session_id, CORRECT)
    assert response.status_code == 409


def test_history_entries_have_required_timing_and_grading_fields(client, session_id):
    client.post(f"/api/sessions/{session_id}/aptitude/start")
    for _ in range(TOTAL_QUESTIONS):
        _answer(client, session_id, CORRECT)

    history = client.get(f"/api/sessions/{session_id}/aptitude/result").json()["history"]
    for entry in history:
        assert entry["presented_at"]
        assert entry["answered_at"]
        assert entry["response_time_seconds"] >= 0
        assert entry["selected_option"] == CORRECT
        assert entry["correct_option"] == CORRECT
        assert entry["is_correct"] is True
