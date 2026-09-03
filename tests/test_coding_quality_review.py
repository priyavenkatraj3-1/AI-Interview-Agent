"""
Integration tests proving the code-quality review is wired correctly into
the coding round's real submit flow:

- runs AFTER functional (hidden-test) grading, as an independent step;
- never changes is_correct/passed_count/total_count;
- is persisted in StageProgress.details and surfaced via get_result();
- is present in the /coding/submit API response;
- flows into the Final Evaluation's coding round summary.

Exercised through the real FastAPI TestClient + real MockCodeQualityReviewerAgent
(MOCK_MODE defaults to true), same style as test_coding_hidden_results.py.
"""
from agents.code_quality.code_quality_reviewer import QUALITY_DIMENSIONS
from agents.code_problem_generator.taxonomy import TOTAL_PROBLEMS
from app.models.session import StageProgress

CORRECT_CODE = "def add_two(a, b):\n    return a + b\n"
WRONG_CODE = "def add_two(a, b):\n    return a - b\n"


def _submit(client, session_id, code=CORRECT_CODE):
    return client.post(f"/api/sessions/{session_id}/coding/submit", json={"code": code})


# --- Structured result present in the API response ---


def test_submit_response_contains_quality_score_and_feedback(client, session_id):
    client.post(f"/api/sessions/{session_id}/coding/start")
    response = _submit(client, session_id, CORRECT_CODE)
    assert response.status_code == 200
    body = response.json()

    assert isinstance(body["quality_score"], int)
    assert 0 <= body["quality_score"] <= 100
    assert body["quality_feedback"]
    assert set(body["quality_dimensions"].keys()) == set(QUALITY_DIMENSIONS)
    for dim in QUALITY_DIMENSIONS:
        assert 0 <= body["quality_dimensions"][dim] <= 100


# --- Quality review never changes the functional result ---


def test_quality_review_does_not_alter_hidden_test_correctness_for_correct_code(client, session_id):
    client.post(f"/api/sessions/{session_id}/coding/start")
    body = _submit(client, session_id, CORRECT_CODE).json()

    assert body["is_correct"] is True
    assert body["passed_count"] == body["total_count"]
    # A quality result is present regardless, but did not flip correctness.
    assert body["quality_score"] is not None


def test_quality_review_does_not_alter_hidden_test_correctness_for_wrong_code(client, session_id):
    client.post(f"/api/sessions/{session_id}/coding/start")
    body = _submit(client, session_id, WRONG_CODE).json()

    assert body["is_correct"] is False
    assert body["passed_count"] == 0
    # Still reviewed for quality even though it failed functionally --
    # quality review is independent of correctness.
    assert body["quality_score"] is not None
    assert 0 <= body["quality_score"] <= 100


def test_unsafe_code_never_reaches_quality_review(client, session_id):
    # Denylisted code is rejected before execution even starts (422) -- the
    # quality reviewer must never be invoked on code that never ran.
    unsafe_code = "import os\ndef add_two(a, b):\n    return a + b\n"
    client.post(f"/api/sessions/{session_id}/coding/start")
    response = _submit(client, session_id, unsafe_code)
    assert response.status_code == 422
    assert "quality_score" not in response.json()


# --- Persistence ---


def test_quality_result_is_persisted_in_stage_progress(client, session_id, db_session):
    client.post(f"/api/sessions/{session_id}/coding/start")
    _submit(client, session_id, CORRECT_CODE)

    progress = (
        db_session.query(StageProgress)
        .filter(StageProgress.session_id == session_id, StageProgress.stage == "coding")
        .one()
    )
    entry = progress.details["asked_problems"][0]
    assert isinstance(entry["quality_score"], int)
    assert entry["quality_feedback"]
    assert set(entry["quality_dimensions"].keys()) == set(QUALITY_DIMENSIONS)


def test_quality_result_appears_in_history_and_result_endpoint(client, session_id):
    client.post(f"/api/sessions/{session_id}/coding/start")
    for _ in range(TOTAL_PROBLEMS):
        _submit(client, session_id, CORRECT_CODE)

    result = client.get(f"/api/sessions/{session_id}/coding/result").json()
    assert len(result["history"]) == TOTAL_PROBLEMS
    for entry in result["history"]:
        assert isinstance(entry["quality_score"], int)
        assert entry["quality_feedback"]
        assert set(entry["quality_dimensions"].keys()) == set(QUALITY_DIMENSIONS)

    assert result["average_quality_score"] is not None
    assert 0 <= result["average_quality_score"] <= 100


# --- Ordering: quality review runs after functional grading ---


def test_quality_reviewer_is_called_after_functional_grading(client, session_id, monkeypatch):
    from app.services import coding_service

    call_order = []

    original_grader_run = coding_service._grader.run
    original_quality_run = coding_service._code_quality_reviewer.run

    async def tracking_grader_run(**kwargs):
        call_order.append("grader")
        return await original_grader_run(**kwargs)

    async def tracking_quality_run(**kwargs):
        call_order.append("quality")
        return await original_quality_run(**kwargs)

    monkeypatch.setattr(coding_service._grader, "run", tracking_grader_run)
    monkeypatch.setattr(coding_service._code_quality_reviewer, "run", tracking_quality_run)

    client.post(f"/api/sessions/{session_id}/coding/start")
    _submit(client, session_id, CORRECT_CODE)

    assert call_order == ["grader", "quality"]


def test_quality_reviewer_never_receives_hidden_test_data(client, session_id, monkeypatch):
    from app.services import coding_service

    captured = {}
    original_run = coding_service._code_quality_reviewer.run

    async def capturing_run(**kwargs):
        captured.update(kwargs)
        return await original_run(**kwargs)

    monkeypatch.setattr(coding_service._code_quality_reviewer, "run", capturing_run)

    client.post(f"/api/sessions/{session_id}/coding/start")
    _submit(client, session_id, CORRECT_CODE)

    assert set(captured.keys()) == {"code", "function_name", "title", "description"}
    for forbidden in ("hidden_tests", "public_tests", "examples"):
        assert forbidden not in captured
