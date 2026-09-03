"""
Integration tests for the final evaluation API (generate / get), exercised
through FastAPI's TestClient against a real (temp) SQLite DB. All four
rounds are driven to completion first via their existing (already-tested)
endpoints and the autouse fake generators from conftest.py — no scoring
logic is reimplemented here: every score comes straight from each round's
own get_result(). No Anthropic API key is used anywhere in this flow
(MOCK_MODE defaults to true, selecting MockFinalEvaluatorAgent).
"""
from agents.code_problem_generator.taxonomy import TOTAL_PROBLEMS
from agents.hr_interviewer.taxonomy import TOTAL_QUESTIONS as HR_TOTAL_QUESTIONS
from agents.orchestrator.final_evaluator import MockFinalEvaluatorAgent
from agents.question_generator.taxonomy import TOTAL_QUESTIONS as APTITUDE_TOTAL_QUESTIONS
from agents.technical_interviewer.taxonomy import TOTAL_QUESTIONS as TECHNICAL_TOTAL_QUESTIONS
from app.models.session import CostLog
from app.services import final_evaluation_service

APTITUDE_CORRECT = 0
APTITUDE_INCORRECT = 1
CODING_CORRECT_CODE = "def add_two(a, b):\n    return a + b\n"
CODING_INCORRECT_CODE = "def add_two(a, b):\n    return a - b\n"
TECHNICAL_CORRECT_ANSWER = "A stack is LIFO and a queue is FIFO."
TECHNICAL_INCORRECT_ANSWER = "I have no idea."
HR_CORRECT_ANSWER = "I had a conflict, we talked to communicate, found a compromise, and managed to resolve it."
HR_INCORRECT_ANSWER = "I have no idea."


def _complete_aptitude(client, session_id, correct=True):
    client.post(f"/api/sessions/{session_id}/aptitude/start")
    selected = APTITUDE_CORRECT if correct else APTITUDE_INCORRECT
    for _ in range(APTITUDE_TOTAL_QUESTIONS):
        client.post(f"/api/sessions/{session_id}/aptitude/answer", json={"selected_option": selected})


def _complete_coding(client, session_id, correct=True):
    client.post(f"/api/sessions/{session_id}/coding/start")
    code = CODING_CORRECT_CODE if correct else CODING_INCORRECT_CODE
    for _ in range(TOTAL_PROBLEMS):
        client.post(f"/api/sessions/{session_id}/coding/submit", json={"code": code})


def _complete_technical(client, session_id, correct=True):
    client.post(f"/api/sessions/{session_id}/technical/start")
    answer = TECHNICAL_CORRECT_ANSWER if correct else TECHNICAL_INCORRECT_ANSWER
    for _ in range(TECHNICAL_TOTAL_QUESTIONS):
        client.post(f"/api/sessions/{session_id}/technical/answer", json={"answer": answer})


def _complete_hr(client, session_id, correct=True):
    client.post(f"/api/sessions/{session_id}/hr/start")
    answer = HR_CORRECT_ANSWER if correct else HR_INCORRECT_ANSWER
    for _ in range(HR_TOTAL_QUESTIONS):
        client.post(f"/api/sessions/{session_id}/hr/answer", json={"answer": answer})


def _complete_all_rounds(client, session_id, *, aptitude=True, coding=True, technical=True, hr=True):
    _complete_aptitude(client, session_id, correct=aptitude)
    _complete_coding(client, session_id, correct=coding)
    _complete_technical(client, session_id, correct=technical)
    _complete_hr(client, session_id, correct=hr)


# --- 5. Incomplete session: expected errors ---


def test_generate_before_any_round_completed_returns_409_naming_aptitude(client, session_id):
    response = client.post(f"/api/sessions/{session_id}/final-evaluation/generate")
    assert response.status_code == 409
    assert "Aptitude" in response.json()["detail"]


def test_generate_with_only_aptitude_completed_returns_409_naming_coding(client, session_id):
    _complete_aptitude(client, session_id)
    response = client.post(f"/api/sessions/{session_id}/final-evaluation/generate")
    assert response.status_code == 409
    assert "Coding" in response.json()["detail"]


def test_get_before_generate_returns_409(client, session_id):
    response = client.get(f"/api/sessions/{session_id}/final-evaluation")
    assert response.status_code == 409


def test_final_evaluation_for_nonexistent_session_returns_404(client):
    response = client.post("/api/sessions/does-not-exist/final-evaluation/generate")
    assert response.status_code == 404


# --- 1 & 2. Complete session: overall score + round-wise breakdown ---


def test_full_flow_generates_final_evaluation_with_expected_shape(client, session_id):
    _complete_all_rounds(client, session_id)

    response = client.post(f"/api/sessions/{session_id}/final-evaluation/generate")
    assert response.status_code == 200
    body = response.json()

    assert body["session_id"] == session_id
    assert body["target_company"] == "TCS_NQT"
    assert body["overall_score"] == 100.0
    assert body["aptitude"] == {
        "score": APTITUDE_TOTAL_QUESTIONS,
        "total": APTITUDE_TOTAL_QUESTIONS,
        "percentage": 100.0,
    }
    # Functional fields: exact, unchanged shape/values.
    assert body["coding"]["score"] == TOTAL_PROBLEMS
    assert body["coding"]["total"] == TOTAL_PROBLEMS
    assert body["coding"]["percentage"] == 100.0
    # Code-quality result: separate from the functional fields above, now
    # included in the coding round summary (never a separate extra round).
    assert body["coding"]["average_quality_score"] is not None
    assert 0 <= body["coding"]["average_quality_score"] <= 100
    assert body["coding"]["quality_feedback_summary"]
    assert body["technical"] == {
        "score": TECHNICAL_TOTAL_QUESTIONS,
        "total": TECHNICAL_TOTAL_QUESTIONS,
        "percentage": 100.0,
    }
    assert body["hr"] == {"score": HR_TOTAL_QUESTIONS, "total": HR_TOTAL_QUESTIONS, "percentage": 100.0}

    # Final evaluation reuses, never sets, current_stage — HR's own
    # completion already advanced it to the terminal stage.
    session = client.get(f"/api/sessions/{session_id}").json()
    assert session["current_stage"] == "completed"


def test_half_correct_yields_fifty_percent_overall_and_matching_breakdown(client, session_id):
    _complete_all_rounds(client, session_id, aptitude=True, coding=True, technical=False, hr=False)
    body = client.post(f"/api/sessions/{session_id}/final-evaluation/generate").json()

    assert body["overall_score"] == 50.0
    assert body["aptitude"]["percentage"] == 100.0
    assert body["coding"]["percentage"] == 100.0
    assert body["technical"]["percentage"] == 0.0
    assert body["hr"]["percentage"] == 0.0


# --- 3. Strengths and weaknesses generation ---


def test_all_correct_produces_strengths_and_default_weaknesses_message(client, session_id):
    _complete_all_rounds(client, session_id)
    body = client.post(f"/api/sessions/{session_id}/final-evaluation/generate").json()

    assert len(body["strengths"]) >= 1
    assert any(label in s for s in body["strengths"] for label in ("Aptitude", "Coding", "Technical", "HR"))
    assert body["weaknesses"] == ["No significant weaknesses identified."]


def test_mixed_scores_flag_weak_rounds_as_weaknesses_and_strong_rounds_as_strengths(client, session_id):
    _complete_all_rounds(client, session_id, aptitude=True, coding=True, technical=False, hr=False)
    body = client.post(f"/api/sessions/{session_id}/final-evaluation/generate").json()

    assert any("Technical" in w for w in body["weaknesses"])
    assert any("HR" in w for w in body["weaknesses"])
    assert any(label in s for s in body["strengths"] for label in ("Aptitude", "Coding"))


# --- 4. Final recommendation/threshold ---


def test_all_correct_yields_strongly_recommend(client, session_id):
    _complete_all_rounds(client, session_id)
    body = client.post(f"/api/sessions/{session_id}/final-evaluation/generate").json()
    assert body["recommendation"] == "Strongly Recommend"


def test_half_correct_yields_recommend_with_reservations(client, session_id):
    _complete_all_rounds(client, session_id, aptitude=True, coding=True, technical=False, hr=False)
    body = client.post(f"/api/sessions/{session_id}/final-evaluation/generate").json()
    assert body["recommendation"] == "Recommend with Reservations"


def test_all_incorrect_yields_do_not_recommend(client, session_id):
    _complete_all_rounds(client, session_id, aptitude=False, coding=False, technical=False, hr=False)
    body = client.post(f"/api/sessions/{session_id}/final-evaluation/generate").json()
    assert body["overall_score"] == 0.0
    assert body["recommendation"] == "Do Not Recommend"


# --- 6. MOCK_MODE works without an Anthropic API key ---


def test_final_evaluation_uses_mock_evaluator_and_logs_no_claude_cost(client, session_id, db_session):
    assert isinstance(final_evaluation_service._final_evaluator, MockFinalEvaluatorAgent)

    _complete_all_rounds(client, session_id)
    response = client.post(f"/api/sessions/{session_id}/final-evaluation/generate")
    assert response.status_code == 200

    # MockFinalEvaluatorAgent.run() sets usage=None -> no CostLog row is
    # written for this stage, confirming no Claude call was made.
    cost_logs = (
        db_session.query(CostLog)
        .filter(CostLog.session_id == session_id, CostLog.stage == "completed")
        .all()
    )
    assert cost_logs == []


# --- 7. Persisted and retrievable ---


def test_generate_is_idempotent_and_does_not_recompute(client, session_id, monkeypatch):
    _complete_all_rounds(client, session_id)

    calls = {"count": 0}
    original_run = final_evaluation_service._final_evaluator.run

    async def counting_run(**kwargs):
        calls["count"] += 1
        return await original_run(**kwargs)

    monkeypatch.setattr(final_evaluation_service._final_evaluator, "run", counting_run)

    first = client.post(f"/api/sessions/{session_id}/final-evaluation/generate")
    second = client.post(f"/api/sessions/{session_id}/final-evaluation/generate")

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
    assert calls["count"] == 1


def test_get_after_generate_returns_the_same_persisted_data(client, session_id):
    _complete_all_rounds(client, session_id)

    generated = client.post(f"/api/sessions/{session_id}/final-evaluation/generate").json()
    fetched = client.get(f"/api/sessions/{session_id}/final-evaluation").json()
    assert generated == fetched


# --- 8. Remediation plan (requirement a & b) ---


def test_remediation_plan_has_exactly_fourteen_entries(client, session_id):
    _complete_all_rounds(client, session_id)
    body = client.post(f"/api/sessions/{session_id}/final-evaluation/generate").json()

    plan = body["remediation_plan"]
    assert len(plan) == 14
    assert [entry["day"] for entry in plan] == list(range(1, 15))
    for entry in plan:
        assert entry["focus"]
        assert entry["action"]


def test_remediation_plan_differs_between_different_performance_profiles(client):
    session_a = client.post("/api/sessions", json={"target_company": "TCS_NQT"}).json()["id"]
    _complete_all_rounds(session_id=session_a, client=client, aptitude=True, coding=True, technical=False, hr=False)
    plan_a = client.post(f"/api/sessions/{session_a}/final-evaluation/generate").json()["remediation_plan"]

    session_b = client.post("/api/sessions", json={"target_company": "TCS_NQT"}).json()["id"]
    _complete_all_rounds(session_id=session_b, client=client, aptitude=False, coding=False, technical=True, hr=True)
    plan_b = client.post(f"/api/sessions/{session_b}/final-evaluation/generate").json()["remediation_plan"]

    assert plan_a != plan_b
    focuses_a = {entry["focus"] for entry in plan_a}
    focuses_b = {entry["focus"] for entry in plan_b}
    assert focuses_a != focuses_b


# --- 9. Hiring verdict (requirement c & d) ---


def test_hiring_verdict_is_a_nonempty_paragraph(client, session_id):
    _complete_all_rounds(client, session_id)
    body = client.post(f"/api/sessions/{session_id}/final-evaluation/generate").json()

    verdict = body["hiring_verdict"]
    assert isinstance(verdict, str)
    assert len(verdict.strip()) > 0
    assert verdict.count(".") >= 2


def test_hiring_verdict_differs_between_different_performance_profiles(client):
    session_a = client.post("/api/sessions", json={"target_company": "TCS_NQT"}).json()["id"]
    _complete_all_rounds(session_id=session_a, client=client)  # all correct
    verdict_a = client.post(f"/api/sessions/{session_a}/final-evaluation/generate").json()["hiring_verdict"]

    session_b = client.post("/api/sessions", json={"target_company": "TCS_NQT"}).json()["id"]
    _complete_all_rounds(
        session_id=session_b, client=client, aptitude=False, coding=False, technical=False, hr=False
    )  # all incorrect
    verdict_b = client.post(f"/api/sessions/{session_b}/final-evaluation/generate").json()["hiring_verdict"]

    assert verdict_a != verdict_b


def test_hiring_verdict_and_recommendation_field_both_present(client, session_id):
    # The existing `recommendation` field must be preserved alongside the
    # new `hiring_verdict` field -- not replaced by it.
    _complete_all_rounds(client, session_id)
    body = client.post(f"/api/sessions/{session_id}/final-evaluation/generate").json()

    assert "recommendation" in body
    assert "hiring_verdict" in body
    assert body["recommendation"] in body["hiring_verdict"]


# --- 10. Existing fields/back-compat check ---


def test_all_previously_existing_fields_are_still_present(client, session_id):
    _complete_all_rounds(client, session_id)
    body = client.post(f"/api/sessions/{session_id}/final-evaluation/generate").json()

    for field in (
        "session_id",
        "target_company",
        "overall_score",
        "aptitude",
        "coding",
        "technical",
        "hr",
        "strengths",
        "weaknesses",
        "recommendation",
        "summary",
        "generated_at",
    ):
        assert field in body, f"pre-existing field '{field}' missing from response"
