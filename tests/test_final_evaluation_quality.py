"""
Proves the coding round's code-quality result actually reaches Final
Evaluation (not just the coding round's own API) -- both the public API
response's coding round summary, and the deterministic MockFinalEvaluatorAgent's
strengths/weaknesses (i.e. the overall recommendation genuinely has access
to it, not just a value sitting unused in the response).

Also proves MOCK_MODE runs the whole flow, including the code-quality
review, with zero Claude API calls (no CostLog rows for the quality
reviewer).
"""
from agents.code_problem_generator.taxonomy import TOTAL_PROBLEMS
from app.models.session import CostLog
from app.services import final_evaluation_service

APTITUDE_CORRECT = 0
TECHNICAL_CORRECT_ANSWER = "A stack is LIFO and a queue is FIFO."
HR_CORRECT_ANSWER = "I had a conflict, we talked to communicate, found a compromise, and managed to resolve it."
CODING_CLEAN_CODE = 'def add_two(a, b):\n    """Return the sum of a and b."""\n    return a + b\n'


def _complete_aptitude(client, session_id):
    from agents.question_generator.taxonomy import TOTAL_QUESTIONS

    client.post(f"/api/sessions/{session_id}/aptitude/start")
    for _ in range(TOTAL_QUESTIONS):
        client.post(f"/api/sessions/{session_id}/aptitude/answer", json={"selected_option": APTITUDE_CORRECT})


def _complete_coding(client, session_id, code=CODING_CLEAN_CODE):
    client.post(f"/api/sessions/{session_id}/coding/start")
    for _ in range(TOTAL_PROBLEMS):
        client.post(f"/api/sessions/{session_id}/coding/submit", json={"code": code})


def _complete_technical(client, session_id):
    from agents.technical_interviewer.taxonomy import TOTAL_QUESTIONS

    client.post(f"/api/sessions/{session_id}/technical/start")
    for _ in range(TOTAL_QUESTIONS):
        client.post(f"/api/sessions/{session_id}/technical/answer", json={"answer": TECHNICAL_CORRECT_ANSWER})


def _complete_hr(client, session_id):
    from agents.hr_interviewer.taxonomy import TOTAL_QUESTIONS

    client.post(f"/api/sessions/{session_id}/hr/start")
    for _ in range(TOTAL_QUESTIONS):
        client.post(f"/api/sessions/{session_id}/hr/answer", json={"answer": HR_CORRECT_ANSWER})


def _complete_all_rounds(client, session_id, coding_code=CODING_CLEAN_CODE):
    _complete_aptitude(client, session_id)
    _complete_coding(client, session_id, coding_code)
    _complete_technical(client, session_id)
    _complete_hr(client, session_id)


def test_final_evaluation_response_includes_coding_quality_result(client, session_id):
    _complete_all_rounds(client, session_id)
    body = client.post(f"/api/sessions/{session_id}/final-evaluation/generate").json()

    assert "average_quality_score" in body["coding"]
    assert body["coding"]["average_quality_score"] is not None
    assert 0 <= body["coding"]["average_quality_score"] <= 100
    assert body["coding"]["quality_feedback_summary"]

    # Other rounds must NOT gain a quality field -- this is a coding-only
    # extension, not a change to the shared RoundScoreSummary shape.
    for round_name in ("aptitude", "technical", "hr"):
        assert "average_quality_score" not in body[round_name]


def test_strong_code_quality_surfaces_as_a_strength(client, session_id):
    # Clean, well-structured, docstring'd code -> the mock heuristic scores
    # it highly -> the deterministic evaluator should call it out.
    _complete_all_rounds(client, session_id, coding_code=CODING_CLEAN_CODE)
    body = client.post(f"/api/sessions/{session_id}/final-evaluation/generate").json()

    assert any("quality" in s.lower() for s in body["strengths"])


def test_weak_code_quality_surfaces_as_a_weakness(client, session_id, monkeypatch):
    # Deterministically simulate a low-quality-but-functionally-correct
    # submission by replacing the quality reviewer's output directly --
    # this tests the strengths/weaknesses INTEGRATION contract (a low
    # average_quality_score must surface as a weakness distinct from
    # functional performance) without depending on the offline heuristic's
    # exact numeric tuning for any particular hand-written "messy" sample.
    from agents.base import AgentResult
    from agents.code_quality.code_quality_reviewer import QUALITY_DIMENSIONS
    from app.services import coding_service

    async def fixed_low_quality_run(**kwargs):
        return AgentResult(
            data={
                "quality_score": 20,
                "dimensions": {dim: 20 for dim in QUALITY_DIMENSIONS},
                "feedback": "Simulated low-quality review for testing.",
            },
            usage=None,
        )

    monkeypatch.setattr(coding_service._code_quality_reviewer, "run", fixed_low_quality_run)

    _complete_all_rounds(client, session_id, coding_code=CODING_CLEAN_CODE)  # functionally correct
    body = client.post(f"/api/sessions/{session_id}/final-evaluation/generate").json()

    assert body["coding"]["score"] == TOTAL_PROBLEMS  # functionally still fully correct
    assert body["coding"]["average_quality_score"] == 20.0
    assert any("quality" in w.lower() for w in body["weaknesses"])


def test_mock_mode_final_evaluation_flow_logs_no_quality_reviewer_cost(client, session_id, db_session):
    _complete_all_rounds(client, session_id)
    response = client.post(f"/api/sessions/{session_id}/final-evaluation/generate")
    assert response.status_code == 200

    # MockCodeQualityReviewerAgent.run() sets usage=None -> no CostLog row
    # for it anywhere in the session, confirming no Claude call was made.
    cost_logs = (
        db_session.query(CostLog)
        .filter(CostLog.session_id == session_id, CostLog.agent.in_(["code_quality_reviewer", "code_quality_reviewer_mock"]))
        .all()
    )
    assert all(log.agent != "code_quality_reviewer" for log in cost_logs)


def test_final_evaluation_service_uses_mock_final_evaluator_by_default():
    from agents.orchestrator.final_evaluator import MockFinalEvaluatorAgent

    assert isinstance(final_evaluation_service._final_evaluator, MockFinalEvaluatorAgent)
