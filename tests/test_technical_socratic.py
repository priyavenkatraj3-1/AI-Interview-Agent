"""
Tests for the technical round's Socratic follow-up and Stage-2 code-probing
behavior (agents/technical_interviewer/technical_interviewer.py,
backend/app/services/technical_service.py).

These deliberately swap technical_service._technical_interviewer to the
REAL MockTechnicalInterviewerAgent (not the simpler FakeTechnicalInterviewer
from conftest.py, which ignores previous_turn/coding_context) so the actual
deterministic weak-answer/coding-context logic under test really runs,
through the real FastAPI TestClient + real coding-round flow. No network
call, no live Claude API credits required (MOCK_MODE stays true throughout).

Uses a spy (same pattern as tests/test_grader_independence.py's _GraderSpy)
to prove exactly what reaches the interviewer's run() at the real call
site, not just what a unit test of the agent class would show.
"""
import pytest

from agents.technical_interviewer.technical_interviewer import MockTechnicalInterviewerAgent
from app.services import coding_service, technical_service

CORRECT_CODING_ANSWER = "def add_two(a, b):\n    return a + b\n"
STRONG_ANSWER = "A stack is LIFO -- last in, first out -- while a queue is FIFO, first in first out."
WEAK_ANSWER = "no idea"


class _InterviewerSpy:
    """Wraps the real interviewer agent, recording every call's kwargs so
    tests can assert exactly what reached the real call site (not just the
    agent class in isolation)."""

    def __init__(self, wrapped):
        self._wrapped = wrapped
        self.name = wrapped.name
        self.calls: list[dict] = []

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        return await self._wrapped.run(**kwargs)


@pytest.fixture
def interviewer_spy(monkeypatch):
    spy = _InterviewerSpy(MockTechnicalInterviewerAgent())
    monkeypatch.setattr(technical_service, "_technical_interviewer", spy)
    return spy


def _complete_coding_round(client, session_id):
    """Runs the real coding round (2 problems, using the deterministic
    fake_code_problem_generator fixture) to completion, so a genuine Stage-2
    submission exists in StageProgress before the technical round starts."""
    client.post(f"/api/sessions/{session_id}/coding/start")
    for _ in range(2):
        response = client.post(f"/api/sessions/{session_id}/coding/submit", json={"code": CORRECT_CODING_ANSWER})
        assert response.status_code == 200


def _db_current_question(db_session, session_id) -> dict:
    from app.models.session import StageProgress

    progress = (
        db_session.query(StageProgress)
        .filter(StageProgress.session_id == session_id, StageProgress.stage == "technical")
        .one()
    )
    return progress.details["current_question"]


# --- Previous-answer context reaches the next interviewer turn ---


def test_previous_candidate_answer_reaches_the_next_interviewer_call(client, session_id, interviewer_spy):
    client.post(f"/api/sessions/{session_id}/technical/start")
    client.post(f"/api/sessions/{session_id}/technical/answer", json={"answer": STRONG_ANSWER})

    assert len(interviewer_spy.calls) == 2  # question 1 (start) + question 2 (after answering)
    second_call_kwargs = interviewer_spy.calls[1]
    assert second_call_kwargs["previous_turn"] is not None
    assert second_call_kwargs["previous_turn"]["candidate_answer"] == STRONG_ANSWER
    assert "question" in second_call_kwargs["previous_turn"]
    assert "model_answer" in second_call_kwargs["previous_turn"]


def test_first_question_has_no_previous_turn(client, session_id, interviewer_spy):
    client.post(f"/api/sessions/{session_id}/technical/start")
    assert len(interviewer_spy.calls) == 1
    assert interviewer_spy.calls[0]["previous_turn"] is None


# --- Weak answer triggers a follow-up; strong answer does not ---


def test_weak_answer_triggers_a_socratic_follow_up(client, session_id, db_session, interviewer_spy):
    first = client.post(f"/api/sessions/{session_id}/technical/start").json()
    first_topic, first_pattern = first["question"]["topic"], first["question"]["pattern"]

    response = client.post(f"/api/sessions/{session_id}/technical/answer", json={"answer": WEAK_ANSWER})
    assert response.status_code == 200

    current_question = _db_current_question(db_session, session_id)
    assert current_question["is_follow_up"] is True
    # "why?"/deeper-reasoning behavior: the follow-up explicitly asks for
    # justification, not a generic new question.
    assert "why" in current_question["question"].lower()
    # Stays on the same subject as the weak answer, rather than silently
    # switching to an unrelated topic.
    assert current_question["topic"] == first_topic
    assert current_question["pattern"] == first_pattern


def test_strong_answer_does_not_trigger_unnecessary_probing(client, session_id, db_session, interviewer_spy):
    client.post(f"/api/sessions/{session_id}/technical/start")
    response = client.post(f"/api/sessions/{session_id}/technical/answer", json={"answer": STRONG_ANSWER})
    assert response.status_code == 200

    current_question = _db_current_question(db_session, session_id)
    assert current_question["is_follow_up"] is False


def test_follow_up_does_not_chain_two_deep(client, session_id, db_session, interviewer_spy):
    # First answer is weak -> triggers a follow-up.
    client.post(f"/api/sessions/{session_id}/technical/start")
    client.post(f"/api/sessions/{session_id}/technical/answer", json={"answer": WEAK_ANSWER})
    assert _db_current_question(db_session, session_id)["is_follow_up"] is True

    # Answering the follow-up itself weakly must NOT trigger a second,
    # chained follow-up -- probe_eligible is False right after a follow-up.
    third_call_kwargs_before = len(interviewer_spy.calls)
    client.post(f"/api/sessions/{session_id}/technical/answer", json={"answer": WEAK_ANSWER})
    assert len(interviewer_spy.calls) == third_call_kwargs_before + 1
    assert interviewer_spy.calls[-1]["previous_turn"]["probe_eligible"] is False
    assert _db_current_question(db_session, session_id)["is_follow_up"] is False


def test_max_turn_limit_is_preserved_even_with_a_follow_up(client, session_id, interviewer_spy):
    from agents.technical_interviewer.taxonomy import TOTAL_QUESTIONS

    client.post(f"/api/sessions/{session_id}/technical/start")
    # First answer weak (triggers a follow-up "turn"), remaining strong.
    body = client.post(f"/api/sessions/{session_id}/technical/answer", json={"answer": WEAK_ANSWER}).json()
    turns = 1
    while not body["completed"]:
        body = client.post(f"/api/sessions/{session_id}/technical/answer", json={"answer": STRONG_ANSWER}).json()
        turns += 1
        assert turns <= TOTAL_QUESTIONS, "answered more than TOTAL_QUESTIONS turns without completing"

    assert turns == TOTAL_QUESTIONS
    assert body["completed"] is True


# --- Stage-2 submitted code reaches the interviewer, and produces a
# question specifically about it ---


def test_stage2_submitted_code_reaches_the_interviewer(client, session_id, interviewer_spy):
    _complete_coding_round(client, session_id)
    client.post(f"/api/sessions/{session_id}/technical/start")

    first_call_kwargs = interviewer_spy.calls[0]
    coding_context = first_call_kwargs["coding_context"]
    assert coding_context is not None
    assert coding_context["submitted_code"] == CORRECT_CODING_ANSWER
    assert coding_context["title"]


def test_first_technical_question_references_the_actual_submitted_code(client, session_id, interviewer_spy):
    _complete_coding_round(client, session_id)
    response = client.post(f"/api/sessions/{session_id}/technical/start")
    assert response.status_code == 200
    question_text = response.json()["question"]["question"]

    # Must reference the real submission, not a generic coding question.
    assert "add_two" in question_text
    assert "return a + b" in question_text


def test_no_coding_context_when_stage2_was_never_attempted(client, session_id, interviewer_spy):
    # No coding round run for this session -- must degrade gracefully
    # (None), not crash, and question generation must still succeed.
    response = client.post(f"/api/sessions/{session_id}/technical/start")
    assert response.status_code == 200
    assert interviewer_spy.calls[0]["coding_context"] is None


def test_coding_context_never_carries_hidden_test_data(client, session_id, interviewer_spy):
    _complete_coding_round(client, session_id)
    client.post(f"/api/sessions/{session_id}/technical/start")

    coding_context = interviewer_spy.calls[0]["coding_context"]
    assert set(coding_context.keys()) == {"title", "topic", "submitted_code"}
    for forbidden in ("hidden_tests", "expected", "args", "public_tests"):
        assert forbidden not in coding_context
