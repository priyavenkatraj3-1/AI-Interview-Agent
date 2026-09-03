"""
End-to-end proof that the aptitude round's MOCK_MODE support
(MockQuestionGeneratorAgent + build_question_generator(), see
agents/question_generator/question_generator.py) actually works through the
real FastAPI app with no Anthropic API call.

Deliberately swaps app.services.aptitude_service._question_generator to the
real MockQuestionGeneratorAgent (not the simpler FakeQuestionGenerator from
conftest.py, which is test-only scaffolding), and blanks out
ANTHROPIC_API_KEY on the question_generator module first -- if anything on
this path ever tried to construct a real anthropic.Anthropic client or make
a network call, either constructing the client with an empty key in a
codepath that's exercised, or any accidental real HTTP call, would surface
as a failure here. A full 15-question round completing successfully with no
such error is the actual proof MOCK_MODE=true needs no live credits.
"""
from agents.question_generator import question_generator as qg_module
from agents.question_generator.question_generator import MockQuestionGeneratorAgent
from agents.question_generator.taxonomy import TOTAL_QUESTIONS
from app.services import aptitude_service

CORRECT_INDEX_FOR_MOCK = 1  # MockQuestionGeneratorAgent always sets correct_option=1


def test_full_aptitude_round_completes_via_mock_generator_with_no_api_key(client, session_id, monkeypatch):
    # No usable Anthropic credentials anywhere on this path.
    monkeypatch.setattr(qg_module, "ANTHROPIC_API_KEY", "")
    monkeypatch.setattr(aptitude_service, "_question_generator", MockQuestionGeneratorAgent())

    start = client.post(f"/api/sessions/{session_id}/aptitude/start")
    assert start.status_code == 200
    assert start.json()["question"] is not None

    last_body = None
    for i in range(TOTAL_QUESTIONS):
        response = client.post(
            f"/api/sessions/{session_id}/aptitude/answer", json={"selected_option": CORRECT_INDEX_FOR_MOCK}
        )
        assert response.status_code == 200
        last_body = response.json()
        is_last = i == TOTAL_QUESTIONS - 1
        assert last_body["completed"] is is_last

    assert last_body["completed"] is True
    assert last_body["score"] == TOTAL_QUESTIONS  # every mock question is answered correctly

    result = client.get(f"/api/sessions/{session_id}/aptitude/result")
    assert result.status_code == 200
    assert len(result.json()["history"]) == TOTAL_QUESTIONS

    session = client.get(f"/api/sessions/{session_id}").json()
    assert session["current_stage"] == "coding"


def test_build_question_generator_selects_mock_when_mock_mode_true(monkeypatch):
    monkeypatch.setattr(qg_module, "MOCK_MODE", True)
    from agents.question_generator.question_generator import build_question_generator

    agent = build_question_generator()
    assert isinstance(agent, MockQuestionGeneratorAgent)
    assert agent.name == "question_generator_mock"
