"""
Dedicated tests proving grader independence: "the grader must never see
the generator's answer key until after it has scored independently."

Three layers of proof, from strongest/most structural to most end-to-end:

1. Type-level: IndependentGradingInput (technical and HR) has no
   model_answer/rubric_keywords field at all -- constructing one with
   either raises TypeError. The answer key cannot be represented in
   phase 1's input type, not merely "unused by convention".

2. Behavioral: the phase-1 (independent) mock agents produce identical
   output for the same (question, candidate_answer) regardless of what
   answer-key-shaped data exists elsewhere -- they do not require, and
   cannot be influenced by, rubric_keywords/model_answer.

3. Data-flow (service layer): a spy on the real module-level agent
   singletons proves that, for an actual /technical/answer and /hr/answer
   request, the independent grader is called BEFORE the key validator,
   and its call never receives "model_answer" or "rubric_keywords" as
   keyword arguments -- this is the literal, production call site, not an
   isolated unit test of the agent class alone.
"""
import dataclasses

import pytest

from agents.hr_interviewer.grader import IndependentGradingInput as HRIndependentGradingInput
from agents.hr_interviewer.grader import MockIndependentHRGraderAgent
from agents.technical_interviewer.grader import IndependentGradingInput as TechnicalIndependentGradingInput
from agents.technical_interviewer.grader import MockIndependentTechnicalGraderAgent
from app.services import hr_service, technical_service

# --- 1. Type-level: the answer key cannot even be constructed into phase 1's input ---


def test_technical_independent_grading_input_has_no_answer_key_fields():
    field_names = {f.name for f in dataclasses.fields(TechnicalIndependentGradingInput)}
    assert field_names == {"question", "candidate_answer"}
    assert "model_answer" not in field_names
    assert "rubric_keywords" not in field_names


def test_technical_independent_grading_input_rejects_model_answer_kwarg():
    with pytest.raises(TypeError):
        TechnicalIndependentGradingInput(
            question="q", candidate_answer="a", model_answer="LEAKED ANSWER KEY"
        )


def test_technical_independent_grading_input_rejects_rubric_keywords_kwarg():
    with pytest.raises(TypeError):
        TechnicalIndependentGradingInput(question="q", candidate_answer="a", rubric_keywords=["leaked"])


def test_hr_independent_grading_input_has_no_answer_key_fields():
    field_names = {f.name for f in dataclasses.fields(HRIndependentGradingInput)}
    assert field_names == {"question", "candidate_answer"}


def test_hr_independent_grading_input_rejects_model_answer_kwarg():
    with pytest.raises(TypeError):
        HRIndependentGradingInput(question="q", candidate_answer="a", model_answer="LEAKED ANSWER KEY")


# --- 2. Behavioral: phase-1 output cannot be influenced by the answer key ---


@pytest.mark.asyncio
async def test_technical_independent_mock_grader_ignores_any_extra_key_shaped_kwargs():
    agent = MockIndependentTechnicalGraderAgent()
    question = "Explain the difference between a stack and a queue."
    answer = "A stack is LIFO and a queue is FIFO."

    # A caller that (incorrectly) tries to smuggle the answer key through
    # extra kwargs -- the agent's run() only ever reads question/
    # candidate_answer out of the kwargs dict, so these are silently
    # ignored rather than used, and two calls differing only in these
    # extra kwargs must produce identical results.
    result_a = await agent.run(
        question=question, candidate_answer=answer, model_answer="X", rubric_keywords=["a", "b"]
    )
    result_b = await agent.run(
        question=question, candidate_answer=answer, model_answer="COMPLETELY DIFFERENT", rubric_keywords=["z"]
    )
    assert result_a.data == result_b.data


@pytest.mark.asyncio
async def test_hr_independent_mock_grader_ignores_any_extra_key_shaped_kwargs():
    agent = MockIndependentHRGraderAgent()
    question = "Describe a time you had a conflict with a teammate."
    answer = "I had a conflict, we talked to communicate, found a compromise, and resolved it."

    result_a = await agent.run(
        question=question, candidate_answer=answer, model_answer="X", rubric_keywords=["a", "b"]
    )
    result_b = await agent.run(
        question=question, candidate_answer=answer, model_answer="COMPLETELY DIFFERENT", rubric_keywords=["z"]
    )
    assert result_a.data == result_b.data


# --- 3. Data-flow: the real service call site never gives phase 1 the key ---


class _GraderSpy:
    """Wraps a real agent's run(), recording every call's kwargs and a
    shared, ordered call-order list -- lets us assert both "what keys were
    present" and "which agent ran first" against the actual production
    call site, not just the agent class in isolation."""

    def __init__(self, wrapped, label, order_log):
        self._wrapped = wrapped
        self.name = wrapped.name
        self._label = label
        self._order_log = order_log
        self.received_kwargs: list[dict] = []

    async def run(self, **kwargs):
        self.received_kwargs.append(kwargs)
        self._order_log.append(self._label)
        return await self._wrapped.run(**kwargs)


@pytest.mark.asyncio
async def test_technical_service_never_gives_the_independent_grader_the_answer_key(client, session_id, monkeypatch):
    order_log: list[str] = []
    independent_spy = _GraderSpy(technical_service._technical_independent_grader, "independent", order_log)
    validator_spy = _GraderSpy(technical_service._technical_key_validator, "validator", order_log)

    monkeypatch.setattr(technical_service, "_technical_independent_grader", independent_spy)
    monkeypatch.setattr(technical_service, "_technical_key_validator", validator_spy)

    client.post(f"/api/sessions/{session_id}/technical/start")
    response = client.post(
        f"/api/sessions/{session_id}/technical/answer",
        json={"answer": "A stack is LIFO and a queue is FIFO."},
    )
    assert response.status_code == 200

    assert len(independent_spy.received_kwargs) == 1
    assert len(validator_spy.received_kwargs) == 1

    # The literal proof: phase 1's actual kwargs at the real call site.
    independent_kwargs = independent_spy.received_kwargs[0]
    assert "model_answer" not in independent_kwargs
    assert "rubric_keywords" not in independent_kwargs
    assert set(independent_kwargs.keys()) == {"question", "candidate_answer"}

    # Phase 2 legitimately receives the key, but only after phase 1 ran.
    validator_kwargs = validator_spy.received_kwargs[0]
    assert "model_answer" in validator_kwargs
    assert "rubric_keywords" in validator_kwargs
    assert order_log == ["independent", "validator"]


@pytest.mark.asyncio
async def test_hr_service_never_gives_the_independent_grader_the_answer_key(client, session_id, monkeypatch):
    order_log: list[str] = []
    independent_spy = _GraderSpy(hr_service._hr_independent_grader, "independent", order_log)
    validator_spy = _GraderSpy(hr_service._hr_key_validator, "validator", order_log)

    monkeypatch.setattr(hr_service, "_hr_independent_grader", independent_spy)
    monkeypatch.setattr(hr_service, "_hr_key_validator", validator_spy)

    client.post(f"/api/sessions/{session_id}/hr/start")
    response = client.post(
        f"/api/sessions/{session_id}/hr/answer",
        json={"answer": "I had a conflict, we talked it out, and resolved it."},
    )
    assert response.status_code == 200

    independent_kwargs = independent_spy.received_kwargs[0]
    assert "model_answer" not in independent_kwargs
    assert "rubric_keywords" not in independent_kwargs
    assert set(independent_kwargs.keys()) == {"question", "candidate_answer"}

    validator_kwargs = validator_spy.received_kwargs[0]
    assert "model_answer" in validator_kwargs
    assert "rubric_keywords" in validator_kwargs
    assert order_log == ["independent", "validator"]
