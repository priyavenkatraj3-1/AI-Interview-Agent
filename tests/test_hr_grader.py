"""
Unit tests for agents/hr_interviewer/grader.py's two-phase grading:

- Phase 1 (independent, answer-key-blind): MockIndependentHRGraderAgent
  (offline) and IndependentHRGraderAgent (real, against a fake Anthropic
  client — no network call) plus the MOCK_MODE factory switch.
- Phase 2 (key-based validation): MockHRKeyValidatorAgent (offline
  keyword-overlap heuristic — the same score formula this project has
  always used for HR grading) and HRKeyValidatorAgent (real) plus the
  MOCK_MODE factory switch.

See tests/test_grader_independence.py for the tests that specifically
prove phase 1 never receives the answer key.
"""
import pytest

from agents.hr_interviewer import grader as grader_module
from agents.hr_interviewer.grader import (
    HRGradingError,
    HRKeyValidatorAgent,
    IndependentHRGraderAgent,
    MockHRKeyValidatorAgent,
    MockIndependentHRGraderAgent,
    build_hr_independent_grader,
    build_hr_key_validator,
)
from agents.hr_interviewer.taxonomy import PASS_THRESHOLD

RUBRIC_KEYWORDS = ["conflict", "communicate", "compromise", "resolve"]
QUESTION = "Describe a time you had a conflict with a teammate and how you resolved it."
MODEL_ANSWER = "A strong answer describes the conflict, communication, compromise, and resolution."

# --- Phase 1: mock (offline, key-independent) ---


@pytest.mark.asyncio
async def test_mock_independent_grader_scores_from_answer_length_only():
    agent = MockIndependentHRGraderAgent()
    result = await agent.run(
        question=QUESTION,
        candidate_answer="I had a conflict, we talked to communicate, found a compromise, and resolved it.",
    )
    assert result.data["draft_score"] >= 0
    assert result.data["draft_feedback"]
    assert result.usage is None  # no Claude call


@pytest.mark.asyncio
async def test_mock_independent_grader_needs_no_answer_key_at_all():
    agent = MockIndependentHRGraderAgent()
    result = await agent.run(question=QUESTION, candidate_answer="Some answer.")
    assert isinstance(result.data["draft_score"], int)


# --- Phase 1: real (fake Anthropic client, no network) ---


class FakeToolUseBlock:
    def __init__(self, input_data: dict):
        self.type = "tool_use"
        self.input = input_data


class FakeUsage:
    def __init__(self, input_tokens=100, output_tokens=40):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class FakeResponse:
    def __init__(self, input_data: dict, input_tokens=100, output_tokens=40):
        self.content = [FakeToolUseBlock(input_data)]
        self.usage = FakeUsage(input_tokens, output_tokens)


class FakeMessages:
    def __init__(self, responses: list[FakeResponse]):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


class FakeClient:
    def __init__(self, responses: list[FakeResponse]):
        self.messages = FakeMessages(responses)


VALID_DRAFT = {"draft_score": 70, "draft_feedback": "Reasonable independent attempt."}


@pytest.mark.asyncio
async def test_independent_grader_valid_output():
    client = FakeClient([FakeResponse(VALID_DRAFT)])
    agent = IndependentHRGraderAgent(client=client)

    result = await agent.run(
        question=QUESTION,
        candidate_answer="I had a conflict, we talked to communicate, found a compromise, and resolved it.",
    )

    assert result.data["draft_score"] == 70
    assert result.data["draft_feedback"] == VALID_DRAFT["draft_feedback"]
    assert result.usage is not None
    sent_prompt = client.messages.calls[0]["messages"][0]["content"]
    assert MODEL_ANSWER not in sent_prompt
    assert "Key qualities/elements a strong answer should mention" not in sent_prompt
    assert client.messages.calls[0]["tool_choice"] == {"type": "tool", "name": "emit_hr_independent_draft"}


@pytest.mark.asyncio
async def test_independent_grader_malformed_output_triggers_retry():
    malformed = {"draft_score": 150, "draft_feedback": "out of range"}  # > 100
    client = FakeClient([FakeResponse(malformed), FakeResponse(VALID_DRAFT)])
    agent = IndependentHRGraderAgent(client=client)

    result = await agent.run(question=QUESTION, candidate_answer="anything")

    assert len(client.messages.calls) == 2
    assert result.data["draft_score"] == 70


@pytest.mark.asyncio
async def test_independent_grader_raises_after_max_attempts():
    incomplete = {"draft_feedback": "missing score"}
    client = FakeClient([FakeResponse(incomplete) for _ in range(3)])
    agent = IndependentHRGraderAgent(client=client)

    with pytest.raises(HRGradingError):
        await agent.run(question=QUESTION, candidate_answer="anything", max_attempts=3)

    assert len(client.messages.calls) == 3


def test_independent_grader_factory_mock_mode_enabled(monkeypatch):
    monkeypatch.setattr(grader_module, "MOCK_MODE", True)
    assert isinstance(build_hr_independent_grader(), MockIndependentHRGraderAgent)


def test_independent_grader_factory_mock_mode_disabled(monkeypatch):
    monkeypatch.setattr(grader_module, "MOCK_MODE", False)
    assert isinstance(build_hr_independent_grader(), IndependentHRGraderAgent)


# --- Phase 2: mock (offline heuristic, same formula as before) ---

DRAFT_KWARGS = {"draft_score": 50, "draft_feedback": "Independent draft placeholder."}


@pytest.mark.asyncio
async def test_key_validator_mock_scores_full_marks_for_answer_covering_all_keywords():
    agent = MockHRKeyValidatorAgent()
    result = await agent.run(
        question=QUESTION,
        model_answer=MODEL_ANSWER,
        rubric_keywords=RUBRIC_KEYWORDS,
        candidate_answer="I had a conflict, we talked to communicate, found a compromise, and resolved it.",
        **DRAFT_KWARGS,
    )
    assert result.data["score"] == 100
    assert result.data["is_correct"] is True
    assert result.usage is None
    assert DRAFT_KWARGS["draft_feedback"] in result.data["feedback"]


@pytest.mark.asyncio
async def test_key_validator_mock_scores_zero_for_unrelated_answer():
    agent = MockHRKeyValidatorAgent()
    result = await agent.run(
        question=QUESTION,
        model_answer=MODEL_ANSWER,
        rubric_keywords=RUBRIC_KEYWORDS,
        candidate_answer="I don't know.",
        **DRAFT_KWARGS,
    )
    assert result.data["score"] == 0
    assert result.data["is_correct"] is False


@pytest.mark.asyncio
async def test_key_validator_mock_partial_credit_below_threshold_is_incorrect():
    agent = MockHRKeyValidatorAgent()
    result = await agent.run(
        question=QUESTION,
        model_answer=MODEL_ANSWER,
        rubric_keywords=RUBRIC_KEYWORDS,
        candidate_answer="There was a conflict once.",  # matches only "conflict" -> 25%
        **DRAFT_KWARGS,
    )
    assert result.data["score"] == 25
    assert result.data["is_correct"] is False
    assert 25 < PASS_THRESHOLD


@pytest.mark.asyncio
async def test_key_validator_mock_partial_credit_at_or_above_threshold_is_correct():
    agent = MockHRKeyValidatorAgent()
    result = await agent.run(
        question=QUESTION,
        model_answer=MODEL_ANSWER,
        rubric_keywords=RUBRIC_KEYWORDS,
        candidate_answer="I had a conflict, we talked to communicate clearly and managed to resolve it.",
        **DRAFT_KWARGS,
    )
    assert result.data["score"] == 75
    assert result.data["is_correct"] is True


@pytest.mark.asyncio
async def test_key_validator_mock_handles_empty_rubric_gracefully():
    agent = MockHRKeyValidatorAgent()
    non_empty = await agent.run(
        question=QUESTION, model_answer=MODEL_ANSWER, rubric_keywords=[], candidate_answer="anything", **DRAFT_KWARGS
    )
    assert non_empty.data["score"] == 100

    empty_answer = await agent.run(
        question=QUESTION, model_answer=MODEL_ANSWER, rubric_keywords=[], candidate_answer="   ", **DRAFT_KWARGS
    )
    assert empty_answer.data["score"] == 0


# --- Phase 2: real (fake Anthropic client, no network) ---

VALID_GRADE = {"score": 85, "feedback": "Good, specific example with a clear resolution."}


@pytest.mark.asyncio
async def test_key_validator_valid_structured_grade_output():
    client = FakeClient([FakeResponse(VALID_GRADE)])
    agent = HRKeyValidatorAgent(client=client)

    result = await agent.run(
        question=QUESTION,
        model_answer=MODEL_ANSWER,
        rubric_keywords=RUBRIC_KEYWORDS,
        candidate_answer="I had a conflict with a teammate and we communicated to resolve it.",
        **DRAFT_KWARGS,
    )

    assert result.data["score"] == 85
    assert result.data["is_correct"] is True
    assert result.data["feedback"] == VALID_GRADE["feedback"]
    assert result.usage is not None
    assert client.messages.calls[0]["tool_choice"] == {"type": "tool", "name": "emit_hr_grade"}


@pytest.mark.asyncio
async def test_key_validator_low_score_is_graded_incorrect():
    low_grade = {"score": 20, "feedback": "Too vague, no specific example given."}
    client = FakeClient([FakeResponse(low_grade)])
    agent = HRKeyValidatorAgent(client=client)

    result = await agent.run(
        question=QUESTION,
        model_answer=MODEL_ANSWER,
        rubric_keywords=RUBRIC_KEYWORDS,
        candidate_answer="Not sure.",
        **DRAFT_KWARGS,
    )
    assert result.data["is_correct"] is False


@pytest.mark.asyncio
async def test_key_validator_malformed_output_triggers_retry():
    malformed = {"score": 150, "feedback": "out of range score"}  # > 100
    client = FakeClient([FakeResponse(malformed), FakeResponse(VALID_GRADE)])
    agent = HRKeyValidatorAgent(client=client)

    result = await agent.run(
        question=QUESTION,
        model_answer=MODEL_ANSWER,
        rubric_keywords=RUBRIC_KEYWORDS,
        candidate_answer="I had a conflict with a teammate and we communicated to resolve it.",
        **DRAFT_KWARGS,
    )

    assert len(client.messages.calls) == 2
    assert result.data["score"] == 85


@pytest.mark.asyncio
async def test_key_validator_raises_after_max_attempts():
    incomplete = {"feedback": "missing score"}
    client = FakeClient([FakeResponse(incomplete) for _ in range(3)])
    agent = HRKeyValidatorAgent(client=client)

    with pytest.raises(HRGradingError):
        await agent.run(
            question=QUESTION,
            model_answer=MODEL_ANSWER,
            rubric_keywords=RUBRIC_KEYWORDS,
            candidate_answer="anything",
            max_attempts=3,
            **DRAFT_KWARGS,
        )

    assert len(client.messages.calls) == 3


def test_key_validator_factory_mock_mode_enabled(monkeypatch):
    monkeypatch.setattr(grader_module, "MOCK_MODE", True)
    assert isinstance(build_hr_key_validator(), MockHRKeyValidatorAgent)


def test_key_validator_factory_mock_mode_disabled(monkeypatch):
    monkeypatch.setattr(grader_module, "MOCK_MODE", False)
    assert isinstance(build_hr_key_validator(), HRKeyValidatorAgent)
