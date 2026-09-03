"""
Unit tests for QuestionGeneratorAgent: structured output shape and
duplicate-prevention retry behavior, against a fake Anthropic client (no
network call — the real anthropic.messages.create() is never invoked).

Also covers MockQuestionGeneratorAgent and the MOCK_MODE-driven
build_question_generator() factory (see that module's docstring) — no
Anthropic client involved in either at all, mirroring the equivalent tests
for the coding/technical/HR generator agents.
"""
import pytest

from agents.question_generator import question_generator as qg_module
from agents.question_generator.question_generator import (
    MockQuestionGeneratorAgent,
    QuestionGenerationError,
    QuestionGeneratorAgent,
    build_question_generator,
)


class FakeToolUseBlock:
    def __init__(self, input_data: dict):
        self.type = "tool_use"
        self.input = input_data


class FakeUsage:
    def __init__(self, input_tokens=100, output_tokens=50):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class FakeResponse:
    def __init__(self, input_data: dict, input_tokens=100, output_tokens=50):
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


VALID_QUESTION = {
    "question": "What is 12 + 30?",
    "options": ["40", "42", "44", "46"],
    "correct_option": 1,
    "explanation": "12 + 30 = 42.",
}


@pytest.mark.asyncio
async def test_valid_structured_question_output():
    client = FakeClient([FakeResponse(VALID_QUESTION)])
    agent = QuestionGeneratorAgent(client=client)

    result = await agent.run(
        target_company="TCS_NQT",
        topic="quantitative",
        pattern="percentages",
        difficulty=3,
        previous_questions=[],
    )

    assert result.data["question"] == VALID_QUESTION["question"]
    assert len(result.data["options"]) == 4
    assert 0 <= result.data["correct_option"] <= 3
    assert result.data["explanation"]
    assert result.data["topic"] == "quantitative"
    assert result.data["pattern"] == "percentages"
    assert result.data["difficulty"] == 3
    assert result.usage is not None
    assert result.usage.model
    assert result.usage.cost_usd >= 0
    # Forced tool call, per the pinned SDK's structured-output mechanism.
    assert client.messages.calls[0]["tool_choice"] == {"type": "tool", "name": "emit_aptitude_question"}


@pytest.mark.asyncio
async def test_uses_requested_model_tier():
    client = FakeClient([FakeResponse(VALID_QUESTION)])
    agent = QuestionGeneratorAgent(client=client)

    result = await agent.run(
        target_company="TCS_NQT",
        topic="quantitative",
        pattern="percentages",
        difficulty=3,
        previous_questions=[],
    )

    # default_tier is CHEAP -> resolve_model should pick MODEL_CHEAP.
    from agents.config import MODEL_CHEAP

    assert result.usage.model == MODEL_CHEAP
    assert client.messages.calls[0]["model"] == MODEL_CHEAP


@pytest.mark.asyncio
async def test_duplicate_question_triggers_retry():
    duplicate = dict(VALID_QUESTION)  # identical text -> similarity 1.0
    fresh = {
        "question": "A train travels 60 km in 1.5 hours. What is its speed in km/h?",
        "options": ["30", "40", "45", "50"],
        "correct_option": 1,
        "explanation": "60 / 1.5 = 40 km/h.",
    }
    client = FakeClient([FakeResponse(duplicate), FakeResponse(fresh)])
    agent = QuestionGeneratorAgent(client=client)

    result = await agent.run(
        target_company="TCS_NQT",
        topic="quantitative",
        pattern="time_speed_and_distance",
        difficulty=3,
        previous_questions=[VALID_QUESTION["question"]],
    )

    assert len(client.messages.calls) == 2  # retried once after the duplicate
    assert result.data["question"] == fresh["question"]


@pytest.mark.asyncio
async def test_malformed_output_triggers_retry():
    malformed = {
        "question": "Broken question with only 2 options",
        "options": ["A", "B"],
        "correct_option": 0,
        "explanation": "n/a",
    }
    client = FakeClient([FakeResponse(malformed), FakeResponse(VALID_QUESTION)])
    agent = QuestionGeneratorAgent(client=client)

    result = await agent.run(
        target_company="TCS_NQT",
        topic="quantitative",
        pattern="percentages",
        difficulty=3,
        previous_questions=[],
    )

    assert len(client.messages.calls) == 2
    assert len(result.data["options"]) == 4


@pytest.mark.asyncio
async def test_gives_up_after_max_attempts_without_infinite_loop():
    # Every attempt is a duplicate of the one prior question — should stop
    # at max_attempts rather than loop forever, and still return something.
    client = FakeClient([FakeResponse(VALID_QUESTION) for _ in range(5)])
    agent = QuestionGeneratorAgent(client=client)

    result = await agent.run(
        target_company="TCS_NQT",
        topic="quantitative",
        pattern="percentages",
        difficulty=3,
        previous_questions=[VALID_QUESTION["question"]],
        max_attempts=3,
    )

    assert len(client.messages.calls) == 3
    assert result.data["question"] == VALID_QUESTION["question"]


@pytest.mark.asyncio
async def test_raises_instead_of_returning_malformed_data_after_max_attempts():
    # Every attempt is malformed (missing "question" entirely) — this must
    # never be handed back to the caller as if it were a usable question,
    # since downstream code (aptitude_service._public_question) indexes
    # required keys unconditionally.
    incomplete = {
        "options": ["A", "B", "C", "D"],
        "correct_option": 0,
        "explanation": "n/a",
    }
    client = FakeClient([FakeResponse(incomplete) for _ in range(3)])
    agent = QuestionGeneratorAgent(client=client)

    with pytest.raises(QuestionGenerationError):
        await agent.run(
            target_company="TCS_NQT",
            topic="quantitative",
            pattern="percentages",
            difficulty=3,
            previous_questions=[],
            max_attempts=3,
        )

    assert len(client.messages.calls) == 3


# --- Mock (offline, no Anthropic client) generator ---


@pytest.mark.asyncio
async def test_mock_generator_returns_a_valid_self_consistent_mcq():
    agent = MockQuestionGeneratorAgent()
    result = await agent.run(
        target_company="TCS_NQT",
        topic="quantitative",
        pattern="percentages",
        difficulty=3,
        previous_questions=[],
    )

    assert result.data["question"]
    assert len(result.data["options"]) == 4
    assert 0 <= result.data["correct_option"] <= 3
    assert result.data["explanation"]
    assert result.data["topic"] == "quantitative"
    assert result.data["pattern"] == "percentages"
    assert result.data["difficulty"] == 3
    assert result.usage is None  # no Claude call, so no cost/usage to log


@pytest.mark.asyncio
async def test_mock_generator_follows_topic_pattern_difficulty_inputs():
    agent = MockQuestionGeneratorAgent()
    result = await agent.run(
        target_company="TCS_NQT",
        topic="logical_reasoning",
        pattern="syllogism",
        difficulty=5,
        previous_questions=[],
    )

    assert result.data["topic"] == "logical_reasoning"
    assert result.data["pattern"] == "syllogism"
    assert result.data["difficulty"] == 5
    # The question text itself reflects what it was asked for, not a fixed
    # canned string independent of the inputs.
    assert "logical reasoning" in result.data["question"].lower()
    assert "syllogism" in result.data["question"].lower()


@pytest.mark.asyncio
async def test_mock_generator_correct_option_is_actually_correct():
    agent = MockQuestionGeneratorAgent()
    result = await agent.run(
        target_company="TCS_NQT", topic="quantitative", pattern="percentages", difficulty=2, previous_questions=[]
    )

    # Self-consistency: the labeled correct_option must actually match the
    # explanation's arithmetic, not just be a well-formed index.
    correct_value = result.data["options"][result.data["correct_option"]]
    assert correct_value in result.data["explanation"]


@pytest.mark.asyncio
async def test_mock_generator_varies_with_previous_questions_length():
    agent = MockQuestionGeneratorAgent()
    first = await agent.run(
        target_company="TCS_NQT", topic="quantitative", pattern="percentages", difficulty=3, previous_questions=[]
    )
    second = await agent.run(
        target_company="TCS_NQT",
        topic="quantitative",
        pattern="percentages",
        difficulty=3,
        previous_questions=[first.data["question"]],
    )
    assert first.data["question"] != second.data["question"]


# --- MOCK_MODE-driven factory ---


def test_factory_returns_mock_agent_when_mock_mode_enabled(monkeypatch):
    monkeypatch.setattr(qg_module, "MOCK_MODE", True)
    agent = build_question_generator()
    assert isinstance(agent, MockQuestionGeneratorAgent)


def test_factory_returns_real_agent_when_mock_mode_disabled(monkeypatch):
    monkeypatch.setattr(qg_module, "MOCK_MODE", False)
    agent = build_question_generator()
    assert isinstance(agent, QuestionGeneratorAgent)
