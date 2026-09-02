"""
Unit tests for agents/technical_interviewer/technical_interviewer.py: the
offline MockTechnicalInterviewerAgent (used when MOCK_MODE is enabled — the
default, since this project currently has no Anthropic API credits), the
real Claude-backed TechnicalInterviewerAgent against a fake Anthropic
client (no network call), and the MOCK_MODE-driven factory switch.
"""
import pytest

from agents.technical_interviewer import technical_interviewer as ti_module
from agents.technical_interviewer.technical_interviewer import (
    MockTechnicalInterviewerAgent,
    TechnicalInterviewerAgent,
    TechnicalQuestionGenerationError,
    build_technical_interviewer,
)

# --- Mock (offline fixture bank) generator ---


@pytest.mark.asyncio
async def test_mock_generator_returns_a_valid_fixture_for_known_topic_pattern():
    agent = MockTechnicalInterviewerAgent()
    result = await agent.run(
        target_company="TCS_NQT",
        topic="data_structures",
        pattern="stacks_and_queues",
        difficulty=3,
        previous_questions=[],
    )

    assert result.data["question"]
    assert result.data["model_answer"]
    assert len(result.data["rubric_keywords"]) >= 3
    assert result.data["topic"] == "data_structures"
    assert result.data["pattern"] == "stacks_and_queues"
    assert result.data["difficulty"] == 3
    assert result.usage is None  # no Claude call


@pytest.mark.asyncio
async def test_mock_generator_raises_for_unknown_topic_pattern():
    agent = MockTechnicalInterviewerAgent()
    with pytest.raises(TechnicalQuestionGenerationError):
        await agent.run(
            target_company="TCS_NQT",
            topic="unknown_topic",
            pattern="unknown_pattern",
            difficulty=3,
            previous_questions=[],
        )


@pytest.mark.asyncio
async def test_mock_generator_fixtures_are_independent_copies():
    agent = MockTechnicalInterviewerAgent()
    first = await agent.run(
        target_company="TCS_NQT",
        topic="oop",
        pattern="encapsulation_and_abstraction",
        difficulty=1,
        previous_questions=[],
    )
    first.data["question"] = "mutated"
    second = await agent.run(
        target_company="TCS_NQT",
        topic="oop",
        pattern="encapsulation_and_abstraction",
        difficulty=1,
        previous_questions=[],
    )
    assert second.data["question"] != "mutated"


@pytest.mark.asyncio
async def test_mock_generator_covers_every_taxonomy_topic_pattern():
    from agents.technical_interviewer.taxonomy import TECHNICAL_TAXONOMY

    agent = MockTechnicalInterviewerAgent()
    for topic, patterns in TECHNICAL_TAXONOMY.items():
        for pattern in patterns:
            result = await agent.run(
                target_company="TCS_NQT", topic=topic, pattern=pattern, difficulty=3, previous_questions=[]
            )
            assert result.data["question"]
            assert len(result.data["rubric_keywords"]) >= 3


# --- Real Claude-backed generator (fake client, no network) ---


class FakeToolUseBlock:
    def __init__(self, input_data: dict):
        self.type = "tool_use"
        self.input = input_data


class FakeUsage:
    def __init__(self, input_tokens=150, output_tokens=90):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class FakeResponse:
    def __init__(self, input_data: dict, input_tokens=150, output_tokens=90):
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
    "question": "What is the difference between a stack and a queue?",
    "model_answer": "A stack is LIFO; a queue is FIFO.",
    "rubric_keywords": ["stack", "queue", "lifo", "fifo"],
}


@pytest.mark.asyncio
async def test_valid_structured_question_output():
    client = FakeClient([FakeResponse(VALID_QUESTION)])
    agent = TechnicalInterviewerAgent(client=client)

    result = await agent.run(
        target_company="TCS_NQT",
        topic="data_structures",
        pattern="stacks_and_queues",
        difficulty=3,
        previous_questions=[],
    )

    assert result.data["question"] == VALID_QUESTION["question"]
    assert len(result.data["rubric_keywords"]) == 4
    assert result.data["topic"] == "data_structures"
    assert result.data["difficulty"] == 3
    assert result.usage is not None
    assert result.usage.cost_usd >= 0
    assert client.messages.calls[0]["tool_choice"] == {"type": "tool", "name": "emit_technical_question"}


@pytest.mark.asyncio
async def test_uses_requested_model_tier():
    client = FakeClient([FakeResponse(VALID_QUESTION)])
    agent = TechnicalInterviewerAgent(client=client)

    result = await agent.run(
        target_company="TCS_NQT",
        topic="data_structures",
        pattern="stacks_and_queues",
        difficulty=3,
        previous_questions=[],
    )

    from agents.config import MODEL_STRONG

    # default_tier is STRONG -> resolve_model should pick MODEL_STRONG.
    assert result.usage.model == MODEL_STRONG
    assert client.messages.calls[0]["model"] == MODEL_STRONG


@pytest.mark.asyncio
async def test_duplicate_question_triggers_retry():
    duplicate = dict(VALID_QUESTION)
    fresh = {
        "question": "What is the time complexity of binary search, and what precondition does it require?",
        "model_answer": "O(log n) since the search space halves each step; the array must be sorted.",
        "rubric_keywords": ["binary search", "o(log n)", "sorted"],
    }
    client = FakeClient([FakeResponse(duplicate), FakeResponse(fresh)])
    agent = TechnicalInterviewerAgent(client=client)

    result = await agent.run(
        target_company="TCS_NQT",
        topic="algorithms",
        pattern="sorting_and_searching",
        difficulty=3,
        previous_questions=[VALID_QUESTION["question"]],
    )

    assert len(client.messages.calls) == 2  # retried once after the duplicate
    assert result.data["question"] == fresh["question"]


@pytest.mark.asyncio
async def test_malformed_output_triggers_retry():
    malformed = {"question": "Broken question", "model_answer": "", "rubric_keywords": ["a", "b"]}
    client = FakeClient([FakeResponse(malformed), FakeResponse(VALID_QUESTION)])
    agent = TechnicalInterviewerAgent(client=client)

    result = await agent.run(
        target_company="TCS_NQT",
        topic="data_structures",
        pattern="stacks_and_queues",
        difficulty=3,
        previous_questions=[],
    )

    assert len(client.messages.calls) == 2
    assert result.data["question"] == VALID_QUESTION["question"]


@pytest.mark.asyncio
async def test_raises_instead_of_returning_malformed_data_after_max_attempts():
    incomplete = {"question": "", "model_answer": "", "rubric_keywords": []}
    client = FakeClient([FakeResponse(incomplete) for _ in range(3)])
    agent = TechnicalInterviewerAgent(client=client)

    with pytest.raises(TechnicalQuestionGenerationError):
        await agent.run(
            target_company="TCS_NQT",
            topic="data_structures",
            pattern="stacks_and_queues",
            difficulty=3,
            previous_questions=[],
            max_attempts=3,
        )

    assert len(client.messages.calls) == 3


# --- MOCK_MODE-driven factory ---


def test_factory_returns_mock_agent_when_mock_mode_enabled(monkeypatch):
    monkeypatch.setattr(ti_module, "MOCK_MODE", True)
    agent = build_technical_interviewer()
    assert isinstance(agent, MockTechnicalInterviewerAgent)


def test_factory_returns_real_agent_when_mock_mode_disabled(monkeypatch):
    monkeypatch.setattr(ti_module, "MOCK_MODE", False)
    agent = build_technical_interviewer()
    assert isinstance(agent, TechnicalInterviewerAgent)
