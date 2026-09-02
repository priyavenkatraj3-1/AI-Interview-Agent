"""
Unit tests for agents/code_problem_generator/code_problem_generator.py: the
offline MockCodeProblemGeneratorAgent (used when MOCK_MODE is enabled — the
default, since this project currently has no Anthropic API credits), the
real Claude-backed CodeProblemGeneratorAgent against a fake Anthropic
client (no network call), and the MOCK_MODE-driven factory switch.
"""
import pytest

from agents.code_problem_generator import code_problem_generator as cpg_module
from agents.code_problem_generator.code_problem_generator import (
    CodeProblemGenerationError,
    CodeProblemGeneratorAgent,
    MockCodeProblemGeneratorAgent,
    build_code_problem_generator,
)

# --- Mock (offline fixture bank) generator ---


@pytest.mark.asyncio
async def test_mock_generator_returns_a_valid_fixture_for_known_topic_pattern():
    agent = MockCodeProblemGeneratorAgent()
    result = await agent.run(topic="arrays", pattern="two_sum", difficulty=3, previous_titles=[])

    assert result.data["function_name"] == "two_sum"
    assert len(result.data["hidden_tests"]) >= 3
    assert len(result.data["public_tests"]) >= 1
    assert result.data["topic"] == "arrays"
    assert result.data["pattern"] == "two_sum"
    assert result.data["difficulty"] == 3
    assert result.usage is None  # no Claude call


@pytest.mark.asyncio
async def test_mock_generator_raises_for_unknown_topic_pattern():
    agent = MockCodeProblemGeneratorAgent()
    with pytest.raises(CodeProblemGenerationError):
        await agent.run(topic="unknown_topic", pattern="unknown_pattern", difficulty=3, previous_titles=[])


@pytest.mark.asyncio
async def test_mock_generator_fixtures_are_independent_copies():
    agent = MockCodeProblemGeneratorAgent()
    first = await agent.run(topic="strings", pattern="reverse_string", difficulty=1, previous_titles=[])
    first.data["title"] = "mutated"
    second = await agent.run(topic="strings", pattern="reverse_string", difficulty=1, previous_titles=[])
    assert second.data["title"] != "mutated"


@pytest.mark.asyncio
async def test_mock_generator_covers_every_taxonomy_topic_pattern():
    from agents.code_problem_generator.taxonomy import CODING_TAXONOMY

    agent = MockCodeProblemGeneratorAgent()
    for topic, patterns in CODING_TAXONOMY.items():
        for pattern in patterns:
            result = await agent.run(topic=topic, pattern=pattern, difficulty=3, previous_titles=[])
            assert result.data["function_name"].isidentifier()


# --- Real Claude-backed generator (fake client, no network) ---


class FakeToolUseBlock:
    def __init__(self, input_data: dict):
        self.type = "tool_use"
        self.input = input_data


class FakeUsage:
    def __init__(self, input_tokens=120, output_tokens=80):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class FakeResponse:
    def __init__(self, input_data: dict, input_tokens=120, output_tokens=80):
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


VALID_PROBLEM = {
    "title": "Sum of Two Numbers",
    "description": "Return a + b.",
    "constraints": "-100 <= a, b <= 100",
    "function_name": "add_two",
    "starter_code": "def add_two(a, b):\n    pass\n",
    "examples": [{"input": "a = 2, b = 3", "output": "5"}],
    "public_tests": [{"args": [2, 3], "expected": 5}],
    "hidden_tests": [
        {"args": [2, 3], "expected": 5},
        {"args": [-1, 1], "expected": 0},
        {"args": [10, 20], "expected": 30},
    ],
}


@pytest.mark.asyncio
async def test_valid_structured_problem_output():
    client = FakeClient([FakeResponse(VALID_PROBLEM)])
    agent = CodeProblemGeneratorAgent(client=client)

    result = await agent.run(topic="arrays", pattern="two_sum", difficulty=3, previous_titles=[])

    assert result.data["function_name"] == "add_two"
    assert len(result.data["hidden_tests"]) == 3
    assert result.data["topic"] == "arrays"
    assert result.data["difficulty"] == 3
    assert result.usage is not None
    assert result.usage.cost_usd >= 0
    assert client.messages.calls[0]["tool_choice"] == {"type": "tool", "name": "emit_coding_problem"}


@pytest.mark.asyncio
async def test_malformed_output_triggers_retry():
    malformed = dict(VALID_PROBLEM)
    malformed["hidden_tests"] = []  # fewer than the required minimum of 3
    client = FakeClient([FakeResponse(malformed), FakeResponse(VALID_PROBLEM)])
    agent = CodeProblemGeneratorAgent(client=client)

    result = await agent.run(topic="arrays", pattern="two_sum", difficulty=3, previous_titles=[])

    assert len(client.messages.calls) == 2
    assert len(result.data["hidden_tests"]) == 3


@pytest.mark.asyncio
async def test_raises_instead_of_returning_malformed_data_after_max_attempts():
    incomplete = {"description": "missing everything else"}
    client = FakeClient([FakeResponse(incomplete) for _ in range(3)])
    agent = CodeProblemGeneratorAgent(client=client)

    with pytest.raises(CodeProblemGenerationError):
        await agent.run(
            topic="arrays", pattern="two_sum", difficulty=3, previous_titles=[], max_attempts=3
        )

    assert len(client.messages.calls) == 3


@pytest.mark.asyncio
async def test_duplicate_title_triggers_retry():
    duplicate = dict(VALID_PROBLEM)
    fresh = dict(VALID_PROBLEM)
    fresh["title"] = "Completely Different Problem About Graph Traversal"
    client = FakeClient([FakeResponse(duplicate), FakeResponse(fresh)])
    agent = CodeProblemGeneratorAgent(client=client)

    result = await agent.run(
        topic="arrays", pattern="two_sum", difficulty=3, previous_titles=[VALID_PROBLEM["title"]]
    )

    assert len(client.messages.calls) == 2
    assert result.data["title"] == fresh["title"]


# --- MOCK_MODE-driven factory ---


def test_factory_returns_mock_agent_when_mock_mode_enabled(monkeypatch):
    monkeypatch.setattr(cpg_module, "MOCK_MODE", True)
    agent = build_code_problem_generator()
    assert isinstance(agent, MockCodeProblemGeneratorAgent)


def test_factory_returns_real_agent_when_mock_mode_disabled(monkeypatch):
    monkeypatch.setattr(cpg_module, "MOCK_MODE", False)
    agent = build_code_problem_generator()
    assert isinstance(agent, CodeProblemGeneratorAgent)
