"""
Unit tests for agents/code_quality/code_quality_reviewer.py: the offline
MockCodeQualityReviewerAgent (used when MOCK_MODE is enabled — the default),
the real Claude-backed CodeQualityReviewerAgent against a fake Anthropic
client (no network call), and the MOCK_MODE-driven factory switch.
"""
import pytest

from agents.code_quality import code_quality_reviewer as cqr_module
from agents.code_quality.code_quality_reviewer import (
    QUALITY_DIMENSIONS,
    CodeQualityReviewError,
    CodeQualityReviewerAgent,
    MockCodeQualityReviewerAgent,
    build_code_quality_reviewer,
)

CLEAN_CODE = 'def add_two(a, b):\n    """Return the sum of a and b."""\n    return a + b\n'
MESSY_CODE = (
    "def add_two(a, b):\n"
    "    temp = a\n"
    "    data = b\n"
    "    if temp > 0:\n"
    "        return temp + data\n"
    "    if temp > 0:\n"
    "        return temp + data\n"
)


# --- Mock (offline fixture-free heuristic) reviewer ---


@pytest.mark.asyncio
async def test_mock_reviewer_returns_a_complete_structured_result():
    agent = MockCodeQualityReviewerAgent()
    result = await agent.run(code=CLEAN_CODE, function_name="add_two")

    assert isinstance(result.data["quality_score"], int)
    assert 0 <= result.data["quality_score"] <= 100
    assert set(result.data["dimensions"].keys()) == set(QUALITY_DIMENSIONS)
    for dim in QUALITY_DIMENSIONS:
        assert isinstance(result.data["dimensions"][dim], int)
        assert 0 <= result.data["dimensions"][dim] <= 100
    assert result.data["feedback"]
    assert result.usage is None  # no Claude call


@pytest.mark.asyncio
async def test_mock_reviewer_never_raises_on_empty_or_broken_code():
    agent = MockCodeQualityReviewerAgent()
    for code in ("", "not: valid python(((", "   \n\n  "):
        result = await agent.run(code=code, function_name="add_two")
        assert 0 <= result.data["quality_score"] <= 100


@pytest.mark.asyncio
async def test_mock_reviewer_scores_cleaner_code_at_least_as_well_as_messy_duplicated_code():
    agent = MockCodeQualityReviewerAgent()
    clean = await agent.run(code=CLEAN_CODE, function_name="add_two")
    messy = await agent.run(code=MESSY_CODE, function_name="add_two")

    assert clean.data["quality_score"] >= messy.data["quality_score"]
    assert clean.data["dimensions"]["duplication"] > messy.data["dimensions"]["duplication"]


@pytest.mark.asyncio
async def test_mock_reviewer_is_deterministic():
    agent = MockCodeQualityReviewerAgent()
    first = await agent.run(code=CLEAN_CODE, function_name="add_two")
    second = await agent.run(code=CLEAN_CODE, function_name="add_two")
    assert first.data == second.data


# --- Real Claude-backed reviewer (fake client, no network) ---


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


VALID_REVIEW = {
    "quality_score": 82,
    "dimensions": {dim: 80 for dim in QUALITY_DIMENSIONS},
    "feedback": "Clear and well-structured; consider a docstring.",
}


@pytest.mark.asyncio
async def test_real_reviewer_valid_structured_output():
    client = FakeClient([FakeResponse(VALID_REVIEW)])
    agent = CodeQualityReviewerAgent(client=client)

    result = await agent.run(
        code=CLEAN_CODE, function_name="add_two", title="Two Sum", description="Return a + b."
    )

    assert result.data["quality_score"] == 82
    assert result.data["dimensions"] == VALID_REVIEW["dimensions"]
    assert result.data["feedback"] == VALID_REVIEW["feedback"]
    assert result.usage is not None
    assert result.usage.cost_usd >= 0
    assert client.messages.calls[0]["tool_choice"] == {"type": "tool", "name": "emit_code_quality_review"}


@pytest.mark.asyncio
async def test_real_reviewer_missing_dimension_triggers_retry():
    incomplete_dims = {dim: 80 for dim in QUALITY_DIMENSIONS if dim != "maintainability"}
    malformed = {"quality_score": 80, "dimensions": incomplete_dims, "feedback": "n/a"}
    client = FakeClient([FakeResponse(malformed), FakeResponse(VALID_REVIEW)])
    agent = CodeQualityReviewerAgent(client=client)

    result = await agent.run(code=CLEAN_CODE, function_name="add_two", title="t", description="d")

    assert len(client.messages.calls) == 2
    assert result.data["quality_score"] == 82


@pytest.mark.asyncio
async def test_real_reviewer_raises_after_max_attempts_with_persistently_malformed_output():
    malformed = {"quality_score": 200, "dimensions": {}, "feedback": ""}  # score out of range too
    client = FakeClient([FakeResponse(malformed) for _ in range(3)])
    agent = CodeQualityReviewerAgent(client=client)

    with pytest.raises(CodeQualityReviewError):
        await agent.run(code=CLEAN_CODE, function_name="add_two", title="t", description="d", max_attempts=3)

    assert len(client.messages.calls) == 3


# --- MOCK_MODE-driven factory ---


def test_factory_returns_mock_reviewer_when_mock_mode_enabled(monkeypatch):
    monkeypatch.setattr(cqr_module, "MOCK_MODE", True)
    agent = build_code_quality_reviewer()
    assert isinstance(agent, MockCodeQualityReviewerAgent)


def test_factory_returns_real_reviewer_when_mock_mode_disabled(monkeypatch):
    monkeypatch.setattr(cqr_module, "MOCK_MODE", False)
    agent = build_code_quality_reviewer()
    assert isinstance(agent, CodeQualityReviewerAgent)
