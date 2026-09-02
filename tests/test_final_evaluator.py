"""
Unit tests for agents/orchestrator/final_evaluator.py: the deterministic
MockFinalEvaluatorAgent (used when MOCK_MODE is enabled — the default,
since this project currently has no Anthropic API credits) and the
MOCK_MODE-driven factory switch. The real Claude-backed FinalEvaluatorAgent
follows the same retry/malformed-output shape already covered for every
other generation agent (see e.g. test_technical_interviewer.py), so this
focuses on what's new here: the mock evaluator's threshold logic.
"""
import pytest

from agents.orchestrator import final_evaluator as fe_module
from agents.orchestrator.final_evaluator import (
    FinalEvaluationError,
    FinalEvaluatorAgent,
    MockFinalEvaluatorAgent,
    RECOMMEND_THRESHOLD,
    RECOMMEND_WITH_RESERVATIONS_THRESHOLD,
    REMEDIATION_PLAN_DAYS,
    STRONGLY_RECOMMEND_THRESHOLD,
    build_final_evaluator,
)


def _round(score, total, percentage, topic_breakdown=None):
    return {
        "score": score,
        "total": total,
        "percentage": percentage,
        "topic_breakdown": topic_breakdown or {},
    }


ALL_STRONG_ROUNDS = {
    "aptitude": _round(15, 15, 100.0, {"quantitative": {"correct": 15, "total": 15}}),
    "coding": _round(2, 2, 100.0, {"arrays": {"correct": 2, "total": 2}}),
    "technical": _round(5, 5, 100.0, {"data_structures": {"correct": 5, "total": 5}}),
    "hr": _round(5, 5, 100.0, {"teamwork": {"correct": 5, "total": 5}}),
}

ALL_WEAK_ROUNDS = {
    "aptitude": _round(0, 15, 0.0, {"quantitative": {"correct": 0, "total": 15}}),
    "coding": _round(0, 2, 0.0, {"arrays": {"correct": 0, "total": 2}}),
    "technical": _round(0, 5, 0.0, {"data_structures": {"correct": 0, "total": 5}}),
    "hr": _round(0, 5, 0.0, {"teamwork": {"correct": 0, "total": 5}}),
}


@pytest.mark.asyncio
async def test_mock_evaluator_flags_strong_rounds_and_topics_as_strengths():
    agent = MockFinalEvaluatorAgent()
    result = await agent.run(overall_score=100.0, rounds=ALL_STRONG_ROUNDS)

    assert result.usage is None  # no Claude call
    assert any("Aptitude" in s for s in result.data["strengths"])
    assert any("quantitative" in s for s in result.data["strengths"])
    assert result.data["weaknesses"] == ["No significant weaknesses identified."]
    assert result.data["recommendation"] == "Strongly Recommend"
    assert result.data["summary"]


@pytest.mark.asyncio
async def test_mock_evaluator_flags_weak_rounds_and_topics_as_weaknesses():
    agent = MockFinalEvaluatorAgent()
    result = await agent.run(overall_score=0.0, rounds=ALL_WEAK_ROUNDS)

    assert any("HR" in w for w in result.data["weaknesses"])
    assert any("teamwork" in w for w in result.data["weaknesses"])
    assert result.data["strengths"] == [
        "No standout strengths identified; performance was consistent but unremarkable across rounds."
    ]
    assert result.data["recommendation"] == "Do Not Recommend"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overall_score,expected",
    [
        (100.0, "Strongly Recommend"),
        (STRONGLY_RECOMMEND_THRESHOLD, "Strongly Recommend"),
        (STRONGLY_RECOMMEND_THRESHOLD - 0.01, "Recommend"),
        (RECOMMEND_THRESHOLD, "Recommend"),
        (RECOMMEND_THRESHOLD - 0.01, "Recommend with Reservations"),
        (RECOMMEND_WITH_RESERVATIONS_THRESHOLD, "Recommend with Reservations"),
        (RECOMMEND_WITH_RESERVATIONS_THRESHOLD - 0.01, "Do Not Recommend"),
        (0.0, "Do Not Recommend"),
    ],
)
async def test_mock_evaluator_recommendation_thresholds(overall_score, expected):
    agent = MockFinalEvaluatorAgent()
    result = await agent.run(overall_score=overall_score, rounds=ALL_STRONG_ROUNDS)
    assert result.data["recommendation"] == expected


@pytest.mark.asyncio
async def test_mock_evaluator_handles_empty_topic_breakdown_without_error():
    rounds = {
        "aptitude": _round(8, 15, 53.33, {}),
        "coding": _round(1, 2, 50.0, {}),
        "technical": _round(3, 5, 60.0, {}),
        "hr": _round(3, 5, 60.0, {}),
    }
    agent = MockFinalEvaluatorAgent()
    result = await agent.run(overall_score=55.83, rounds=rounds)
    assert result.data["recommendation"] == "Recommend with Reservations"


# --- Remediation plan (requirement a & b) ---


@pytest.mark.asyncio
async def test_remediation_plan_has_exactly_fourteen_entries_for_weak_profile():
    agent = MockFinalEvaluatorAgent()
    result = await agent.run(overall_score=0.0, rounds=ALL_WEAK_ROUNDS)
    plan = result.data["remediation_plan"]
    assert len(plan) == 14
    assert len(plan) == REMEDIATION_PLAN_DAYS
    assert [entry["day"] for entry in plan] == list(range(1, 15))
    for entry in plan:
        assert entry["focus"]
        assert entry["action"]


@pytest.mark.asyncio
async def test_remediation_plan_has_exactly_fourteen_entries_for_strong_profile():
    agent = MockFinalEvaluatorAgent()
    result = await agent.run(overall_score=100.0, rounds=ALL_STRONG_ROUNDS)
    plan = result.data["remediation_plan"]
    assert len(plan) == 14
    for entry in plan:
        assert entry["focus"]
        assert entry["action"]


@pytest.mark.asyncio
async def test_remediation_plan_content_differs_between_different_weak_profiles():
    # Two candidates with different weak areas must get different plans —
    # never a generic identical plan.
    weak_in_technical_and_hr = {
        "aptitude": _round(15, 15, 100.0, {"quantitative": {"correct": 15, "total": 15}}),
        "coding": _round(2, 2, 100.0, {"arrays": {"correct": 2, "total": 2}}),
        "technical": _round(0, 5, 0.0, {"data_structures": {"correct": 0, "total": 5}}),
        "hr": _round(0, 5, 0.0, {"teamwork": {"correct": 0, "total": 5}}),
    }
    weak_in_aptitude_and_coding = {
        "aptitude": _round(0, 15, 0.0, {"quantitative": {"correct": 0, "total": 15}}),
        "coding": _round(0, 2, 0.0, {"arrays": {"correct": 0, "total": 2}}),
        "technical": _round(5, 5, 100.0, {"data_structures": {"correct": 5, "total": 5}}),
        "hr": _round(5, 5, 100.0, {"teamwork": {"correct": 5, "total": 5}}),
    }

    agent = MockFinalEvaluatorAgent()
    plan_a = (await agent.run(overall_score=50.0, rounds=weak_in_technical_and_hr)).data["remediation_plan"]
    plan_b = (await agent.run(overall_score=50.0, rounds=weak_in_aptitude_and_coding)).data["remediation_plan"]

    focuses_a = {entry["focus"] for entry in plan_a}
    focuses_b = {entry["focus"] for entry in plan_b}
    assert focuses_a != focuses_b
    assert any("technical" in f.lower() or "teamwork" in f.lower() or "hr" in f.lower() for f in focuses_a)
    assert any("aptitude" in f.lower() or "quantitative" in f.lower() or "coding" in f.lower() or "arrays" in f.lower() for f in focuses_b)


@pytest.mark.asyncio
async def test_remediation_plan_for_strong_profile_is_not_identical_to_weak_profile_plan():
    agent = MockFinalEvaluatorAgent()
    weak_plan = (await agent.run(overall_score=0.0, rounds=ALL_WEAK_ROUNDS)).data["remediation_plan"]
    strong_plan = (await agent.run(overall_score=100.0, rounds=ALL_STRONG_ROUNDS)).data["remediation_plan"]
    assert weak_plan != strong_plan


# --- Hiring verdict (requirement c & d) ---


@pytest.mark.asyncio
async def test_hiring_verdict_is_a_nonempty_paragraph():
    agent = MockFinalEvaluatorAgent()
    result = await agent.run(overall_score=52.5, rounds=ALL_WEAK_ROUNDS)
    verdict = result.data["hiring_verdict"]
    assert isinstance(verdict, str)
    assert len(verdict.strip()) > 0
    assert verdict.count(".") >= 2  # more than one sentence -> a paragraph, not a one-liner


@pytest.mark.asyncio
async def test_hiring_verdict_reflects_actual_strengths_and_weaknesses():
    mixed_rounds = {
        "aptitude": _round(9, 15, 60.0, {"quantitative": {"correct": 3, "total": 5}}),
        "coding": _round(2, 2, 100.0, {"arrays": {"correct": 2, "total": 2}}),
        "technical": _round(0, 5, 0.0, {"algorithms": {"correct": 0, "total": 5}}),
        "hr": _round(3, 5, 60.0, {"teamwork": {"correct": 3, "total": 5}}),
    }
    agent = MockFinalEvaluatorAgent()
    result = await agent.run(overall_score=55.0, rounds=mixed_rounds)
    verdict = result.data["hiring_verdict"]

    # References the strongest identified item (Coding/arrays, 100%)...
    assert "arrays" in verdict or "Coding" in verdict
    # ...and the weakest identified item (Technical/algorithms, 0%).
    assert "algorithms" in verdict or "Technical" in verdict
    assert result.data["recommendation"] in verdict


@pytest.mark.asyncio
async def test_hiring_verdict_differs_between_different_performance_profiles():
    agent = MockFinalEvaluatorAgent()
    strong_verdict = (await agent.run(overall_score=100.0, rounds=ALL_STRONG_ROUNDS)).data["hiring_verdict"]
    weak_verdict = (await agent.run(overall_score=0.0, rounds=ALL_WEAK_ROUNDS)).data["hiring_verdict"]
    assert strong_verdict != weak_verdict


# --- Real Claude-backed agent (fake client, no network): schema enforcement ---


class FakeToolUseBlock:
    def __init__(self, input_data: dict):
        self.type = "tool_use"
        self.input = input_data


class FakeUsage:
    def __init__(self, input_tokens=200, output_tokens=150):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class FakeResponse:
    def __init__(self, input_data: dict, input_tokens=200, output_tokens=150):
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


def _valid_remediation_plan() -> list[dict]:
    return [{"day": d, "focus": f"Focus {d}", "action": f"Action {d}"} for d in range(1, 15)]


VALID_EVALUATION = {
    "strengths": ["Strong in Coding (100%)."],
    "weaknesses": ["Technical needs work (20%)."],
    "recommendation": "Recommend",
    "summary": "Solid overall performance with room to grow in Technical.",
    "remediation_plan": _valid_remediation_plan(),
    "hiring_verdict": "Yes, likely hireable given strong Coding performance despite a weaker Technical round.",
}


@pytest.mark.asyncio
async def test_real_agent_valid_output_includes_remediation_plan_and_verdict():
    client = FakeClient([FakeResponse(VALID_EVALUATION)])
    agent = FinalEvaluatorAgent(client=client)

    result = await agent.run(target_company="TCS_NQT", overall_score=70.0, rounds={})

    assert len(result.data["remediation_plan"]) == 14
    assert result.data["hiring_verdict"] == VALID_EVALUATION["hiring_verdict"]
    assert result.usage is not None


@pytest.mark.asyncio
async def test_real_agent_wrong_remediation_plan_length_triggers_retry():
    malformed = dict(VALID_EVALUATION)
    malformed["remediation_plan"] = _valid_remediation_plan()[:10]  # only 10 days, not 14
    client = FakeClient([FakeResponse(malformed), FakeResponse(VALID_EVALUATION)])
    agent = FinalEvaluatorAgent(client=client)

    result = await agent.run(target_company="TCS_NQT", overall_score=70.0, rounds={})

    assert len(client.messages.calls) == 2
    assert len(result.data["remediation_plan"]) == 14


@pytest.mark.asyncio
async def test_real_agent_missing_hiring_verdict_triggers_retry():
    malformed = dict(VALID_EVALUATION)
    malformed["hiring_verdict"] = ""
    client = FakeClient([FakeResponse(malformed), FakeResponse(VALID_EVALUATION)])
    agent = FinalEvaluatorAgent(client=client)

    result = await agent.run(target_company="TCS_NQT", overall_score=70.0, rounds={})

    assert len(client.messages.calls) == 2
    assert result.data["hiring_verdict"] == VALID_EVALUATION["hiring_verdict"]


@pytest.mark.asyncio
async def test_real_agent_raises_after_max_attempts_with_persistently_malformed_plan():
    malformed = dict(VALID_EVALUATION)
    malformed["remediation_plan"] = []  # never valid
    client = FakeClient([FakeResponse(malformed) for _ in range(3)])
    agent = FinalEvaluatorAgent(client=client)

    with pytest.raises(FinalEvaluationError):
        await agent.run(target_company="TCS_NQT", overall_score=70.0, rounds={}, max_attempts=3)

    assert len(client.messages.calls) == 3


def test_factory_returns_mock_agent_when_mock_mode_enabled(monkeypatch):
    monkeypatch.setattr(fe_module, "MOCK_MODE", True)
    agent = build_final_evaluator()
    assert isinstance(agent, MockFinalEvaluatorAgent)


def test_factory_returns_real_agent_when_mock_mode_disabled(monkeypatch):
    monkeypatch.setattr(fe_module, "MOCK_MODE", False)
    agent = build_final_evaluator()
    assert isinstance(agent, FinalEvaluatorAgent)
