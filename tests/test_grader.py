"""Unit tests for GraderAgent — deterministic MCQ comparison, no API call."""
import pytest

from agents.grader.grader import GraderAgent


@pytest.mark.asyncio
async def test_correct_answer_is_graded_correct():
    agent = GraderAgent()
    result = await agent.run(selected_option=2, correct_option=2)
    assert result.data["is_correct"] is True
    assert result.usage is None  # deterministic grading makes no API call


@pytest.mark.asyncio
async def test_incorrect_answer_is_graded_incorrect():
    agent = GraderAgent()
    result = await agent.run(selected_option=0, correct_option=3)
    assert result.data["is_correct"] is False


@pytest.mark.asyncio
async def test_result_echoes_selected_and_correct_option():
    agent = GraderAgent()
    result = await agent.run(selected_option=1, correct_option=3)
    assert result.data["selected_option"] == 1
    assert result.data["correct_option"] == 3


# --- Coding submissions (Day 3): aggregates pre-computed test-case results
# from agents.code_executor — still no API call, no CostLog entry. ---


@pytest.mark.asyncio
async def test_coding_all_tests_passed_is_graded_correct():
    agent = GraderAgent()
    result = await agent.run(test_results=[{"passed": True}, {"passed": True}])
    assert result.data["is_correct"] is True
    assert result.data["passed_count"] == 2
    assert result.data["total_count"] == 2
    assert result.usage is None  # deterministic grading makes no API call


@pytest.mark.asyncio
async def test_coding_partial_pass_is_graded_incorrect():
    agent = GraderAgent()
    result = await agent.run(test_results=[{"passed": True}, {"passed": False}])
    assert result.data["is_correct"] is False
    assert result.data["passed_count"] == 1
    assert result.data["total_count"] == 2


@pytest.mark.asyncio
async def test_coding_empty_test_results_is_graded_incorrect():
    agent = GraderAgent()
    result = await agent.run(test_results=[])
    assert result.data["is_correct"] is False
    assert result.data["passed_count"] == 0
    assert result.data["total_count"] == 0
