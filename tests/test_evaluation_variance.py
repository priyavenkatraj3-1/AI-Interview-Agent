"""
Unit tests for evaluation/variance.py, exercised against the real
(MOCK_MODE) phase-2 key-validator grader agent — no Anthropic API key
needed. Also empirically demonstrates the "mock keyword-overlap grading is
gameable" failure mode documented in docs/evaluation_report.md.

Grading is now a two-phase pipeline (see agents/technical_interviewer/
grader.py and tests/test_grader_independence.py): phase 1 produces an
answer-key-blind draft, phase 2 validates against the model_answer/
rubric_keywords. These tests exercise phase 2 directly with a fixed draft
held constant, since phase 2's keyword-overlap formula is the deterministic
piece being measured for variance.
"""
import pytest

from agents.technical_interviewer.grader import MockTechnicalKeyValidatorAgent
from evaluation.variance import grade_fixed_answer_n_times

QUESTION = "Explain the difference between a stack and a queue."
MODEL_ANSWER = "A stack is LIFO; a queue is FIFO."
RUBRIC_KEYWORDS = ["stack", "queue", "lifo", "fifo"]
FIXED_STRONG_ANSWER = "A stack is LIFO and a queue is FIFO."
FIXED_DRAFT_KWARGS = {"draft_score": 90, "draft_feedback": "Independent draft: looks solid."}


@pytest.mark.asyncio
async def test_grading_the_same_fixed_answer_five_times_has_zero_variance_in_mock_mode():
    grader = MockTechnicalKeyValidatorAgent()
    result = await grade_fixed_answer_n_times(
        grader,
        {
            "question": QUESTION,
            "model_answer": MODEL_ANSWER,
            "rubric_keywords": RUBRIC_KEYWORDS,
            "candidate_answer": FIXED_STRONG_ANSWER,
            **FIXED_DRAFT_KWARGS,
        },
        n=5,
    )
    assert len(result.scores) == 5
    assert result.scores == [100.0] * 5
    assert result.mean == 100.0
    assert result.variance == 0.0


@pytest.mark.asyncio
async def test_grading_a_fixed_partial_answer_five_times_also_has_zero_variance():
    # A different fixed answer (partial keyword match) — still deterministic.
    grader = MockTechnicalKeyValidatorAgent()
    result = await grade_fixed_answer_n_times(
        grader,
        {
            "question": QUESTION,
            "model_answer": MODEL_ANSWER,
            "rubric_keywords": RUBRIC_KEYWORDS,
            "candidate_answer": "A stack is a data structure.",  # matches only "stack"
            **FIXED_DRAFT_KWARGS,
        },
        n=5,
    )
    assert result.scores == [25.0] * 5
    assert result.variance == 0.0


@pytest.mark.asyncio
async def test_keyword_stuffed_nonsense_answer_scores_full_marks_in_mock_mode():
    # Documents failure mode #2 from docs/evaluation_report.md: the mock
    # keyword-overlap grader can't tell a genuinely strong answer from a
    # nonsense sentence that just contains every rubric keyword. This is a
    # limitation of phase 2 (key-based validation) specifically -- phase 1
    # (independent, see tests/test_grader_independence.py) never sees the
    # rubric at all, so it cannot be gamed this way, though its own crude
    # length-based heuristic has its own honestly-documented limitations.
    grader = MockTechnicalKeyValidatorAgent()
    nonsense_answer = "stack queue lifo fifo banana banana zzz not a real sentence"
    result = await grader.run(
        question=QUESTION,
        model_answer=MODEL_ANSWER,
        rubric_keywords=RUBRIC_KEYWORDS,
        candidate_answer=nonsense_answer,
        **FIXED_DRAFT_KWARGS,
    )
    assert result.data["score"] == 100
    assert result.data["is_correct"] is True
