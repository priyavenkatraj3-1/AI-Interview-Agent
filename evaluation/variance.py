"""
Grading-variance check: grade the SAME fixed candidate answer N times via
a round's real grader agent and report the variance of the resulting
scores.

Reuses the real grader agents directly — no new grading logic is
introduced here. In MOCK_MODE (the default in this environment, since
Anthropic API credits are unavailable) grading is a deterministic function
of its inputs, so repeated grading of an unchanged input is expected to
produce variance == 0.0 — that is the correct, honest result to report,
not a limitation of this harness. This function can equally be pointed at
the real Claude-backed graders (MOCK_MODE=false, with credits) to measure
genuine LLM grading variance; that path is not exercised in this
environment, and results must never be reported as if it were.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass
class VarianceResult:
    scores: list[float]
    mean: float
    variance: float


async def grade_fixed_answer_n_times(grader, grade_kwargs: dict, n: int = 5) -> VarianceResult:
    """Call `grader.run(**grade_kwargs)` `n` times on the same fixed input
    and return the resulting scores plus their mean/population variance."""
    scores: list[float] = []
    for _ in range(n):
        result = await grader.run(**grade_kwargs)
        scores.append(float(result.data["score"]))

    mean = statistics.mean(scores)
    variance = statistics.pvariance(scores) if len(scores) > 1 else 0.0
    return VarianceResult(scores=scores, mean=mean, variance=variance)
