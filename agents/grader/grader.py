"""
GraderAgent: scores a candidate's answer (aptitude MCQ, coding submission,
or free-text response) against the question's expected answer/rubric.

Day 2 implemented the aptitude MCQ case. Day 3 adds the coding case:
given already-computed per-test-case pass/fail results (the actual code
execution happens in agents.code_executor, not here), this just aggregates
them into a score/is_correct verdict. Both cases are deterministic
comparisons, not something that needs a Claude call — so neither makes an
API call or produces a CostLog entry. Kept as one agent (rather than a
second one per stage) since question/problem generation and grading stay
separate responsibilities regardless of which stage is being graded, and
later stages (technical/HR) can still swap in model-based grading behind
the same `run()` contract.
"""
from agents.base import AgentResult, BaseAgent, ModelTier


class GraderAgent(BaseAgent):
    """Produces a score plus structured feedback for one submitted answer."""

    name = "grader"
    default_tier = ModelTier.CHEAP

    async def run(self, **kwargs) -> AgentResult:
        if "test_results" in kwargs:
            return self._grade_coding(kwargs["test_results"])
        return self._grade_mcq(kwargs["selected_option"], kwargs["correct_option"])

    def _grade_mcq(self, selected_option: int, correct_option: int) -> AgentResult:
        is_correct = selected_option == correct_option
        return AgentResult(
            data={
                "is_correct": is_correct,
                "selected_option": selected_option,
                "correct_option": correct_option,
            },
            usage=None,
        )

    def _grade_coding(self, test_results: list[dict]) -> AgentResult:
        total_count = len(test_results)
        passed_count = sum(1 for t in test_results if t["passed"])
        is_correct = total_count > 0 and passed_count == total_count
        return AgentResult(
            data={
                "is_correct": is_correct,
                "passed_count": passed_count,
                "total_count": total_count,
            },
            usage=None,
        )
