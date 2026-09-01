"""
GraderAgent: scores a candidate's answer (aptitude MCQ, coding submission,
or free-text response) against the question's expected answer/rubric.

Day 2: implements the aptitude MCQ case. Grading an MCQ against an
already-known correct option is a deterministic comparison, not something
that needs a Claude call — so this makes no API call and produces no
CostLog entry. Kept as its own agent (rather than inlined into the
service) so question generation and grading stay separate
responsibilities, and so later stages (coding/technical/HR) can swap in
model-based grading behind the same `run()` contract.
"""
from agents.base import AgentResult, BaseAgent, ModelTier


class GraderAgent(BaseAgent):
    """Produces a score plus structured feedback for one submitted answer."""

    name = "grader"
    default_tier = ModelTier.CHEAP

    async def run(self, **kwargs) -> AgentResult:
        selected_option: int = kwargs["selected_option"]
        correct_option: int = kwargs["correct_option"]
        is_correct = selected_option == correct_option
        return AgentResult(
            data={
                "is_correct": is_correct,
                "selected_option": selected_option,
                "correct_option": correct_option,
            },
            usage=None,
        )
