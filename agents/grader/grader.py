"""
GraderAgent: scores a candidate's answer (aptitude MCQ, coding submission,
or free-text response) against the question's expected answer/rubric.

Day 1 placeholder — defines only the class shape from agents/base.py.
Code-execution-based grading (Judge0/Piston) is intentionally out of
scope until that stage is implemented.
"""
from agents.base import AgentResult, BaseAgent, ModelTier


class GraderAgent(BaseAgent):
    """Produces a score plus structured feedback for one submitted answer."""

    name = "grader"
    default_tier = ModelTier.CHEAP

    async def run(self, **kwargs) -> AgentResult:
        raise NotImplementedError(
            "GraderAgent.run() is a Day 1 placeholder — grading logic is "
            "added when this stage is built."
        )
