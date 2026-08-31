"""
HRInterviewerAgent: conducts the conversational HR-interview stage
(behavioural/situational questions, culture-fit signal).

Day 1 placeholder — defines only the class shape from agents/base.py.
The multi-persona evaluation approach is intentionally out of scope
until that stage is implemented.
"""
from agents.base import AgentResult, BaseAgent, ModelTier


class HRInterviewerAgent(BaseAgent):
    """Drives one turn of the HR interview conversation."""

    name = "hr_interviewer"
    default_tier = ModelTier.STRONG

    async def run(self, **kwargs) -> AgentResult:
        raise NotImplementedError(
            "HRInterviewerAgent.run() is a Day 1 placeholder — conversation "
            "logic is added when this stage is built."
        )
