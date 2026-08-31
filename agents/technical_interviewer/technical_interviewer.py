"""
TechnicalInterviewerAgent: conducts the conversational technical-interview
stage — asks follow-up questions, probes depth on the candidate's
answers, and decides when the technical round is complete.

Day 1 placeholder — defines only the class shape from agents/base.py.
"""
from agents.base import AgentResult, BaseAgent, ModelTier


class TechnicalInterviewerAgent(BaseAgent):
    """Drives one turn of the technical interview conversation."""

    name = "technical_interviewer"
    default_tier = ModelTier.STRONG

    async def run(self, **kwargs) -> AgentResult:
        raise NotImplementedError(
            "TechnicalInterviewerAgent.run() is a Day 1 placeholder — "
            "conversation logic is added when this stage is built."
        )
