"""
QuestionGeneratorAgent: generates stage-specific interview questions
(aptitude / coding / technical) on demand via the Claude API.

Day 1 placeholder — defines only the class shape from agents/base.py.
No hardcoded question bank and no scraping: when implemented, questions
are generated per-request by prompting Claude, not pulled from a static
list or an external site.
"""
from agents.base import AgentResult, BaseAgent, ModelTier


class QuestionGeneratorAgent(BaseAgent):
    """Produces one question (plus structured metadata) per call for the
    current stage and difficulty level."""

    name = "question_generator"
    default_tier = ModelTier.CHEAP

    async def run(self, **kwargs) -> AgentResult:
        raise NotImplementedError(
            "QuestionGeneratorAgent.run() is a Day 1 placeholder — prompt "
            "design and Claude API calls are added when this stage is built."
        )
