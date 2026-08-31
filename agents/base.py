"""
Shared contract that every stage agent (question generator, grader,
technical interviewer, hr interviewer) implements, and that the
orchestrator drives.

Full agents are NOT implemented yet — this defines shape only:
- structured JSON in/out (never free-form text parsing downstream)
- a model tier per call, resolved to a concrete model by model_router.py
- a place to attach token/cost usage for the cost-tracking requirement

Concrete prompting, Claude API calls, and JSON schema validation are added
when each stage is built.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ModelTier(str, Enum):
    """Cost/capability tier used for model routing (see agents/model_router.py)."""

    CHEAP = "cheap"  # high-volume, low-risk: question generation, first-pass grading
    STRONG = "strong"  # deeper reasoning: technical follow-ups, final verdict synthesis


@dataclass
class AgentUsage:
    """Token/cost accounting for a single agent call, persisted via the CostLog table."""

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class AgentResult:
    """
    Every agent returns structured data (a dict matching a defined JSON
    schema) plus usage info — never a raw free-form string that a
    downstream component has to parse with regex/heuristics.
    """

    data: dict[str, Any]
    usage: AgentUsage | None = None


class BaseAgent(ABC):
    """Base class for all AI agents in the system."""

    name: str = "base_agent"
    default_tier: ModelTier = ModelTier.CHEAP

    @abstractmethod
    async def run(self, **kwargs) -> AgentResult:
        """
        Execute the agent for one unit of work (one generated question, one
        graded answer, one interview turn) and return structured output.

        Day 1 placeholder — not implemented yet.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.run() is not implemented yet (Day 1 placeholder)."
        )
