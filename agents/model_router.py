"""
Model routing: maps a cost/capability tier to an actual Claude model id.

Kept as a single, tiny indirection point so stage agents never hardcode a
model id — they ask for a ModelTier and this resolves the concrete model.
This is what lets the system route cheaper models to high-volume,
low-risk calls (question generation, first-pass grading) while reserving
stronger models for calls that need deeper reasoning (technical interview
follow-ups, HR nuance judgement, final hire/no-hire verdict), without
touching agent code when the routing policy changes.
"""
from agents.base import ModelTier
from agents.config import MODEL_CHEAP, MODEL_STRONG


def resolve_model(tier: ModelTier) -> str:
    return MODEL_STRONG if tier == ModelTier.STRONG else MODEL_CHEAP
