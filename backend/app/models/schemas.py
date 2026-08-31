"""Pydantic request/response schemas for the API layer (kept separate from
the SQLAlchemy ORM models in session.py)."""
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class SessionCreateRequest(BaseModel):
    candidate_name: str | None = None
    candidate_email: str | None = None
    target_company: str = "TCS_NQT"


class SessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    candidate_name: str | None
    candidate_email: str | None
    target_company: str
    current_stage: str
    status: str
    state: dict
    created_at: datetime
    updated_at: datetime
