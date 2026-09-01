"""Pydantic request/response schemas for the API layer (kept separate from
the SQLAlchemy ORM models in session.py)."""
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


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


# --- Aptitude round (Day 2) ---


class AptitudeQuestionPublic(BaseModel):
    """The question as sent to the client — never includes correct_option
    or explanation until after the answer is submitted."""

    topic: str
    pattern: str
    question: str
    options: list[str]
    difficulty: int
    presented_at: datetime


class AptitudeStateResponse(BaseModel):
    """Shared shape for the start and current-question endpoints."""

    session_id: str
    target_company: str
    total_questions: int
    current_index: int  # 1-based number of the question being served
    difficulty: int
    score: int
    completed: bool
    started_at: datetime | None
    question: AptitudeQuestionPublic | None


class AnswerSubmitRequest(BaseModel):
    selected_option: int = Field(ge=0, le=3)


class AnswerSubmitResponse(BaseModel):
    is_correct: bool
    correct_option: int
    explanation: str
    score: int
    difficulty: int
    current_index: int
    total_questions: int
    response_time_seconds: float
    completed: bool
    next_question: AptitudeQuestionPublic | None


class AptitudeHistoryItem(BaseModel):
    index: int
    topic: str
    pattern: str
    question: str
    options: list[str]
    correct_option: int
    selected_option: int
    is_correct: bool
    difficulty: int
    presented_at: datetime
    answered_at: datetime
    response_time_seconds: float | None
    explanation: str


class AptitudeResultResponse(BaseModel):
    session_id: str
    target_company: str
    total_questions: int
    score: int
    percentage: float
    difficulty_final: int
    difficulty_history: list[int]
    total_time_seconds: float
    started_at: datetime | None
    completed_at: datetime | None
    history: list[AptitudeHistoryItem]
    topic_breakdown: dict[str, dict[str, int]]
