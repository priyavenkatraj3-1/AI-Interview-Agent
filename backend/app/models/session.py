"""
Core persistence models for interview state.

These are intentionally the only tables created on Day 1. They exist so
that (a) a session survives a page refresh/reconnect, and (b) per-stage
cost tracking is available from the start rather than bolted on later.
Stage-specific schemas (question payloads, grading rubrics, transcripts)
will be added as each stage is implemented, generally as JSON blobs inside
`StageProgress.details` rather than new tables, to keep the schema stable.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class InterviewSession(Base):
    """One end-to-end placement simulation attempt by a student."""

    __tablename__ = "interview_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    candidate_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    candidate_email: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # One of: TCS_NQT, INFOSYS, WIPRO
    target_company: Mapped[str] = mapped_column(String(50), default="TCS_NQT")

    # One of: aptitude, coding, technical, hr, completed
    current_stage: Mapped[str] = mapped_column(String(50), default="aptitude")

    # One of: in_progress, completed, abandoned
    status: Mapped[str] = mapped_column(String(20), default="in_progress")

    # Free-form JSON for in-progress stage state (e.g. current question index,
    # adaptive difficulty tracker, timer start). Lets a refresh resume exactly
    # where the student left off without new tables per stage.
    state: Mapped[dict] = mapped_column(JSON, default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    stage_progress: Mapped[list["StageProgress"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    cost_logs: Mapped[list["CostLog"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class StageProgress(Base):
    """Per-stage outcome for a session (aptitude / coding / technical / hr)."""

    __tablename__ = "stage_progress"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("interview_sessions.id"))

    stage: Mapped[str] = mapped_column(String(50))
    status: Mapped[str] = mapped_column(String(20), default="not_started")
    score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Stage-specific results: question set, transcript, grading breakdown, etc.
    details: Mapped[dict] = mapped_column(JSON, default=dict)

    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    session: Mapped[InterviewSession] = relationship(back_populates="stage_progress")


class CostLog(Base):
    """Per-call AI cost/usage record, so cost can be aggregated per stage or session."""

    __tablename__ = "cost_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("interview_sessions.id"))

    stage: Mapped[str] = mapped_column(String(50))
    agent: Mapped[str] = mapped_column(String(100))
    model: Mapped[str] = mapped_column(String(100))

    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    session: Mapped[InterviewSession] = relationship(back_populates="cost_logs")
