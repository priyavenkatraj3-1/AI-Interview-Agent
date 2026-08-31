"""
Minimal session endpoints.

These exist on Day 1 purely to prove the persistence layer works end to
end (create a session, refresh the page, fetch it back by id). The real
stage-driving logic (starting aptitude questions, advancing state machine,
etc.) is added when each stage is implemented.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.schemas import SessionCreateRequest, SessionResponse
from app.models.session import InterviewSession

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.post("", response_model=SessionResponse)
def create_session(payload: SessionCreateRequest, db: Session = Depends(get_db)):
    session = InterviewSession(
        candidate_name=payload.candidate_name,
        candidate_email=payload.candidate_email,
        target_company=payload.target_company,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@router.get("/{session_id}", response_model=SessionResponse)
def get_session(session_id: str, db: Session = Depends(get_db)):
    session = db.get(InterviewSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session
