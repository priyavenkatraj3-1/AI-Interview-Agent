"""
Aptitude round endpoints (Day 2).

Thin HTTP layer over app.services.aptitude_service — request/response
validation via Pydantic schemas, session lookup via the existing get_db
dependency, and AptitudeServiceError -> HTTPException translation. All
actual state machine / generation / grading logic lives in the service and
the agents/ package.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.schemas import (
    AnswerSubmitRequest,
    AnswerSubmitResponse,
    AptitudeResultResponse,
    AptitudeStateResponse,
)
from app.services.aptitude_service import (
    AptitudeServiceError,
    get_current,
    get_result,
    start_aptitude,
    submit_answer,
)

router = APIRouter(prefix="/api/sessions/{session_id}/aptitude", tags=["aptitude"])


@router.post("/start", response_model=AptitudeStateResponse)
async def start(session_id: str, db: Session = Depends(get_db)):
    try:
        return await start_aptitude(db, session_id)
    except AptitudeServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.get("/current", response_model=AptitudeStateResponse)
def current(session_id: str, db: Session = Depends(get_db)):
    try:
        return get_current(db, session_id)
    except AptitudeServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.post("/answer", response_model=AnswerSubmitResponse)
async def answer(session_id: str, payload: AnswerSubmitRequest, db: Session = Depends(get_db)):
    try:
        return await submit_answer(db, session_id, payload.selected_option)
    except AptitudeServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.get("/result", response_model=AptitudeResultResponse)
def result(session_id: str, db: Session = Depends(get_db)):
    try:
        return get_result(db, session_id)
    except AptitudeServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
