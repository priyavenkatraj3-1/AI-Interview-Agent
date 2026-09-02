"""
HR round endpoints (Day 3 continued).

Thin HTTP layer over app.services.hr_service — mirrors
app.routes.technical's shape exactly: request/response validation via
Pydantic schemas, session lookup via the existing get_db dependency, and
HRServiceError -> HTTPException translation. All actual state machine /
generation / grading logic lives in the service and the agents/ package.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.schemas import (
    HRAnswerSubmitRequest,
    HRAnswerSubmitResponse,
    HRResultResponse,
    HRStateResponse,
)
from app.services.hr_service import (
    HRServiceError,
    get_current,
    get_result,
    start_hr,
    submit_answer,
)

router = APIRouter(prefix="/api/sessions/{session_id}/hr", tags=["hr"])


@router.post("/start", response_model=HRStateResponse)
async def start(session_id: str, db: Session = Depends(get_db)):
    try:
        return await start_hr(db, session_id)
    except HRServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.get("/current", response_model=HRStateResponse)
def current(session_id: str, db: Session = Depends(get_db)):
    try:
        return get_current(db, session_id)
    except HRServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.post("/answer", response_model=HRAnswerSubmitResponse)
async def answer(session_id: str, payload: HRAnswerSubmitRequest, db: Session = Depends(get_db)):
    try:
        return await submit_answer(db, session_id, payload.answer)
    except HRServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.get("/result", response_model=HRResultResponse)
def result(session_id: str, db: Session = Depends(get_db)):
    try:
        return get_result(db, session_id)
    except HRServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
