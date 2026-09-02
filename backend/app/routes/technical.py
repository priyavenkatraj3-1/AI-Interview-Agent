"""
Technical round endpoints (Day 4).

Thin HTTP layer over app.services.technical_service — mirrors
app.routes.aptitude's shape exactly: request/response validation via
Pydantic schemas, session lookup via the existing get_db dependency, and
TechnicalServiceError -> HTTPException translation. All actual state
machine / generation / grading logic lives in the service and the agents/
package.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.schemas import (
    TechnicalAnswerSubmitRequest,
    TechnicalAnswerSubmitResponse,
    TechnicalResultResponse,
    TechnicalStateResponse,
)
from app.services.technical_service import (
    TechnicalServiceError,
    get_current,
    get_result,
    start_technical,
    submit_answer,
)

router = APIRouter(prefix="/api/sessions/{session_id}/technical", tags=["technical"])


@router.post("/start", response_model=TechnicalStateResponse)
async def start(session_id: str, db: Session = Depends(get_db)):
    try:
        return await start_technical(db, session_id)
    except TechnicalServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.get("/current", response_model=TechnicalStateResponse)
def current(session_id: str, db: Session = Depends(get_db)):
    try:
        return get_current(db, session_id)
    except TechnicalServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.post("/answer", response_model=TechnicalAnswerSubmitResponse)
async def answer(session_id: str, payload: TechnicalAnswerSubmitRequest, db: Session = Depends(get_db)):
    try:
        return await submit_answer(db, session_id, payload.answer)
    except TechnicalServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.get("/result", response_model=TechnicalResultResponse)
def result(session_id: str, db: Session = Depends(get_db)):
    try:
        return get_result(db, session_id)
    except TechnicalServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
