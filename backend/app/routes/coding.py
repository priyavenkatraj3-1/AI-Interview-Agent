"""
Coding round endpoints (Day 3).

Thin HTTP layer over app.services.coding_service — mirrors
app.routes.aptitude's shape exactly: request/response validation via
Pydantic schemas, session lookup via the existing get_db dependency, and
CodingServiceError -> HTTPException translation. All actual state machine /
generation / execution / grading logic lives in the service and the
agents/ package.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.schemas import (
    CodeRunRequest,
    CodeRunResponse,
    CodeSubmitRequest,
    CodeSubmitResponse,
    CodingResultResponse,
    CodingStateResponse,
)
from app.services.coding_service import (
    CodingServiceError,
    get_current,
    get_result,
    run_code,
    start_coding,
    submit_code,
)

router = APIRouter(prefix="/api/sessions/{session_id}/coding", tags=["coding"])


@router.post("/start", response_model=CodingStateResponse)
async def start(session_id: str, db: Session = Depends(get_db)):
    try:
        return await start_coding(db, session_id)
    except CodingServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.get("/current", response_model=CodingStateResponse)
def current(session_id: str, db: Session = Depends(get_db)):
    try:
        return get_current(db, session_id)
    except CodingServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.post("/run", response_model=CodeRunResponse)
async def run(session_id: str, payload: CodeRunRequest, db: Session = Depends(get_db)):
    try:
        return await run_code(db, session_id, payload.code)
    except CodingServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.post("/submit", response_model=CodeSubmitResponse)
async def submit(session_id: str, payload: CodeSubmitRequest, db: Session = Depends(get_db)):
    try:
        return await submit_code(db, session_id, payload.code)
    except CodingServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.get("/result", response_model=CodingResultResponse)
def result(session_id: str, db: Session = Depends(get_db)):
    try:
        return get_result(db, session_id)
    except CodingServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
