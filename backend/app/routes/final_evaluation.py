"""
Final evaluation endpoint (Day 3 continued).

Thin HTTP layer over app.services.final_evaluation_service — mirrors the
other stage routers' shape: request/response validation via Pydantic
schemas, session lookup via the existing get_db dependency, and
FinalEvaluationServiceError -> HTTPException translation. All actual
aggregation / synthesis logic lives in the service and the agents/
package.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.schemas import FinalEvaluationResponse
from app.services.final_evaluation_service import (
    FinalEvaluationServiceError,
    generate_final_evaluation,
    get_final_evaluation,
)

router = APIRouter(prefix="/api/sessions/{session_id}/final-evaluation", tags=["final-evaluation"])


@router.post("/generate", response_model=FinalEvaluationResponse)
async def generate(session_id: str, db: Session = Depends(get_db)):
    try:
        return await generate_final_evaluation(db, session_id)
    except FinalEvaluationServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.get("", response_model=FinalEvaluationResponse)
def result(session_id: str, db: Session = Depends(get_db)):
    try:
        return get_final_evaluation(db, session_id)
    except FinalEvaluationServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
