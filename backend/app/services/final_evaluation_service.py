"""
Final evaluation business logic — the backend-layer glue between the four
completed stage results (aptitude, coding, technical, hr) and the
framework-agnostic final-evaluation agent.

Mirrors every other *_service.py module's shape: an idempotent "generate"
entry point (like each round's start_*), backed by a StageProgress row for
stage="completed" — the terminal stage in
agents.orchestrator.orchestrator.STAGE_SEQUENCE, which every session is
already sitting at once the HR round finishes — holding the consolidated
report in `details`, the same JSON-blob pattern used everywhere else. No
new tables.

Reuses each round's own get_result() directly rather than re-deriving any
scoring/topic-breakdown logic, so this module has no scoring rules of its
own beyond the overall-score average.
"""
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from agents.orchestrator.final_evaluator import build_final_evaluator
from app.models.session import CostLog, InterviewSession, StageProgress
from app.services.aptitude_service import AptitudeServiceError
from app.services.aptitude_service import get_result as _get_aptitude_result
from app.services.coding_service import CodingServiceError
from app.services.coding_service import get_result as _get_coding_result
from app.services.hr_service import HRServiceError
from app.services.hr_service import get_result as _get_hr_result
from app.services.technical_service import TechnicalServiceError
from app.services.technical_service import get_result as _get_technical_result

STAGE = "completed"

_final_evaluator = build_final_evaluator()

# Any of the four rounds' get_result() raises its own *ServiceError (always
# with a status_code) when that round hasn't started/finished yet — caught
# together here since they all mean the same thing at this layer: the
# session isn't ready for a final evaluation yet.
_ROUND_RESULT_ERRORS = (AptitudeServiceError, CodingServiceError, TechnicalServiceError, HRServiceError)


class FinalEvaluationServiceError(Exception):
    """Raised for any final-evaluation error; routes map status_code to an HTTPException."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def _get_session(db: Session, session_id: str) -> InterviewSession:
    session = db.get(InterviewSession, session_id)
    if session is None:
        raise FinalEvaluationServiceError("Session not found", status_code=404)
    return session


def _get_stage_progress(db: Session, session_id: str) -> StageProgress | None:
    return (
        db.query(StageProgress)
        .filter(StageProgress.session_id == session_id, StageProgress.stage == STAGE)
        .one_or_none()
    )


def _log_cost(db: Session, session_id: str, agent_name: str, usage) -> None:
    if usage is None:
        return
    db.add(
        CostLog(
            session_id=session_id,
            stage=STAGE,
            agent=agent_name,
            model=usage.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=usage.cost_usd,
        )
    )


def _round_summary(
    label: str, db: Session, session_id: str, get_result_fn, total_key: str, *, extra_keys: tuple[str, ...] = ()
) -> dict:
    try:
        result = get_result_fn(db, session_id)
    except _ROUND_RESULT_ERRORS as exc:
        raise FinalEvaluationServiceError(
            f"Cannot generate final evaluation: the {label} round is not completed yet ({exc})",
            status_code=409,
        ) from exc

    summary = {
        "score": result["score"],
        "total": result[total_key],
        "percentage": result["percentage"],
        "topic_breakdown": result["topic_breakdown"],
    }
    # Round-specific extras (currently only the coding round's
    # average_quality_score) -- passed straight through from that round's
    # own get_result(), never affecting the other rounds' plain shape.
    for key in extra_keys:
        summary[key] = result.get(key)
    return summary


def _public_evaluation(session: InterviewSession, details: dict) -> dict:
    return {
        "session_id": session.id,
        "target_company": session.target_company,
        "overall_score": details["overall_score"],
        "aptitude": details["rounds"]["aptitude"],
        "coding": details["rounds"]["coding"],
        "technical": details["rounds"]["technical"],
        "hr": details["rounds"]["hr"],
        "strengths": details["strengths"],
        "weaknesses": details["weaknesses"],
        "recommendation": details["recommendation"],
        "summary": details["summary"],
        "remediation_plan": details["remediation_plan"],
        "hiring_verdict": details["hiring_verdict"],
        "generated_at": details["generated_at"],
    }


async def generate_final_evaluation(db: Session, session_id: str) -> dict:
    session = _get_session(db, session_id)

    progress = _get_stage_progress(db, session_id)
    if progress is not None:
        # Idempotent: return the already-generated evaluation rather than
        # re-running the agent (and re-charging Claude credits), same as
        # every other round's start_* idempotency.
        return _public_evaluation(session, progress.details)

    rounds = {
        "aptitude": _round_summary("Aptitude", db, session_id, _get_aptitude_result, "total_questions"),
        "coding": _round_summary(
            "Coding", db, session_id, _get_coding_result, "total_problems", extra_keys=("average_quality_score",)
        ),
        "technical": _round_summary("Technical", db, session_id, _get_technical_result, "total_questions"),
        "hr": _round_summary("HR", db, session_id, _get_hr_result, "total_questions"),
    }

    # Human-readable summary of the coding round's code-quality result
    # (separate from its functional score/percentage above), so both the
    # Claude-backed and mock final evaluators -- and the public API
    # response's CodingRoundSummary -- have direct access to it.
    coding_quality_score = rounds["coding"].get("average_quality_score")
    if coding_quality_score is not None:
        rounds["coding"]["quality_feedback_summary"] = (
            f"Average code-quality score across submissions: {coding_quality_score}/100 "
            "(readability, naming, structure, duplication, best practices, maintainability)."
        )

    overall_score = round(sum(r["percentage"] for r in rounds.values()) / len(rounds), 2)

    try:
        result = await _final_evaluator.run(
            target_company=session.target_company,
            overall_score=overall_score,
            rounds=rounds,
        )
    except Exception as exc:  # anthropic API errors, malformed tool output, etc.
        raise FinalEvaluationServiceError(f"Final evaluation generation failed: {exc}", status_code=502) from exc

    now = datetime.now(timezone.utc)
    details = {
        "overall_score": overall_score,
        "rounds": rounds,
        "strengths": result.data["strengths"],
        "weaknesses": result.data["weaknesses"],
        "recommendation": result.data["recommendation"],
        "summary": result.data["summary"],
        "remediation_plan": result.data["remediation_plan"],
        "hiring_verdict": result.data["hiring_verdict"],
        "generated_at": now.isoformat(),
    }

    progress = StageProgress(
        session_id=session_id,
        stage=STAGE,
        status="completed",
        score=overall_score,
        details=details,
        started_at=now,
        completed_at=now,
    )
    db.add(progress)
    _log_cost(db, session_id, _final_evaluator.name, result.usage)
    db.commit()
    db.refresh(progress)

    return _public_evaluation(session, progress.details)


def get_final_evaluation(db: Session, session_id: str) -> dict:
    session = _get_session(db, session_id)
    progress = _get_stage_progress(db, session_id)
    if progress is None:
        raise FinalEvaluationServiceError("Final evaluation not generated yet for this session", status_code=409)
    return _public_evaluation(session, progress.details)
