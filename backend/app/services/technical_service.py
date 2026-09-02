"""
Technical round business logic — the backend-layer glue between the
persistence models (InterviewSession, StageProgress, CostLog) and the
framework-agnostic agents/ package.

Mirrors app.services.aptitude_service's shape (see that module's
docstring): this decides *when* to call which agent for the technical
stage, persists the result, and uses agents.orchestrator.get_next_stage()
for the one stage transition it's responsible for (technical -> hr).

All in-progress technical state (topic sequence, difficulty, question
history, the currently pending question) lives in StageProgress.details as
a JSON blob, same as aptitude/coding — no new tables.
"""
from datetime import datetime, timezone

from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from agents.orchestrator.orchestrator import get_next_stage
from agents.technical_interviewer.grader import build_technical_independent_grader, build_technical_key_validator
from agents.technical_interviewer.taxonomy import (
    START_DIFFICULTY,
    SUPPORTED_COMPANIES,
    TOTAL_QUESTIONS,
    build_topic_sequence,
    clamp_difficulty,
    pick_pattern,
    session_rng,
)
from agents.technical_interviewer.technical_interviewer import build_technical_interviewer
from app.models.session import CostLog, InterviewSession, StageProgress

STAGE = "technical"

_technical_interviewer = build_technical_interviewer()
# Two independent agents, not one: _technical_independent_grader must
# finish before _technical_key_validator is ever called (see
# submit_answer() below), and the former's call site never has access to
# the generator's model_answer/rubric_keywords — see
# tests/test_grader_independence.py.
_technical_independent_grader = build_technical_independent_grader()
_technical_key_validator = build_technical_key_validator()


class TechnicalServiceError(Exception):
    """Raised for any technical-flow error; routes map status_code to an HTTPException."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def _get_session(db: Session, session_id: str) -> InterviewSession:
    session = db.get(InterviewSession, session_id)
    if session is None:
        raise TechnicalServiceError("Session not found", status_code=404)
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


async def _generate_question(session_id: str, target_company: str, details: dict, index: int) -> tuple[dict, object]:
    topic = details["topic_sequence"][index]
    used_patterns = {q["pattern"] for q in details["asked_questions"] if q["topic"] == topic}
    pattern = pick_pattern(topic, used_patterns, session_rng(session_id, index))
    previous_questions = [q["question"] for q in details["asked_questions"]]

    try:
        result = await _technical_interviewer.run(
            target_company=target_company,
            topic=topic,
            pattern=pattern,
            difficulty=details["difficulty"],
            previous_questions=previous_questions,
        )
    except Exception as exc:  # anthropic API errors, malformed tool output, missing mock fixture, etc.
        raise TechnicalServiceError(f"Technical question generation failed: {exc}", status_code=502) from exc

    return result.data, result.usage


def _public_question(question: dict) -> dict:
    return {
        "topic": question["topic"],
        "pattern": question["pattern"],
        "question": question["question"],
        "difficulty": question["difficulty"],
        "presented_at": question["presented_at"],
    }


def _state(session: InterviewSession, progress: StageProgress) -> dict:
    details = progress.details
    completed = progress.status == "completed"
    current_question = None
    if not completed and details.get("current_question") is not None:
        current_question = _public_question(details["current_question"])
    return {
        "session_id": session.id,
        "target_company": session.target_company,
        "total_questions": TOTAL_QUESTIONS,
        "current_index": min(details["current_index"] + 1, TOTAL_QUESTIONS),
        "difficulty": details["difficulty"],
        "score": details["score"],
        "completed": completed,
        "started_at": details.get("started_at"),
        "question": current_question,
    }


async def start_technical(db: Session, session_id: str) -> dict:
    session = _get_session(db, session_id)
    if session.target_company not in SUPPORTED_COMPANIES:
        raise TechnicalServiceError(
            f"Unsupported target_company '{session.target_company}'. "
            f"Supported: {', '.join(SUPPORTED_COMPANIES)}",
            status_code=400,
        )

    progress = _get_stage_progress(db, session_id)
    if progress is not None:
        # Idempotent: resume rather than restart, so a refresh-triggered
        # re-call doesn't wipe progress or re-charge for question 1.
        return _state(session, progress)

    now = datetime.now(timezone.utc)
    details = {
        "difficulty": START_DIFFICULTY,
        "difficulty_history": [],
        "topic_sequence": build_topic_sequence(session.target_company),
        "asked_questions": [],
        "current_question": None,
        "current_index": 0,
        "score": 0,
        "started_at": now.isoformat(),
        "completed_at": None,
    }

    question_data, usage = await _generate_question(session_id, session.target_company, details, 0)
    question_data["presented_at"] = datetime.now(timezone.utc).isoformat()
    details["current_question"] = question_data

    progress = StageProgress(
        session_id=session_id,
        stage=STAGE,
        status="in_progress",
        score=0,
        details=details,
        started_at=now,
    )
    db.add(progress)
    _log_cost(db, session_id, _technical_interviewer.name, usage)

    session.current_stage = STAGE
    db.add(session)
    db.commit()
    db.refresh(progress)
    db.refresh(session)

    return _state(session, progress)


def get_current(db: Session, session_id: str) -> dict:
    session = _get_session(db, session_id)
    progress = _get_stage_progress(db, session_id)
    if progress is None:
        raise TechnicalServiceError("Technical round not started for this session", status_code=409)
    return _state(session, progress)


async def submit_answer(db: Session, session_id: str, candidate_answer: str) -> dict:
    session = _get_session(db, session_id)
    progress = _get_stage_progress(db, session_id)
    if progress is None:
        raise TechnicalServiceError("Technical round not started for this session", status_code=409)
    if progress.status == "completed":
        raise TechnicalServiceError("Technical round already completed", status_code=409)

    details = progress.details
    current_question = details.get("current_question")
    if current_question is None:
        raise TechnicalServiceError("No pending question to answer", status_code=409)

    # Phase 1 — independent scoring: this call site does not have (and
    # does not pass) model_answer/rubric_keywords at all. The result is
    # fully produced before phase 2 ever begins.
    draft_result = await _technical_independent_grader.run(
        question=current_question["question"],
        candidate_answer=candidate_answer,
    )
    _log_cost(db, session_id, _technical_independent_grader.name, draft_result.usage)

    # Phase 2 — key-based validation: only now does the generator's answer
    # key (model_answer/rubric_keywords) enter the pipeline, alongside the
    # already-produced independent draft.
    grade_result = await _technical_key_validator.run(
        question=current_question["question"],
        model_answer=current_question["model_answer"],
        rubric_keywords=current_question["rubric_keywords"],
        candidate_answer=candidate_answer,
        draft_score=draft_result.data["draft_score"],
        draft_feedback=draft_result.data["draft_feedback"],
    )
    _log_cost(db, session_id, _technical_key_validator.name, grade_result.usage)

    is_correct: bool = grade_result.data["is_correct"]
    answer_score: int = grade_result.data["score"]
    feedback: str = grade_result.data["feedback"]

    presented_at = datetime.fromisoformat(current_question["presented_at"])
    answered_at = datetime.now(timezone.utc)
    response_time_seconds = max(0.0, (answered_at - presented_at).total_seconds())

    index = details["current_index"]
    details["asked_questions"].append(
        {
            "index": index,
            "topic": current_question["topic"],
            "pattern": current_question["pattern"],
            "question": current_question["question"],
            "candidate_answer": candidate_answer,
            "model_answer": current_question["model_answer"],
            "feedback": feedback,
            "answer_score": answer_score,
            "is_correct": is_correct,
            "difficulty": current_question["difficulty"],
            "presented_at": current_question["presented_at"],
            "answered_at": answered_at.isoformat(),
            "response_time_seconds": response_time_seconds,
        }
    )
    if is_correct:
        details["score"] += 1
    details["difficulty_history"].append(details["difficulty"])
    details["difficulty"] = clamp_difficulty(details["difficulty"] + (1 if is_correct else -1))
    details["current_index"] = index + 1
    details["current_question"] = None

    completed = details["current_index"] >= TOTAL_QUESTIONS
    next_question_public = None

    if completed:
        details["completed_at"] = answered_at.isoformat()
        progress.status = "completed"
        progress.completed_at = answered_at
        progress.score = details["score"]
        session.current_stage = get_next_stage(STAGE)
        db.add(session)
    else:
        question_data, usage = await _generate_question(
            session_id, session.target_company, details, details["current_index"]
        )
        _log_cost(db, session_id, _technical_interviewer.name, usage)
        question_data["presented_at"] = datetime.now(timezone.utc).isoformat()
        details["current_question"] = question_data
        next_question_public = _public_question(question_data)

    progress.details = details
    flag_modified(progress, "details")
    db.add(progress)
    db.commit()
    db.refresh(progress)

    return {
        "is_correct": is_correct,
        "answer_score": answer_score,
        "feedback": feedback,
        "score": details["score"],
        "difficulty": details["difficulty"],
        "current_index": min(details["current_index"], TOTAL_QUESTIONS),
        "total_questions": TOTAL_QUESTIONS,
        "response_time_seconds": response_time_seconds,
        "completed": completed,
        "next_question": next_question_public,
    }


def get_result(db: Session, session_id: str) -> dict:
    session = _get_session(db, session_id)
    progress = _get_stage_progress(db, session_id)
    if progress is None:
        raise TechnicalServiceError("Technical round not started for this session", status_code=409)
    if progress.status != "completed":
        raise TechnicalServiceError("Technical round not completed yet", status_code=409)

    details = progress.details
    history = details["asked_questions"]
    total_time = sum(q["response_time_seconds"] for q in history)

    topic_breakdown: dict[str, dict[str, int]] = {}
    for q in history:
        bucket = topic_breakdown.setdefault(q["topic"], {"correct": 0, "total": 0})
        bucket["total"] += 1
        if q["is_correct"]:
            bucket["correct"] += 1

    return {
        "session_id": session.id,
        "target_company": session.target_company,
        "total_questions": TOTAL_QUESTIONS,
        "score": details["score"],
        "percentage": round(100 * details["score"] / TOTAL_QUESTIONS, 2),
        "difficulty_final": details["difficulty"],
        "difficulty_history": details["difficulty_history"],
        "total_time_seconds": round(total_time, 2),
        "started_at": details.get("started_at"),
        "completed_at": details.get("completed_at"),
        "history": history,
        "topic_breakdown": topic_breakdown,
    }
