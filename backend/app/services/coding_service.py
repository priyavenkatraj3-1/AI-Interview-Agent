"""
Coding round business logic — the backend-layer glue between the
persistence models (InterviewSession, StageProgress, CostLog) and the
framework-agnostic agents/ package.

Mirrors app.services.aptitude_service's shape exactly (see that module's
docstring): this decides *when* to call which agent for the coding stage,
persists the result, and uses agents.orchestrator.get_next_stage() for the
one stage transition it's responsible for (coding -> whatever's next).

All in-progress coding state (topic sequence, difficulty, problem history,
the currently pending problem, submitted code) lives in
StageProgress.details as a JSON blob, same as the aptitude stage — no new
tables.
"""
import asyncio
import json
from datetime import datetime, timezone

from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from agents.code_executor.executor import UnsafeCodeError, run_test_cases
from agents.code_problem_generator.code_problem_generator import build_code_problem_generator
from agents.code_problem_generator.taxonomy import (
    START_DIFFICULTY,
    TOTAL_PROBLEMS,
    build_topic_sequence,
    clamp_difficulty,
    pick_pattern,
    session_rng,
)
from agents.grader.grader import GraderAgent
from agents.orchestrator.orchestrator import get_next_stage
from app.models.session import CostLog, InterviewSession, StageProgress

STAGE = "coding"

_code_problem_generator = build_code_problem_generator()
_grader = GraderAgent()


class CodingServiceError(Exception):
    """Raised for any coding-flow error; routes map status_code to an HTTPException."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def _get_session(db: Session, session_id: str) -> InterviewSession:
    session = db.get(InterviewSession, session_id)
    if session is None:
        raise CodingServiceError("Session not found", status_code=404)
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


async def _generate_problem(session_id: str, details: dict, index: int) -> tuple[dict, object]:
    topic = details["topic_sequence"][index]
    used_patterns = {p["pattern"] for p in details["asked_problems"] if p["topic"] == topic}
    pattern = pick_pattern(topic, used_patterns, session_rng(session_id, index))
    previous_titles = [p["title"] for p in details["asked_problems"]]

    try:
        result = await _code_problem_generator.run(
            topic=topic,
            pattern=pattern,
            difficulty=details["difficulty"],
            previous_titles=previous_titles,
        )
    except Exception as exc:  # anthropic API errors, malformed tool output, missing mock fixture, etc.
        raise CodingServiceError(f"Coding problem generation failed: {exc}", status_code=502) from exc

    return result.data, result.usage


def _public_problem(problem: dict) -> dict:
    return {
        "topic": problem["topic"],
        "pattern": problem["pattern"],
        "title": problem["title"],
        "description": problem["description"],
        "constraints": problem["constraints"],
        "function_name": problem["function_name"],
        "starter_code": problem["starter_code"],
        "examples": problem["examples"],
        "difficulty": problem["difficulty"],
        "presented_at": problem["presented_at"],
    }


def _public_results(results: list[dict]) -> list[dict]:
    return [
        {
            "input": json.dumps(r["args"]),
            "expected_output": json.dumps(r["expected"]),
            "actual_output": json.dumps(r["actual"]) if r["error"] is None else None,
            "passed": r["passed"],
            "error": r["error"],
            "timed_out": r["timed_out"],
            "stdout": r["stdout"],
            "stderr": r["stderr"],
        }
        for r in results
    ]


def _state(session: InterviewSession, progress: StageProgress) -> dict:
    details = progress.details
    completed = progress.status == "completed"
    current_problem = None
    if not completed and details.get("current_problem") is not None:
        current_problem = _public_problem(details["current_problem"])
    return {
        "session_id": session.id,
        "target_company": session.target_company,
        "total_problems": TOTAL_PROBLEMS,
        "current_index": min(details["current_index"] + 1, TOTAL_PROBLEMS),
        "difficulty": details["difficulty"],
        "score": details["score"],
        "completed": completed,
        "started_at": details.get("started_at"),
        "problem": current_problem,
    }


async def start_coding(db: Session, session_id: str) -> dict:
    session = _get_session(db, session_id)

    progress = _get_stage_progress(db, session_id)
    if progress is not None:
        # Idempotent: resume rather than restart, so a refresh-triggered
        # re-call doesn't wipe progress or re-charge for problem 1.
        return _state(session, progress)

    now = datetime.now(timezone.utc)
    details = {
        "difficulty": START_DIFFICULTY,
        "difficulty_history": [],
        "topic_sequence": build_topic_sequence(TOTAL_PROBLEMS),
        "asked_problems": [],
        "current_problem": None,
        "current_index": 0,
        "score": 0,
        "started_at": now.isoformat(),
        "completed_at": None,
    }

    problem_data, usage = await _generate_problem(session_id, details, 0)
    problem_data["presented_at"] = datetime.now(timezone.utc).isoformat()
    details["current_problem"] = problem_data

    progress = StageProgress(
        session_id=session_id,
        stage=STAGE,
        status="in_progress",
        score=0,
        details=details,
        started_at=now,
    )
    db.add(progress)
    _log_cost(db, session_id, _code_problem_generator.name, usage)

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
        raise CodingServiceError("Coding round not started for this session", status_code=409)
    return _state(session, progress)


def _require_pending_problem(progress: StageProgress) -> dict:
    if progress.status == "completed":
        raise CodingServiceError("Coding round already completed", status_code=409)
    current_problem = progress.details.get("current_problem")
    if current_problem is None:
        raise CodingServiceError("No pending coding problem", status_code=409)
    return current_problem


async def run_code(db: Session, session_id: str, code: str) -> dict:
    """Non-scored 'run': execute against the problem's public/sample test
    case(s) only. Doesn't consume an attempt or affect score/difficulty."""
    _get_session(db, session_id)
    progress = _get_stage_progress(db, session_id)
    if progress is None:
        raise CodingServiceError("Coding round not started for this session", status_code=409)
    current_problem = _require_pending_problem(progress)

    try:
        summary = await asyncio.to_thread(
            run_test_cases, code, current_problem["function_name"], current_problem["public_tests"]
        )
    except UnsafeCodeError as exc:
        raise CodingServiceError(str(exc), status_code=422) from exc

    return {
        "results": _public_results(summary["results"]),
        "passed_count": summary["passed_count"],
        "total_count": summary["total_count"],
    }


async def submit_code(db: Session, session_id: str, code: str) -> dict:
    session = _get_session(db, session_id)
    progress = _get_stage_progress(db, session_id)
    if progress is None:
        raise CodingServiceError("Coding round not started for this session", status_code=409)
    current_problem = _require_pending_problem(progress)

    try:
        summary = await asyncio.to_thread(
            run_test_cases, code, current_problem["function_name"], current_problem["hidden_tests"]
        )
    except UnsafeCodeError as exc:
        raise CodingServiceError(str(exc), status_code=422) from exc

    grade_result = await _grader.run(test_results=summary["results"])
    is_correct: bool = grade_result.data["is_correct"]
    passed_count: int = grade_result.data["passed_count"]
    total_count: int = grade_result.data["total_count"]

    details = progress.details
    presented_at = datetime.fromisoformat(current_problem["presented_at"])
    submitted_at = datetime.now(timezone.utc)
    response_time_seconds = max(0.0, (submitted_at - presented_at).total_seconds())

    index = details["current_index"]
    details["asked_problems"].append(
        {
            "index": index,
            "topic": current_problem["topic"],
            "pattern": current_problem["pattern"],
            "title": current_problem["title"],
            "difficulty": current_problem["difficulty"],
            "submitted_code": code,
            "passed_count": passed_count,
            "total_count": total_count,
            "is_correct": is_correct,
            "presented_at": current_problem["presented_at"],
            "submitted_at": submitted_at.isoformat(),
            "response_time_seconds": response_time_seconds,
        }
    )
    if is_correct:
        details["score"] += 1
    details["difficulty_history"].append(details["difficulty"])
    details["difficulty"] = clamp_difficulty(details["difficulty"] + (1 if is_correct else -1))
    details["current_index"] = index + 1
    details["current_problem"] = None

    completed = details["current_index"] >= TOTAL_PROBLEMS
    next_problem_public = None

    if completed:
        details["completed_at"] = submitted_at.isoformat()
        progress.status = "completed"
        progress.completed_at = submitted_at
        progress.score = details["score"]
        session.current_stage = get_next_stage(STAGE)
        db.add(session)
    else:
        problem_data, usage = await _generate_problem(session_id, details, details["current_index"])
        _log_cost(db, session_id, _code_problem_generator.name, usage)
        problem_data["presented_at"] = datetime.now(timezone.utc).isoformat()
        details["current_problem"] = problem_data
        next_problem_public = _public_problem(problem_data)

    progress.details = details
    flag_modified(progress, "details")
    db.add(progress)
    db.commit()
    db.refresh(progress)

    return {
        "is_correct": is_correct,
        "passed_count": passed_count,
        "total_count": total_count,
        "results": _public_results(summary["results"]),
        "score": details["score"],
        "difficulty": details["difficulty"],
        "current_index": min(details["current_index"], TOTAL_PROBLEMS),
        "total_problems": TOTAL_PROBLEMS,
        "response_time_seconds": response_time_seconds,
        "completed": completed,
        "next_problem": next_problem_public,
    }


def get_result(db: Session, session_id: str) -> dict:
    session = _get_session(db, session_id)
    progress = _get_stage_progress(db, session_id)
    if progress is None:
        raise CodingServiceError("Coding round not started for this session", status_code=409)
    if progress.status != "completed":
        raise CodingServiceError("Coding round not completed yet", status_code=409)

    details = progress.details
    history = details["asked_problems"]
    total_time = sum(p["response_time_seconds"] for p in history)

    topic_breakdown: dict[str, dict[str, int]] = {}
    for p in history:
        bucket = topic_breakdown.setdefault(p["topic"], {"correct": 0, "total": 0})
        bucket["total"] += 1
        if p["is_correct"]:
            bucket["correct"] += 1

    return {
        "session_id": session.id,
        "target_company": session.target_company,
        "total_problems": TOTAL_PROBLEMS,
        "score": details["score"],
        "percentage": round(100 * details["score"] / TOTAL_PROBLEMS, 2),
        "difficulty_final": details["difficulty"],
        "difficulty_history": details["difficulty_history"],
        "total_time_seconds": round(total_time, 2),
        "started_at": details.get("started_at"),
        "completed_at": details.get("completed_at"),
        "history": history,
        "topic_breakdown": topic_breakdown,
    }
