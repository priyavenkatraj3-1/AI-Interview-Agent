"""
Tests for the aptitude round's backend-enforced per-question timeout
(agents.question_generator.taxonomy.MAX_TIME_PER_QUESTION_SECONDS, checked
in app.services.aptitude_service.submit_answer() before grading).

Simulates "too much time has passed" by rewriting the persisted
current_question's presented_at directly in the DB (via the db_session
fixture), rather than actually sleeping past the limit -- deterministic and
fast, and proves enforcement is driven by a server-side timestamp the
client never supplies (never by anything the frontend does or doesn't do).
"""
from datetime import datetime, timedelta, timezone

from agents.question_generator.taxonomy import (
    MAX_TIME_PER_QUESTION_SECONDS,
    START_DIFFICULTY,
    TOTAL_QUESTIONS,
)
from app.models.session import StageProgress

CORRECT = 0
INCORRECT = 1


def _expire_current_question(db_session, session_id):
    """Rewrites the persisted current_question's presented_at to be far
    enough in the past that MAX_TIME_PER_QUESTION_SECONDS has already
    elapsed by the time the next /answer call is made."""
    from sqlalchemy.orm.attributes import flag_modified

    progress = (
        db_session.query(StageProgress)
        .filter(StageProgress.session_id == session_id, StageProgress.stage == "aptitude")
        .one()
    )
    details = dict(progress.details)
    expired_at = datetime.now(timezone.utc) - timedelta(seconds=MAX_TIME_PER_QUESTION_SECONDS + 5)
    details["current_question"] = {**details["current_question"], "presented_at": expired_at.isoformat()}
    progress.details = details
    flag_modified(progress, "details")
    db_session.add(progress)
    db_session.commit()


def test_answer_within_time_limit_is_not_marked_timed_out(client, session_id):
    client.post(f"/api/sessions/{session_id}/aptitude/start")
    response = client.post(f"/api/sessions/{session_id}/aptitude/answer", json={"selected_option": CORRECT})
    body = response.json()
    assert body["timed_out"] is False
    assert body["is_correct"] is True


def test_timed_out_answer_is_marked_incorrect_even_with_the_correct_option(client, session_id, db_session):
    client.post(f"/api/sessions/{session_id}/aptitude/start")
    _expire_current_question(db_session, session_id)

    response = client.post(f"/api/sessions/{session_id}/aptitude/answer", json={"selected_option": CORRECT})
    assert response.status_code == 200
    body = response.json()

    assert body["timed_out"] is True
    assert body["is_correct"] is False
    assert body["score"] == 0


def test_timeout_decreases_difficulty_like_a_wrong_answer(client, session_id, db_session):
    client.post(f"/api/sessions/{session_id}/aptitude/start")
    _expire_current_question(db_session, session_id)

    body = client.post(f"/api/sessions/{session_id}/aptitude/answer", json={"selected_option": CORRECT}).json()
    assert body["difficulty"] == max(START_DIFFICULTY - 1, 1)


def test_timeout_still_advances_to_the_next_question(client, session_id, db_session):
    client.post(f"/api/sessions/{session_id}/aptitude/start")
    _expire_current_question(db_session, session_id)

    body = client.post(f"/api/sessions/{session_id}/aptitude/answer", json={"selected_option": CORRECT}).json()
    assert body["completed"] is False
    assert body["next_question"] is not None
    assert body["current_index"] == 1  # one question answered so far (this response's own convention)


def test_history_entry_records_timed_out_flag(client, session_id, db_session):
    client.post(f"/api/sessions/{session_id}/aptitude/start")
    _expire_current_question(db_session, session_id)
    client.post(f"/api/sessions/{session_id}/aptitude/answer", json={"selected_option": CORRECT})

    for _ in range(TOTAL_QUESTIONS - 1):
        client.post(f"/api/sessions/{session_id}/aptitude/answer", json={"selected_option": CORRECT})

    history = client.get(f"/api/sessions/{session_id}/aptitude/result").json()["history"]
    assert len(history) == TOTAL_QUESTIONS
    assert history[0]["timed_out"] is True
    assert history[0]["is_correct"] is False
    assert all(entry["timed_out"] is False for entry in history[1:])


def test_round_still_completes_at_exactly_fifteen_questions_with_a_timeout(client, session_id, db_session):
    client.post(f"/api/sessions/{session_id}/aptitude/start")
    _expire_current_question(db_session, session_id)

    last_body = None
    for _ in range(TOTAL_QUESTIONS):
        last_body = client.post(
            f"/api/sessions/{session_id}/aptitude/answer", json={"selected_option": CORRECT}
        ).json()

    assert last_body["completed"] is True
    assert last_body["current_index"] == TOTAL_QUESTIONS
    assert last_body["next_question"] is None

    result = client.get(f"/api/sessions/{session_id}/aptitude/result").json()
    assert len(result["history"]) == TOTAL_QUESTIONS

    session = client.get(f"/api/sessions/{session_id}").json()
    assert session["current_stage"] == "coding"
