"""
Persona simulation harness: drives one persona through the real four-round
API (via the same FastAPI TestClient used by the pytest suite) and
collects per-question (round, topic, pattern, difficulty, is_correct)
records plus each round's final get_result() scores.

This module makes no Claude calls and does no mocking itself — it relies
entirely on the offline agents already wired up by MOCK_MODE (coding,
technical, hr) and on tests/conftest.py's autouse `fake_question_generator`
fixture (aptitude has no MOCK_MODE fallback at all). It must always be
called with a `client` fixture from a pytest test where those fixtures are
active (see tests/test_evaluation_harness.py) — it is NOT meant to be run
against a live, un-mocked backend.

No scoring/grading logic is reimplemented here: every score comes from the
real aptitude_service / coding_service / technical_service / hr_service
endpoints, exactly as a real candidate's session would.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from agents.code_problem_generator.taxonomy import TOTAL_PROBLEMS
from agents.hr_interviewer.taxonomy import TOTAL_QUESTIONS as HR_TOTAL_QUESTIONS
from agents.question_generator.taxonomy import TOTAL_QUESTIONS as APTITUDE_TOTAL_QUESTIONS
from agents.technical_interviewer.taxonomy import TOTAL_QUESTIONS as TECHNICAL_TOTAL_QUESTIONS
from evaluation.personas import Persona

# Must match tests/conftest.py's fake generators exactly — this harness
# doesn't control their fixed content, only which answer to submit.
APTITUDE_CORRECT_OPTION = 0  # FakeQuestionGenerator always sets correct_option=0
CODING_CORRECT_CODE = "def add_two(a, b):\n    return a + b\n"
CODING_INCORRECT_CODE = "def add_two(a, b):\n    return a - b\n"
TEXT_ROUND_KEYWORDS = {
    "technical": ["stack", "queue", "lifo", "fifo"],  # FakeTechnicalInterviewer's fixed rubric
    "hr": ["conflict", "communicate", "compromise", "resolve"],  # FakeHRInterviewer's fixed rubric
}


@dataclass
class QuestionRecord:
    round: str
    topic: str
    pattern: str
    difficulty: int
    is_correct: bool


@dataclass
class PersonaResult:
    persona: Persona
    session_id: str = ""
    questions: list[QuestionRecord] = field(default_factory=list)
    round_scores: dict[str, dict] = field(default_factory=dict)

    def avg_difficulty(self, round_name: str) -> float | None:
        matching = [q.difficulty for q in self.questions if q.round == round_name]
        return sum(matching) / len(matching) if matching else None


def _rng_for(persona: Persona, round_name: str) -> random.Random:
    return random.Random(f"eval:{persona.name}:{round_name}")


def _keyword_answer(rng: random.Random, keywords: list[str], ability: float) -> str:
    """Build a free-text answer including a subset of `keywords` sized
    proportionally to `ability`, so the real keyword-overlap mock grader
    produces a graded score approximately equal to ability * 100."""
    num_include = round(ability * len(keywords))
    if num_include <= 0:
        return "I am not sure how to answer this."
    included = rng.sample(keywords, num_include)
    return "This is my answer: " + ", ".join(included)


def _run_aptitude(client, session_id: str, persona: Persona, records: list[QuestionRecord]) -> dict:
    rng = _rng_for(persona, "aptitude")
    ability = persona.aptitude_ability

    state = client.post(f"/api/sessions/{session_id}/aptitude/start").json()
    for _ in range(APTITUDE_TOTAL_QUESTIONS):
        question = state["question"]
        selected = APTITUDE_CORRECT_OPTION if rng.random() < ability else (APTITUDE_CORRECT_OPTION + 1) % 4
        result = client.post(
            f"/api/sessions/{session_id}/aptitude/answer", json={"selected_option": selected}
        ).json()
        records.append(
            QuestionRecord(
                round="aptitude",
                topic=question["topic"],
                pattern=question["pattern"],
                difficulty=question["difficulty"],
                is_correct=result["is_correct"],
            )
        )
        if result["completed"]:
            break
        state = {**state, "question": result["next_question"]}

    return client.get(f"/api/sessions/{session_id}/aptitude/result").json()


def _run_coding(client, session_id: str, persona: Persona, records: list[QuestionRecord]) -> dict:
    rng = _rng_for(persona, "coding")
    ability = persona.coding_ability

    state = client.post(f"/api/sessions/{session_id}/coding/start").json()
    for _ in range(TOTAL_PROBLEMS):
        problem = state["problem"]
        code = CODING_CORRECT_CODE if rng.random() < ability else CODING_INCORRECT_CODE
        result = client.post(f"/api/sessions/{session_id}/coding/submit", json={"code": code}).json()
        records.append(
            QuestionRecord(
                round="coding",
                topic=problem["topic"],
                pattern=problem["pattern"],
                difficulty=problem["difficulty"],
                is_correct=result["is_correct"],
            )
        )
        if result["completed"]:
            break
        state = {**state, "problem": result["next_problem"]}

    return client.get(f"/api/sessions/{session_id}/coding/result").json()


def _run_text_round(
    client, session_id: str, round_name: str, total_questions: int, ability: float, rng: random.Random,
    records: list[QuestionRecord],
) -> dict:
    keywords = TEXT_ROUND_KEYWORDS[round_name]

    state = client.post(f"/api/sessions/{session_id}/{round_name}/start").json()
    for _ in range(total_questions):
        question = state["question"]
        answer = _keyword_answer(rng, keywords, ability)
        result = client.post(f"/api/sessions/{session_id}/{round_name}/answer", json={"answer": answer}).json()
        records.append(
            QuestionRecord(
                round=round_name,
                topic=question["topic"],
                pattern=question["pattern"],
                difficulty=question["difficulty"],
                is_correct=result["is_correct"],
            )
        )
        if result["completed"]:
            break
        state = {**state, "question": result["next_question"]}

    return client.get(f"/api/sessions/{session_id}/{round_name}/result").json()


def simulate_persona(client, persona: Persona) -> PersonaResult:
    """Create a session and drive `persona` through all four rounds using
    the real API endpoints. Returns per-question records plus each round's
    final get_result() scores."""
    session = client.post("/api/sessions", json={"target_company": persona.target_company}).json()
    session_id = session["id"]

    records: list[QuestionRecord] = []
    result = PersonaResult(persona=persona, session_id=session_id, questions=records)

    aptitude_result = _run_aptitude(client, session_id, persona, records)
    coding_result = _run_coding(client, session_id, persona, records)
    technical_result = _run_text_round(
        client, session_id, "technical", TECHNICAL_TOTAL_QUESTIONS, persona.technical_ability,
        _rng_for(persona, "technical"), records,
    )
    hr_result = _run_text_round(
        client, session_id, "hr", HR_TOTAL_QUESTIONS, persona.hr_ability, _rng_for(persona, "hr"), records
    )

    result.round_scores = {
        "aptitude": {
            "score": aptitude_result["score"],
            "total": aptitude_result["total_questions"],
            "percentage": aptitude_result["percentage"],
        },
        "coding": {
            "score": coding_result["score"],
            "total": coding_result["total_problems"],
            "percentage": coding_result["percentage"],
        },
        "technical": {
            "score": technical_result["score"],
            "total": technical_result["total_questions"],
            "percentage": technical_result["percentage"],
        },
        "hr": {
            "score": hr_result["score"],
            "total": hr_result["total_questions"],
            "percentage": hr_result["percentage"],
        },
    }
    return result
