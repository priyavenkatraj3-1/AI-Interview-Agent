"""
Integration test: drives all 20 synthetic personas through the real
four-round API (see evaluation/harness.py), using the same `client`
fixture (and its autouse fake generators) as every other API test in this
suite — no new mocking is introduced here. Also runs the grading-variance
check and generates the Evaluation Report + plot as a side effect: this is
the "run the evaluation" step of the mandatory evaluation framework,
executed as a pytest test so it reuses the existing test infrastructure
exactly, rather than a separate untested script.
"""
import json
from pathlib import Path

import pytest

from agents.code_problem_generator.code_problem_generator import CodeProblemGeneratorAgent
from agents.config import MODEL_CHEAP, MODEL_STRONG
from agents.hr_interviewer.grader import MockHRKeyValidatorAgent
from agents.hr_interviewer.hr_interviewer import HRInterviewerAgent
from agents.question_generator.question_generator import QuestionGeneratorAgent
from agents.technical_interviewer.grader import MockTechnicalKeyValidatorAgent
from agents.technical_interviewer.technical_interviewer import TechnicalInterviewerAgent
from app.models.session import CostLog
from app.services import aptitude_service, coding_service, hr_service, technical_service
from evaluation.harness import simulate_persona
from evaluation.personas import PERSONAS
from evaluation.plot import plot_difficulty_vs_ability
from evaluation.report import render_report
from evaluation.variance import grade_fixed_answer_n_times

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"
PLOT_PATH = DOCS_DIR / "evaluation_plot.png"
REPORT_PATH = DOCS_DIR / "evaluation_report.md"
DATA_PATH = DOCS_DIR / "evaluation_data.json"

# Fixed inputs for the grading-variance check (section 3 of the report) —
# arbitrary but must be internally consistent (matches each fake's rubric).
# This exercises phase 2 (key-based validation) with a fixed phase-1 draft
# held constant, since grader independence (see
# tests/test_grader_independence.py) means phase 1 and phase 2 are now
# separate agents — variance of the FINAL score is what this measures,
# and phase 2's keyword-overlap formula is unaffected by the draft's
# exact wording/score.
TECHNICAL_GRADER_FIXED_KWARGS = {
    "question": "Explain the difference between a stack and a queue.",
    "model_answer": "A stack is LIFO; a queue is FIFO.",
    "rubric_keywords": ["stack", "queue", "lifo", "fifo"],
    "candidate_answer": "A stack is LIFO and a queue is FIFO.",
    "draft_score": 90,
    "draft_feedback": "Independent draft: looks solid.",
}
HR_GRADER_FIXED_KWARGS = {
    "question": "Describe a time you had a conflict with a teammate and how you resolved it.",
    "model_answer": "A strong answer describes the conflict, communication, compromise, and resolution.",
    "rubric_keywords": ["conflict", "communicate", "compromise", "resolve"],
    "candidate_answer": "I had a conflict, we talked to communicate, found a compromise, and resolved it.",
    "draft_score": 90,
    "draft_feedback": "Independent draft: looks solid.",
}


def test_no_real_claude_backed_generator_is_active_during_the_pytest_run():
    # Confirms no live Claude call can happen anywhere in this evaluation.
    # tests/conftest.py's autouse fixtures override ALL FOUR rounds'
    # question/problem generators with deterministic test doubles (this is
    # the existing, established pattern every other API test in this suite
    # already relies on — not something new added for this evaluation).
    # Aptitude has no MOCK_MODE feature at all, so its Fake is the only way
    # it can ever run offline; Coding/Technical/HR do have a real MOCK_MODE
    # fixture-bank agent, but conftest.py fakes them too for pytest speed
    # and determinism (see FakeCodeProblemGenerator etc. in conftest.py).
    assert not isinstance(aptitude_service._question_generator, QuestionGeneratorAgent)
    assert not isinstance(coding_service._code_problem_generator, CodeProblemGeneratorAgent)
    assert not isinstance(technical_service._technical_interviewer, TechnicalInterviewerAgent)
    assert not isinstance(hr_service._hr_interviewer, HRInterviewerAgent)


def test_technical_and_hr_key_validators_are_on_the_real_mock_mode_agent():
    # Unlike the generators above, conftest.py does NOT fake the graders —
    # these go through the app's actual MOCK_MODE selection
    # (build_technical_key_validator() / build_hr_key_validator()), which
    # resolves to the offline keyword-heuristic validator here since
    # MOCK_MODE defaults to true (no Anthropic credits available in this
    # environment). The independent (phase-1) graders are likewise real
    # MOCK_MODE agents — see tests/test_grader_independence.py for the
    # dedicated proof that phase 1 never sees the answer key.
    assert isinstance(technical_service._technical_key_validator, MockTechnicalKeyValidatorAgent)
    assert isinstance(hr_service._hr_key_validator, MockHRKeyValidatorAgent)


@pytest.mark.slow  # real subprocess-heavy: 20 personas x coding hidden tests, ~10 min. See pytest.ini.
@pytest.mark.asyncio
async def test_all_twenty_personas_complete_the_loop_and_evaluation_artifacts_are_generated(client, db_session):
    # --- Requirements 1, 2, 4: run all 20 personas through the complete
    # interview loop, collecting per-question difficulty-vs-ability data. ---
    persona_results = [simulate_persona(client, persona) for persona in PERSONAS]

    assert len(persona_results) == 20
    for pr in persona_results:
        assert set(pr.round_scores.keys()) == {"aptitude", "coding", "technical", "hr"}
        assert len(pr.questions) > 0

    # Empirical proof (not just a type check) that no real Claude model was
    # ever billed for any of the 20 personas: every CostLog row written
    # during this run must use a fake/offline model name, never the real
    # configured MODEL_CHEAP/MODEL_STRONG.
    session_ids = {pr.session_id for pr in persona_results}
    cost_log_models = {
        row.model for row in db_session.query(CostLog).filter(CostLog.session_id.in_(session_ids)).all()
    }
    assert MODEL_CHEAP not in cost_log_models
    assert MODEL_STRONG not in cost_log_models

    # Sanity checks on the REAL adaptive-difficulty mechanism (not
    # fabricated expectations): a much stronger persona should reach the
    # same or a higher average difficulty than a much weaker one.
    high = next(pr for pr in persona_results if pr.persona.name == "all_rounder_high")
    low = next(pr for pr in persona_results if pr.persona.name == "all_rounder_low")
    assert high.avg_difficulty("aptitude") >= low.avg_difficulty("aptitude")
    assert high.avg_difficulty("coding") >= low.avg_difficulty("coding")
    assert high.round_scores["hr"]["percentage"] >= low.round_scores["hr"]["percentage"]

    # --- Requirement 3: grading-variance check on a fixed answer, 5x,
    # reusing the real (mock) phase-2 key validator agents (phase 1's
    # draft is held fixed here -- see TECHNICAL_GRADER_FIXED_KWARGS above). ---
    variance_results = {
        "technical": await grade_fixed_answer_n_times(
            technical_service._technical_key_validator, TECHNICAL_GRADER_FIXED_KWARGS, n=5
        ),
        "hr": await grade_fixed_answer_n_times(hr_service._hr_key_validator, HR_GRADER_FIXED_KWARGS, n=5),
    }
    for round_name, vr in variance_results.items():
        assert len(vr.scores) == 5
        assert vr.variance == 0.0, f"{round_name} grading should be deterministic in MOCK_MODE"

    # --- Requirements 5 & 6: difficulty-vs-ability plot. ---
    points = [
        {"round": q.round, "ability": pr.persona.ability_by_round[q.round], "difficulty": q.difficulty}
        for pr in persona_results
        for q in pr.questions
    ]
    plot_difficulty_vs_ability(points, PLOT_PATH)
    assert PLOT_PATH.exists()
    assert PLOT_PATH.stat().st_size > 0

    # --- Requirement 7: Evaluation Report + raw backing data. ---
    report_markdown = render_report(persona_results, variance_results, "evaluation_plot.png")
    REPORT_PATH.write_text(report_markdown, encoding="utf-8")
    assert REPORT_PATH.exists()
    assert "MOCK" in report_markdown  # the offline-evaluation disclosure must be present

    raw_data = {
        "personas": [
            {
                "name": pr.persona.name,
                "description": pr.persona.description,
                "abilities": pr.persona.ability_by_round,
                "target_company": pr.persona.target_company,
                "round_scores": pr.round_scores,
                "questions": [
                    {
                        "round": q.round,
                        "topic": q.topic,
                        "pattern": q.pattern,
                        "difficulty": q.difficulty,
                        "is_correct": q.is_correct,
                    }
                    for q in pr.questions
                ],
            }
            for pr in persona_results
        ],
        "variance": {
            round_name: {"scores": vr.scores, "mean": vr.mean, "variance": vr.variance}
            for round_name, vr in variance_results.items()
        },
    }
    DATA_PATH.write_text(json.dumps(raw_data, indent=2), encoding="utf-8")
    assert DATA_PATH.exists()
