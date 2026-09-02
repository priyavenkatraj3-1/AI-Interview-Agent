"""
Renders the markdown Evaluation Report from collected persona results,
variance results, and the saved plot. Pure string templating — no
scoring/grading logic lives here (that's aptitude_service / coding_service
/ technical_service / hr_service's job, reused as-is via
evaluation/harness.py).
"""
from __future__ import annotations

from datetime import datetime, timezone

from evaluation.harness import PersonaResult
from evaluation.variance import VarianceResult

FAILURE_MODES = [
    {
        "title": "Aptitude has no MOCK_MODE fallback",
        "detail": (
            "Unlike Coding/Technical/HR, the Aptitude round's `QuestionGeneratorAgent` "
            "(`agents/question_generator/question_generator.py`) always calls the real "
            "Anthropic API — it predates the `MOCK_MODE` pattern introduced for later "
            "rounds. Running this evaluation without Claude credits was only possible "
            "by reusing the same `FakeQuestionGenerator` test double the pytest suite "
            "already relies on (`tests/conftest.py`). **What was attempted:** no code "
            "change was made to add `MOCK_MODE` support to Aptitude — that would be out "
            "of scope for an evaluation task and would risk the explicit instruction not "
            "to modify the existing four rounds. The evaluation harness instead works "
            "around the gap at the test layer, exactly as the existing pytest suite "
            "already does for every Aptitude test."
        ),
    },
    {
        "title": "Mock keyword-overlap grading (Technical/HR) is gameable — now scoped to phase 2 only",
        "detail": (
            "Grading was reworked into two phases to satisfy the grader-independence "
            "requirement ('the grader must never see the generator's answer key until "
            "after it has scored independently'): phase 1 "
            "(`MockIndependentTechnicalGraderAgent` / `MockIndependentHRGraderAgent`) never "
            "receives `model_answer`/`rubric_keywords` at all — see "
            "`tests/test_grader_independence.py` for the structural and behavioral proof. "
            "The FINAL score, however, still comes from phase 2 "
            "(`MockTechnicalKeyValidatorAgent` / `MockHRKeyValidatorAgent`, "
            "`agents/technical_interviewer/grader.py`, `agents/hr_interviewer/grader.py`), "
            "which scores by how many rubric keywords a free-text answer contains as "
            "substrings — an answer that stuffs in every rubric keyword with no real "
            "content still scores 100%, identical to a genuinely strong answer, "
            "demonstrated empirically in "
            "`tests/test_evaluation_variance.py::test_keyword_stuffed_nonsense_answer_scores_full_marks_in_mock_mode`. "
            "This is now a narrower, explicitly-scoped limitation of phase 2's offline "
            "heuristic specifically (the requirement explicitly allows the key to be used "
            "for secondary validation at this stage) — not present in the real "
            "Claude-backed phase 2. **What was attempted:** phase 2's keyword heuristic "
            "was not redesigned — it exists only as a small dev/demo fallback, and its "
            "scope is now explicit and bounded (secondary validation, after independent "
            "scoring) rather than the sole grading mechanism. This evaluation's mock-mode "
            "scores are still not over-trusted as a proxy for genuine answer quality."
        ),
    },
    {
        "title": "Mock fixture question text is difficulty-invariant",
        "detail": (
            "`MockCodeProblemGeneratorAgent` / `MockTechnicalInterviewerAgent` / "
            "`MockHRInterviewerAgent` all return one fixed fixture's text unchanged "
            "regardless of the requested difficulty — only the numeric `difficulty` "
            "label attached to the response changes. So the difficulty-vs-ability plot "
            "in this (mock) evaluation reflects how responsively the adaptive-difficulty "
            "*number* tracks a persona's correctness, not genuine content-hardness "
            "scaling, which only the real Claude-generated path provides. **What was "
            "attempted:** no fix was attempted — the mock fixture banks are documented "
            "(in each generator module's own docstring) as a small, hand-verified, fixed "
            "test double; varying fixture text by difficulty would defeat their purpose "
            "as a reproducible offline fallback."
        ),
    },
]


def _persona_table(persona_results: list[PersonaResult]) -> list[str]:
    lines = ["| Persona | Description | Aptitude | Coding | Technical | HR | Company |",
             "|---|---|---|---|---|---|---|"]
    for pr in persona_results:
        p = pr.persona
        lines.append(
            f"| {p.name} | {p.description} | {p.aptitude_ability} | {p.coding_ability} | "
            f"{p.technical_ability} | {p.hr_ability} | {p.target_company} |"
        )
    return lines


def _scores_table(persona_results: list[PersonaResult]) -> list[str]:
    lines = ["| Persona | Aptitude % | Coding % | Technical % | HR % |", "|---|---|---|---|---|"]
    for pr in persona_results:
        rs = pr.round_scores
        lines.append(
            f"| {pr.persona.name} | {rs['aptitude']['percentage']} | {rs['coding']['percentage']} | "
            f"{rs['technical']['percentage']} | {rs['hr']['percentage']} |"
        )
    return lines


def _variance_table(variance_results: dict[str, VarianceResult]) -> list[str]:
    lines = ["| Round | Scores (5 runs) | Mean | Variance |", "|---|---|---|---|"]
    for round_name, vr in variance_results.items():
        lines.append(f"| {round_name} | {vr.scores} | {vr.mean} | {vr.variance} |")
    return lines


def render_report(
    persona_results: list[PersonaResult],
    variance_results: dict[str, VarianceResult],
    plot_relative_path: str,
) -> str:
    lines: list[str] = []
    lines.append("# Evaluation Report")
    lines.append("")
    lines.append(f"_Generated: {datetime.now(timezone.utc).isoformat()}_")
    lines.append("")
    lines.append("## Evaluation mode: fully OFFLINE / MOCK — no live Claude calls")
    lines.append("")
    lines.append(
        "Every result in this report was produced offline (Anthropic API credits are "
        "unavailable in this environment). Two distinct offline mechanisms are in play, "
        "and it matters which is which: **(a)** `tests/conftest.py`'s autouse fixtures "
        "override all four rounds' question/problem *generators* with deterministic test "
        "doubles (`FakeQuestionGenerator`, `FakeCodeProblemGenerator`, "
        "`FakeTechnicalInterviewer`, `FakeHRInterviewer`) — the same test doubles every "
        "other API test in this suite already relies on, not something added for this "
        "evaluation, and not the app's `MOCK_MODE` feature. **(b)** The Technical/HR "
        "*graders* are NOT faked by conftest.py — they go through the app's real, "
        "documented `MOCK_MODE` selection (`build_technical_independent_grader()` / "
        "`build_technical_key_validator()` and their HR equivalents), which resolves to "
        "the offline heuristic agents since `MOCK_MODE=true` here. Grading itself is a "
        "two-phase pipeline — an answer-key-blind independent draft, then a separate "
        "key-based validation step — so the grader never sees the generator's answer key "
        "until after it has scored independently; see "
        "`tests/test_grader_independence.py` for the proof. Both phases are verified "
        "directly in `tests/test_evaluation_harness.py`. **No live Claude API call occurs anywhere "
        "in this report.** Where this environment's constraints limit what could be "
        "measured (e.g. real LLM grading variance), that is stated explicitly below "
        "rather than substituted with a fabricated number."
    )
    lines.append("")

    lines.append("## 1. Personas")
    lines.append("")
    lines.append(
        f"{len(persona_results)} synthetic personas, each with a fixed 4-round ability "
        "profile (probability of a correct/strong answer per round, used as a per-question "
        "trial — a deliberate simplification, not a full IRT model). See `evaluation/personas.py`."
    )
    lines.append("")
    lines.extend(_persona_table(persona_results))
    lines.append("")

    lines.append("## 2. Round-wise results per persona")
    lines.append("")
    lines.append(
        "Every score below comes directly from each round's own `get_result()` — no "
        "scoring logic is reimplemented in the evaluation framework."
    )
    lines.append("")
    lines.extend(_scores_table(persona_results))
    lines.append("")

    lines.append("## 3. Grading score variance (same fixed answer, graded 5x)")
    lines.append("")
    lines.append(
        "Each round's real grader agent was called 5 times on one unchanged fixed "
        "answer. In MOCK_MODE, grading is a pure deterministic function of its inputs "
        "(keyword-overlap for Technical/HR) — **variance == 0.0 is the expected and "
        "correct result here, not a bug.** This does NOT measure real Claude grading "
        "variance, which would require live API credits that are unavailable in this "
        "environment."
    )
    lines.append("")
    lines.extend(_variance_table(variance_results))
    lines.append("")

    lines.append("## 4. Difficulty vs ability")
    lines.append("")
    lines.append(f"![Difficulty vs ability]({plot_relative_path})")
    lines.append("")
    lines.append(
        "Scatter of every answered question's (persona ability, difficulty at "
        "presentation), one series per round, generated with matplotlib "
        "(`evaluation/plot.py`). A positive within-round trend demonstrates the "
        "existing adaptive-difficulty mechanism (`clamp_difficulty` in each round's "
        "taxonomy module) responding correctly to a persona's simulated correctness — "
        "this evaluation did not require any change to that mechanism."
    )
    lines.append("")

    lines.append("## 5. Top 3 failure modes")
    lines.append("")
    for i, fm in enumerate(FAILURE_MODES, start=1):
        lines.append(f"### {i}. {fm['title']}")
        lines.append("")
        lines.append(fm["detail"])
        lines.append("")

    return "\n".join(lines)
