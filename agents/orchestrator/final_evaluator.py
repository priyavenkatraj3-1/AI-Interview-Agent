"""
FinalEvaluatorAgent: synthesizes a consolidated final evaluation (overall
score, per-round scores, strengths, weaknesses, a hiring recommendation,
a 14-day personalised remediation plan, and a one-paragraph hiring
verdict) from the four completed stage results (aptitude, coding,
technical, hr).

This is the orchestrator's "aggregate per-stage results into the final
scorecard ... and verdict" responsibility (see orchestrator.py's
docstring), implemented as its own module rather than inside the
(currently unused) Orchestrator class, since every stage so far is driven
directly by its own service module rather than through that class — this
follows the same actual pattern.

Mirrors the MOCK_MODE factory pattern established by
agents/technical_interviewer/grader.py and agents/hr_interviewer/grader.py:
a real Claude-backed agent for genuine qualitative synthesis, and a
deterministic MockFinalEvaluatorAgent (threshold-based, no Claude call)
used instead whenever MOCK_MODE is enabled — the default, since Anthropic
API credits are currently unavailable.
"""
import anthropic

from agents.base import AgentResult, AgentUsage, BaseAgent, ModelTier
from agents.config import ANTHROPIC_API_KEY, MOCK_MODE
from agents.model_router import resolve_model
from agents.pricing import estimate_cost_usd

EMIT_EVALUATION_TOOL_NAME = "emit_final_evaluation"

RECOMMENDATIONS = [
    "Strongly Recommend",
    "Recommend",
    "Recommend with Reservations",
    "Do Not Recommend",
]

REMEDIATION_PLAN_DAYS = 14

EMIT_EVALUATION_TOOL = {
    "name": EMIT_EVALUATION_TOOL_NAME,
    "description": "Return a structured final evaluation summarizing the candidate's performance across all rounds.",
    "input_schema": {
        "type": "object",
        "properties": {
            "strengths": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 6,
                "description": "Specific, evidence-based strengths observed across the rounds.",
            },
            "weaknesses": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 6,
                "description": "Specific, evidence-based areas for improvement observed across the rounds.",
            },
            "recommendation": {
                "type": "string",
                "enum": RECOMMENDATIONS,
                "description": "Overall hiring recommendation based on performance across all four rounds.",
            },
            "summary": {
                "type": "string",
                "description": "A short (2-4 sentence) overall evaluation summary.",
            },
            "remediation_plan": {
                "type": "array",
                "minItems": REMEDIATION_PLAN_DAYS,
                "maxItems": REMEDIATION_PLAN_DAYS,
                "items": {
                    "type": "object",
                    "properties": {
                        "day": {"type": "integer", "minimum": 1, "maximum": REMEDIATION_PLAN_DAYS},
                        "focus": {"type": "string", "description": "Short focus area for this day."},
                        "action": {"type": "string", "description": "A concrete, specific action to take this day."},
                    },
                    "required": ["day", "focus", "action"],
                },
                "description": (
                    "Exactly 14 day-by-day entries personalised to this candidate's specific weak topics/rounds "
                    "(or, if none, their strong areas) — reference the actual topics/rounds from the data above, "
                    "never a generic identical plan."
                ),
            },
            "hiring_verdict": {
                "type": "string",
                "description": (
                    "A complete one-paragraph verdict directly answering 'would this student get hired?', with "
                    "reasoning that cites this candidate's actual overall score and specific strong/weak rounds "
                    "or topics — not a generic template that would read the same for a different candidate."
                ),
            },
        },
        "required": ["strengths", "weaknesses", "recommendation", "summary", "remediation_plan", "hiring_verdict"],
    },
    # Byte-identical on every call — the one genuinely cacheable static
    # block. The candidate's overall_score/rounds data stays in the
    # per-call user message, outside this cached block.
    "cache_control": {"type": "ephemeral"},
}

# Mirrors every other generation agent's retry/backoff-free bound: a bad
# call can't loop indefinitely or blow past the per-session cost budget.
DEFAULT_MAX_ATTEMPTS = 3

# Overall-score thresholds the mock evaluator uses to pick a recommendation.
STRONGLY_RECOMMEND_THRESHOLD = 75
RECOMMEND_THRESHOLD = 60
RECOMMEND_WITH_RESERVATIONS_THRESHOLD = 40

# A round/topic percentage at/above this is called out as a strength;
# at/below this (WEAKNESS_THRESHOLD) is called out as a weakness.
STRENGTH_THRESHOLD = 75
WEAKNESS_THRESHOLD = 50

_STAGE_DISPLAY_NAMES = {
    "aptitude": "Aptitude",
    "coding": "Coding",
    "technical": "Technical",
    "hr": "HR",
}

_VERDICT_LEAD_BY_RECOMMENDATION = {
    "Strongly Recommend": "Yes — this candidate would very likely get hired.",
    "Recommend": "Yes — this candidate would likely get hired.",
    "Recommend with Reservations": "Possibly — this candidate could be hired, but with reservations.",
    "Do Not Recommend": "No — this candidate would likely not get hired at this stage.",
}

# Cycled by "pass number" (how many times the day-cycle has gone around the
# candidate's list of weak items) so a candidate with few weak areas still
# gets varied day-to-day actions across the 14 days rather than the same
# sentence repeated.
_WEAK_AREA_ACTION_TEMPLATES = [
    "Review core fundamentals of {subject} (currently {pct}%) and solve 3-5 easy/medium practice problems.",
    "Attempt harder, timed practice questions on {subject} to build speed and confidence.",
    "Take a short mock quiz focused on {subject} and carefully review every mistake.",
    "Explain {subject} concepts out loud (or to a peer), then solve 2-3 fresh problems to confirm real understanding.",
]

_STRONG_AREA_ACTION_TEMPLATES = [
    "You're already strong in {subject} ({pct}%) — attempt advanced-level problems to deepen mastery.",
    "Revisit {subject} ({pct}%) under time pressure to make sure your strength holds up in a real interview setting.",
    "Mentor-style self-check: explain {subject} as if teaching someone else, then try one genuinely hard problem.",
]


class FinalEvaluationError(Exception):
    """Raised when Claude still returns a malformed evaluation after every
    retry attempt — never returned as if it were usable."""


def _is_malformed(candidate: dict) -> bool:
    strengths = candidate.get("strengths")
    if not isinstance(strengths, list) or not strengths or not all(
        isinstance(s, str) and s.strip() for s in strengths
    ):
        return True
    weaknesses = candidate.get("weaknesses")
    if not isinstance(weaknesses, list) or not weaknesses or not all(
        isinstance(w, str) and w.strip() for w in weaknesses
    ):
        return True
    if candidate.get("recommendation") not in RECOMMENDATIONS:
        return True
    if not candidate.get("summary"):
        return True

    remediation_plan = candidate.get("remediation_plan")
    if not isinstance(remediation_plan, list) or len(remediation_plan) != REMEDIATION_PLAN_DAYS:
        return True
    for entry in remediation_plan:
        if not isinstance(entry, dict):
            return True
        day = entry.get("day")
        if not isinstance(day, int) or isinstance(day, bool):
            return True
        if not entry.get("focus") or not entry.get("action"):
            return True

    if not candidate.get("hiring_verdict"):
        return True
    return False


def _recommendation_for_score(overall_score: float) -> str:
    if overall_score >= STRONGLY_RECOMMEND_THRESHOLD:
        return "Strongly Recommend"
    if overall_score >= RECOMMEND_THRESHOLD:
        return "Recommend"
    if overall_score >= RECOMMEND_WITH_RESERVATIONS_THRESHOLD:
        return "Recommend with Reservations"
    return "Do Not Recommend"


def _analyze_rounds(rounds: dict) -> tuple[list[dict], list[dict]]:
    """Returns (weak_items, strong_items) in the SAME iteration order the
    original strengths/weaknesses-text logic always used (round-level
    check, then that round's topics, before moving to the next round) —
    each item is {"label": round display name, "topic": topic label or
    None, "pct": float}. Callers that need a "weakest/strongest first"
    view (remediation plan, hiring verdict) sort a copy locally rather
    than changing this order, so existing strengths/weaknesses output is
    unaffected."""
    weak: list[dict] = []
    strong: list[dict] = []

    for stage_name, summary in rounds.items():
        label = _STAGE_DISPLAY_NAMES.get(stage_name, stage_name.capitalize())
        pct = summary["percentage"]
        if pct >= STRENGTH_THRESHOLD:
            strong.append({"label": label, "topic": None, "pct": pct})
        elif pct <= WEAKNESS_THRESHOLD:
            weak.append({"label": label, "topic": None, "pct": pct})

        for topic, stats in summary["topic_breakdown"].items():
            topic_label = topic.replace("_", " ")
            topic_pct = round(100 * stats["correct"] / stats["total"], 2) if stats["total"] else 0.0
            if topic_pct >= STRENGTH_THRESHOLD:
                strong.append({"label": label, "topic": topic_label, "pct": topic_pct})
            elif topic_pct <= WEAKNESS_THRESHOLD:
                weak.append({"label": label, "topic": topic_label, "pct": topic_pct})

        # Coding-round-only: its code-quality result (separate from the
        # functional percentage handled above) gets the same strength/
        # weakness treatment, so the mock evaluator's strengths/weaknesses/
        # remediation-plan/hiring-verdict actually reflect it too, not just
        # the real Claude-backed prompt (see _build_prompt below).
        quality_score = summary.get("average_quality_score")
        if quality_score is not None:
            if quality_score >= STRENGTH_THRESHOLD:
                strong.append({"label": label, "topic": "code quality", "pct": quality_score})
            elif quality_score <= WEAKNESS_THRESHOLD:
                weak.append({"label": label, "topic": "code quality", "pct": quality_score})

    return weak, strong


def _format_strength(item: dict) -> str:
    if item["topic"] is None:
        return f"Strong performance in the {item['label']} round ({item['pct']}%)."
    return f"Solid grasp of '{item['topic']}' in the {item['label']} round ({item['pct']}%)."


def _format_weakness(item: dict) -> str:
    if item["topic"] is None:
        return f"{item['label']} round score was low ({item['pct']}%) — needs more practice."
    return f"'{item['topic']}' in the {item['label']} round needs improvement ({item['pct']}%)."


def _subject_phrase(item: dict) -> str:
    return f"'{item['topic']}' in the {item['label']} round" if item["topic"] else f"the {item['label']} round"


def _build_mock_remediation_plan(weak_items: list[dict], strong_items: list[dict]) -> list[dict]:
    """Builds exactly REMEDIATION_PLAN_DAYS day-by-day entries, personalised
    to this candidate: cycles through their actual identified weak
    topics/rounds (escalating the action's intensity on each lap through
    the list, so even a single weak area gets varied day-to-day actions),
    closing with an integrated review day and a full mock-interview day
    that name the candidate's own weak areas. A candidate with no
    identified weaknesses gets a plan built from their strengths instead
    (still personalised, never a copy of the weak-area plan) plus an
    advanced mock interview and a final confidence-building review."""
    plan: list[dict] = []
    cycle_days = REMEDIATION_PLAN_DAYS - 2

    if weak_items:
        weak_sorted = sorted(weak_items, key=lambda item: item["pct"])
        for day in range(1, cycle_days + 1):
            index = (day - 1) % len(weak_sorted)
            pass_number = (day - 1) // len(weak_sorted)
            item = weak_sorted[index]
            subject = _subject_phrase(item)
            template = _WEAK_AREA_ACTION_TEMPLATES[pass_number % len(_WEAK_AREA_ACTION_TEMPLATES)]
            action = template.format(subject=subject, pct=item["pct"])
            plan.append({"day": day, "focus": subject.capitalize(), "action": action})

        weak_summary = ", ".join(
            dict.fromkeys(_subject_phrase(item) for item in weak_sorted)  # de-duplicated, order preserved
        )
        plan.append(
            {
                "day": REMEDIATION_PLAN_DAYS - 1,
                "focus": "Integrated weak-area review",
                "action": f"Do a combined practice session covering all previously weak areas: {weak_summary}.",
            }
        )
        plan.append(
            {
                "day": REMEDIATION_PLAN_DAYS,
                "focus": "Full mock interview run-through",
                "action": (
                    "Simulate the complete Aptitude, Coding, Technical, and HR flow end-to-end, paying extra "
                    f"attention to: {weak_summary}."
                ),
            }
        )
    elif strong_items:
        strong_sorted = sorted(strong_items, key=lambda item: -item["pct"])
        for day in range(1, cycle_days + 1):
            index = (day - 1) % len(strong_sorted)
            pass_number = (day - 1) // len(strong_sorted)
            item = strong_sorted[index]
            subject = _subject_phrase(item)
            template = _STRONG_AREA_ACTION_TEMPLATES[pass_number % len(_STRONG_AREA_ACTION_TEMPLATES)]
            action = template.format(subject=subject, pct=item["pct"])
            plan.append({"day": day, "focus": subject.capitalize(), "action": action})

        plan.append(
            {
                "day": REMEDIATION_PLAN_DAYS - 1,
                "focus": "Advanced mock interview",
                "action": "Attempt a full-length, higher-difficulty mock interview across all four rounds to simulate real placement conditions.",
            }
        )
        plan.append(
            {
                "day": REMEDIATION_PLAN_DAYS,
                "focus": "Final review and confidence check",
                "action": "Do a light, confidence-building review of your strongest areas and rest before the actual interview.",
            }
        )
    else:
        for day in range(1, cycle_days + 1):
            plan.append(
                {
                    "day": day,
                    "focus": "General readiness practice",
                    "action": "Maintain consistent, balanced practice across Aptitude, Coding, Technical, and HR to stay sharp.",
                }
            )
        plan.append(
            {
                "day": REMEDIATION_PLAN_DAYS - 1,
                "focus": "Full mock interview",
                "action": "Attempt a full-length mock interview across all four rounds to simulate real placement conditions.",
            }
        )
        plan.append(
            {
                "day": REMEDIATION_PLAN_DAYS,
                "focus": "Final review",
                "action": "Do a light review across all four rounds and rest before the actual interview.",
            }
        )

    return plan


def _build_mock_hiring_verdict(
    overall_score: float, recommendation: str, weak_items: list[dict], strong_items: list[dict], rounds: dict
) -> str:
    """A complete one-paragraph verdict citing this candidate's actual
    overall score, per-round percentages, and (when available) their
    single strongest and weakest identified item — genuinely different
    for different performance profiles, not a fixed template string."""
    round_summary = "; ".join(
        f"{_STAGE_DISPLAY_NAMES.get(name, name.capitalize())} {r['percentage']}%" for name, r in rounds.items()
    )

    strength_phrase = ""
    if strong_items:
        strongest = sorted(strong_items, key=lambda item: -item["pct"])[0]
        strength_phrase = f"The candidate showed particular strength in {_subject_phrase(strongest)} ({strongest['pct']}%). "

    weakness_phrase = ""
    if weak_items:
        weakest = sorted(weak_items, key=lambda item: item["pct"])[0]
        weakness_phrase = f"However, {_subject_phrase(weakest)} ({weakest['pct']}%) stood out as an area needing improvement. "

    verdict_lead = _VERDICT_LEAD_BY_RECOMMENDATION[recommendation]

    return (
        f"{verdict_lead} Across an overall score of {overall_score}% ({round_summary}). "
        f"{strength_phrase}{weakness_phrase}"
        f"Overall recommendation: {recommendation}."
    )


class FinalEvaluatorAgent(BaseAgent):
    """Synthesizes strengths/weaknesses/recommendation/remediation-plan/
    hiring-verdict from round summaries via a forced Claude tool call."""

    name = "final_evaluator"
    default_tier = ModelTier.STRONG

    def __init__(self, client: anthropic.Anthropic | None = None):
        self._client = client

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        return self._client

    def _build_prompt(self, target_company: str, overall_score: float, rounds: dict) -> str:
        lines = [
            "You are writing the final evaluation for a candidate who completed a placement interview "
            f"simulation in the style of {target_company.replace('_', ' ')}, across four rounds: "
            "Aptitude, Coding, Technical, and HR.",
            f"Overall score: {overall_score}/100.",
        ]
        for stage_name, summary in rounds.items():
            label = _STAGE_DISPLAY_NAMES.get(stage_name, stage_name.capitalize())
            lines.append(
                f"- {label}: {summary['score']}/{summary['total']} ({summary['percentage']}%). "
                f"Topic breakdown: {summary['topic_breakdown']}"
            )
            quality_score = summary.get("average_quality_score")
            if quality_score is not None:
                # Coding-round-only: its code-quality result, separate from
                # the functional score/percentage above.
                lines.append(
                    f"  Code quality (independent of functional correctness): {quality_score}/100."
                )
        lines.append(
            "Based on this performance, identify specific strengths and weaknesses (grounded in the "
            "data above, not generic), and give an overall hiring recommendation "
            f"(one of: {', '.join(RECOMMENDATIONS)}) with a brief summary. Also produce a personalised "
            f"{REMEDIATION_PLAN_DAYS}-day remediation plan (one entry per day, each with a specific focus "
            "and a concrete action) built around this candidate's actual weak topics/rounds (or, if none, "
            "their strengths) — never a generic plan that would read the same for a different candidate. "
            "Finally, write a complete one-paragraph hiring verdict directly answering 'would this student "
            "get hired?', citing this candidate's actual overall score and specific strong/weak rounds or "
            "topics as reasoning. "
            f"Call the {EMIT_EVALUATION_TOOL_NAME} tool with the result. Do not include any other text."
        )
        return "\n".join(lines)

    async def run(self, **kwargs) -> AgentResult:
        target_company: str = kwargs["target_company"]
        overall_score: float = kwargs["overall_score"]
        rounds: dict = kwargs["rounds"]
        max_attempts: int = kwargs.get("max_attempts", DEFAULT_MAX_ATTEMPTS)
        tier: ModelTier = kwargs.get("tier", self.default_tier)
        model = resolve_model(tier)

        usage: AgentUsage | None = None
        data: dict | None = None
        malformed = True

        for _attempt in range(1, max_attempts + 1):
            prompt = self._build_prompt(target_company, overall_score, rounds)
            response = self.client.messages.create(
                model=model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
                tools=[EMIT_EVALUATION_TOOL],
                tool_choice={"type": "tool", "name": EMIT_EVALUATION_TOOL_NAME},
            )
            tool_block = next(b for b in response.content if b.type == "tool_use")
            candidate = dict(tool_block.input)
            usage = AgentUsage(
                model=model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cost_usd=estimate_cost_usd(model, response.usage.input_tokens, response.usage.output_tokens),
            )
            data = candidate
            malformed = _is_malformed(candidate)
            if not malformed:
                break

        if malformed:
            raise FinalEvaluationError(
                f"Claude returned a malformed final evaluation after {max_attempts} attempt(s): {data!r}"
            )

        return AgentResult(data=data, usage=usage)


class MockFinalEvaluatorAgent(BaseAgent):
    """Deterministic, fully offline stand-in for FinalEvaluatorAgent: derives
    strengths/weaknesses/remediation-plan/hiring-verdict from round/topic
    score thresholds and picks a recommendation from the overall score —
    no Claude call. Used when MOCK_MODE is enabled (the default — no
    Anthropic API credits currently available). A clearly-labeled dev/demo
    fallback, not a substitute for genuine qualitative synthesis."""

    name = "final_evaluator_mock"
    default_tier = ModelTier.STRONG

    async def run(self, **kwargs) -> AgentResult:
        overall_score: float = kwargs["overall_score"]
        rounds: dict = kwargs["rounds"]

        weak_items, strong_items = _analyze_rounds(rounds)

        strengths = [_format_strength(item) for item in strong_items]
        weaknesses = [_format_weakness(item) for item in weak_items]

        if not strengths:
            strengths.append(
                "No standout strengths identified; performance was consistent but unremarkable across rounds."
            )
        if not weaknesses:
            weaknesses.append("No significant weaknesses identified.")

        recommendation = _recommendation_for_score(overall_score)
        summary_text = (
            f"The candidate achieved an overall score of {overall_score}% across the Aptitude, Coding, "
            f"Technical, and HR rounds. Based on this performance, the recommendation is: {recommendation}."
        )

        remediation_plan = _build_mock_remediation_plan(weak_items, strong_items)
        hiring_verdict = _build_mock_hiring_verdict(overall_score, recommendation, weak_items, strong_items, rounds)

        return AgentResult(
            data={
                "strengths": strengths,
                "weaknesses": weaknesses,
                "recommendation": recommendation,
                "summary": summary_text,
                "remediation_plan": remediation_plan,
                "hiring_verdict": hiring_verdict,
            },
            usage=None,
        )


def build_final_evaluator() -> BaseAgent:
    """Return the Claude-backed evaluator, or the offline deterministic
    evaluator when MOCK_MODE is enabled (default; see agents/config.py)."""
    if MOCK_MODE:
        return MockFinalEvaluatorAgent()
    return FinalEvaluatorAgent()
