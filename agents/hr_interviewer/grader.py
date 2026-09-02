"""
Two-phase grading for the HR round's free-text answers, so the grading
pipeline satisfies the requirement: "the grader must never see the
generator's answer key until after it has scored independently." Mirrors
agents/technical_interviewer/grader.py's shape exactly.

Phase 1 — IndependentHRGraderAgent / MockIndependentHRGraderAgent:
    receives ONLY `question` and `candidate_answer` (see
    IndependentGradingInput below — a frozen dataclass with no
    `model_answer` / `rubric_keywords` field at all, so the answer key
    cannot even be constructed into this phase's input; passing either as
    a keyword argument raises TypeError). Produces a draft score/feedback
    using nothing but the question and the candidate's own answer.

Phase 2 — HRKeyValidatorAgent / MockHRKeyValidatorAgent:
    receives the phase-1 draft PLUS the generator's `model_answer` /
    `rubric_keywords`, and produces the final score/feedback. This is the
    only point in the pipeline where the answer key is used, and
    app.services.hr_service.submit_answer() always calls phase 1 to
    completion before phase 2 ever runs (see that module, and
    tests/test_grader_independence.py for the data-flow proof).
"""
from dataclasses import dataclass

import anthropic

from agents.base import AgentResult, AgentUsage, BaseAgent, ModelTier
from agents.config import ANTHROPIC_API_KEY, MOCK_MODE
from agents.model_router import resolve_model
from agents.pricing import estimate_cost_usd

from .taxonomy import PASS_THRESHOLD

# --- Phase 1: independent, answer-key-blind assessment ---


@dataclass(frozen=True)
class IndependentGradingInput:
    """Everything phase 1 is allowed to see. Deliberately has no
    `model_answer` / `rubric_keywords` field — constructing one with
    either raises TypeError (unexpected keyword argument), which is the
    structural guarantee that the answer key cannot reach phase 1. See
    tests/test_grader_independence.py."""

    question: str
    candidate_answer: str


EMIT_DRAFT_TOOL_NAME = "emit_hr_independent_draft"

EMIT_DRAFT_TOOL = {
    "name": EMIT_DRAFT_TOOL_NAME,
    "description": (
        "Return an independent draft grade for a candidate's HR/behavioral answer, based only on the "
        "question and the answer itself — no model answer or rubric has been provided."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "draft_score": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
                "description": (
                    "Your own independent judgement of how well the answer demonstrates the qualities a "
                    "strong behavioral answer should have, 0-100, using only your own judgement."
                ),
            },
            "draft_feedback": {
                "type": "string",
                "description": "Brief (1-2 sentence) independent feedback.",
            },
        },
        "required": ["draft_score", "draft_feedback"],
    },
    # Byte-identical on every call — the one genuinely cacheable static
    # block. The candidate's question/answer stay in the per-call user
    # message, outside this cached block.
    "cache_control": {"type": "ephemeral"},
}

DEFAULT_MAX_ATTEMPTS = 3


class HRGradingError(Exception):
    """Raised when Claude still returns a malformed grade (draft or final)
    after every retry attempt — never returned as if it were usable."""


def _is_draft_malformed(candidate: dict) -> bool:
    score = candidate.get("draft_score")
    if not isinstance(score, int) or isinstance(score, bool) or not (0 <= score <= 100):
        return True
    if not candidate.get("draft_feedback"):
        return True
    return False


class IndependentHRGraderAgent(BaseAgent):
    """Phase 1: produces a draft grade using ONLY the question and the
    candidate's answer. Its `run()` reads exactly `question` and
    `candidate_answer` out of kwargs and constructs an
    IndependentGradingInput from them — any other kwargs the caller might
    (mistakenly) pass are never read, and could not be represented in that
    type even if someone tried."""

    name = "hr_independent_grader"
    default_tier = ModelTier.STRONG

    def __init__(self, client: anthropic.Anthropic | None = None):
        self._client = client

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        return self._client

    def _build_prompt(self, grading_input: IndependentGradingInput) -> str:
        return (
            "You are independently grading a candidate's answer in an HR/behavioral placement interview.\n"
            f"Question: {grading_input.question}\n"
            f"Candidate's answer: {grading_input.candidate_answer}\n"
            "You have NOT been given a model answer or grading rubric — assess the answer's substance, "
            "specificity, and professionalism using only your own judgement. Score 0-100.\n"
            f"Call the {EMIT_DRAFT_TOOL_NAME} tool with the result. Do not include any other text."
        )

    async def run(self, **kwargs) -> AgentResult:
        grading_input = IndependentGradingInput(
            question=kwargs["question"], candidate_answer=kwargs["candidate_answer"]
        )
        max_attempts: int = kwargs.get("max_attempts", DEFAULT_MAX_ATTEMPTS)
        tier: ModelTier = kwargs.get("tier", self.default_tier)
        model = resolve_model(tier)

        usage: AgentUsage | None = None
        data: dict | None = None
        malformed = True

        for _attempt in range(1, max_attempts + 1):
            prompt = self._build_prompt(grading_input)
            response = self.client.messages.create(
                model=model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
                tools=[EMIT_DRAFT_TOOL],
                tool_choice={"type": "tool", "name": EMIT_DRAFT_TOOL_NAME},
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
            malformed = _is_draft_malformed(candidate)
            if not malformed:
                break

        if malformed:
            raise HRGradingError(
                f"Claude returned a malformed draft grade after {max_attempts} attempt(s): {data!r}"
            )

        return AgentResult(
            data={"draft_score": data["draft_score"], "draft_feedback": data["draft_feedback"]}, usage=usage
        )


class MockIndependentHRGraderAgent(BaseAgent):
    """Offline phase 1: a crude, deterministic, answer-key-independent
    heuristic (based only on the candidate_answer's own length — never
    rubric_keywords, never model_answer). Used when MOCK_MODE is enabled
    (the default — no Anthropic API credits currently available). A
    clearly-labeled dev/demo fallback, not a claim of genuine independent
    understanding assessment."""

    name = "hr_independent_grader_mock"
    default_tier = ModelTier.STRONG

    async def run(self, **kwargs) -> AgentResult:
        grading_input = IndependentGradingInput(
            question=kwargs["question"], candidate_answer=kwargs["candidate_answer"]
        )
        word_count = len(grading_input.candidate_answer.split())
        draft_score = min(100, word_count * 8)
        draft_feedback = f"Independent draft (no rubric seen): answer has {word_count} word(s)."
        return AgentResult(data={"draft_score": draft_score, "draft_feedback": draft_feedback}, usage=None)


def build_hr_independent_grader() -> BaseAgent:
    """Return the Claude-backed phase-1 grader, or the offline mock when
    MOCK_MODE is enabled (default; see agents/config.py)."""
    if MOCK_MODE:
        return MockIndependentHRGraderAgent()
    return IndependentHRGraderAgent()


# --- Phase 2: key-based validation (the only phase allowed to see the key) ---

EMIT_GRADE_TOOL_NAME = "emit_hr_grade"

EMIT_GRADE_TOOL = {
    "name": EMIT_GRADE_TOOL_NAME,
    "description": "Return the final structured grade for one candidate's free-text HR/behavioral answer.",
    "input_schema": {
        "type": "object",
        "properties": {
            "score": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
                "description": "How well the candidate's answer covers the expected qualities/elements, 0-100.",
            },
            "feedback": {
                "type": "string",
                "description": "Brief (1-3 sentence) feedback on the candidate's answer.",
            },
        },
        "required": ["score", "feedback"],
    },
    # Byte-identical on every call — the one genuinely cacheable static
    # block. The candidate's answer, model_answer/rubric_keywords, and the
    # phase-1 draft stay in the per-call user message, outside this
    # cached block.
    "cache_control": {"type": "ephemeral"},
}


def _is_malformed(candidate: dict) -> bool:
    score = candidate.get("score")
    if not isinstance(score, int) or isinstance(score, bool) or not (0 <= score <= 100):
        return True
    if not candidate.get("feedback"):
        return True
    return False


class HRKeyValidatorAgent(BaseAgent):
    """Phase 2: receives the phase-1 draft PLUS the generator's
    `model_answer` / `rubric_keywords`, and produces the final grade. This
    is the only agent in the HR-grading pipeline whose `run()` signature
    accepts the answer key, and callers only reach it after phase 1 has
    already returned its draft."""

    name = "hr_key_validator"
    default_tier = ModelTier.STRONG

    def __init__(self, client: anthropic.Anthropic | None = None):
        self._client = client

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        return self._client

    def _build_prompt(
        self,
        question: str,
        model_answer: str,
        rubric_keywords: list[str],
        candidate_answer: str,
        draft_score: int,
        draft_feedback: str,
    ) -> str:
        keywords = ", ".join(rubric_keywords)
        return (
            "You are finalizing the grade for a candidate's answer in an HR/behavioral placement "
            "interview. An independent first-pass assessment has already been made, without access to "
            "the model answer or rubric below.\n"
            f"Question: {question}\n"
            f"Candidate's answer: {candidate_answer}\n"
            f"Independent draft assessment (made without seeing the model answer): {draft_score}/100 — "
            f"{draft_feedback}\n"
            f"What a strong answer covers: {model_answer}\n"
            f"Key qualities/elements a strong answer should mention: {keywords}\n"
            "Now that you can see the model answer/rubric, give the FINAL score 0-100 (judge substance, "
            "specificity, and professionalism, not exact wording). The draft above is your own earlier "
            "first impression, not a fixed anchor — you may confirm or revise it. "
            f"Call the {EMIT_GRADE_TOOL_NAME} tool with the result. Do not include any other text."
        )

    async def run(self, **kwargs) -> AgentResult:
        question: str = kwargs["question"]
        model_answer: str = kwargs["model_answer"]
        rubric_keywords: list[str] = kwargs["rubric_keywords"]
        candidate_answer: str = kwargs["candidate_answer"]
        draft_score: int = kwargs["draft_score"]
        draft_feedback: str = kwargs["draft_feedback"]
        max_attempts: int = kwargs.get("max_attempts", DEFAULT_MAX_ATTEMPTS)
        tier: ModelTier = kwargs.get("tier", self.default_tier)
        model = resolve_model(tier)

        usage: AgentUsage | None = None
        data: dict | None = None
        malformed = True

        for _attempt in range(1, max_attempts + 1):
            prompt = self._build_prompt(
                question, model_answer, rubric_keywords, candidate_answer, draft_score, draft_feedback
            )
            response = self.client.messages.create(
                model=model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
                tools=[EMIT_GRADE_TOOL],
                tool_choice={"type": "tool", "name": EMIT_GRADE_TOOL_NAME},
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
            raise HRGradingError(f"Claude returned a malformed grade after {max_attempts} attempt(s): {data!r}")

        score = data["score"]
        return AgentResult(
            data={"is_correct": score >= PASS_THRESHOLD, "score": score, "feedback": data["feedback"]},
            usage=usage,
        )


class MockHRKeyValidatorAgent(BaseAgent):
    """Offline phase 2: the same keyword-overlap heuristic this project has
    always used for HR grading in MOCK_MODE — now explicitly the
    secondary, key-based validation step that runs strictly after (and
    references, via feedback text) the phase-1 independent draft. Used
    when MOCK_MODE is enabled (the default — no Anthropic API credits
    currently available). A clearly-labeled dev/demo fallback, not a
    substitute for real judgment of behavioral answer quality."""

    name = "hr_key_validator_mock"
    default_tier = ModelTier.STRONG

    async def run(self, **kwargs) -> AgentResult:
        rubric_keywords: list[str] = kwargs["rubric_keywords"]
        candidate_answer: str = kwargs["candidate_answer"]
        draft_score: int = kwargs["draft_score"]
        draft_feedback: str = kwargs["draft_feedback"]

        answer_lower = candidate_answer.lower()
        total = len(rubric_keywords)
        matched = sum(1 for keyword in rubric_keywords if keyword.lower() in answer_lower)

        if total > 0:
            score = round(100 * matched / total)
        else:
            score = 100 if candidate_answer.strip() else 0

        if total:
            feedback = (
                f"Independent draft (no rubric seen): {draft_feedback} | "
                f"Key-based validation: matched {matched}/{total} key quality/element(s)."
            )
        else:
            feedback = f"Independent draft (no rubric seen): {draft_feedback} | No rubric keywords to validate against."

        return AgentResult(
            data={"is_correct": score >= PASS_THRESHOLD, "score": score, "feedback": feedback},
            usage=None,
        )


def build_hr_key_validator() -> BaseAgent:
    """Return the Claude-backed phase-2 validator, or the offline
    keyword-heuristic validator when MOCK_MODE is enabled (default; see
    agents/config.py)."""
    if MOCK_MODE:
        return MockHRKeyValidatorAgent()
    return HRKeyValidatorAgent()
