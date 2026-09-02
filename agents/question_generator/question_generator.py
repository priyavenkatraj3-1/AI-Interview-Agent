"""
QuestionGeneratorAgent: generates stage-specific interview questions
(aptitude / coding / technical) on demand via the Claude API.

Day 2: implements the aptitude MCQ case. No hardcoded question bank and no
scraping — every question is generated per-request by prompting Claude,
constrained to structured JSON by forcing a tool call (the pinned SDK
version, anthropic==0.42.0, predates the `output_config.format` structured
output feature, so a forced `tool_choice` is the reliable structured-output
mechanism available here). The result is always parsed from `tool_use.input`,
never scraped out of free text.
"""
from difflib import SequenceMatcher

import anthropic

from agents.base import AgentResult, AgentUsage, BaseAgent, ModelTier
from agents.config import ANTHROPIC_API_KEY
from agents.model_router import resolve_model
from agents.pricing import estimate_cost_usd

from .taxonomy import DIFFICULTY_LABELS

EMIT_QUESTION_TOOL_NAME = "emit_aptitude_question"

EMIT_QUESTION_TOOL = {
    "name": EMIT_QUESTION_TOOL_NAME,
    "description": "Return exactly one structured multiple-choice aptitude question.",
    "input_schema": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question stem, self-contained and unambiguous.",
            },
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 4,
                "maxItems": 4,
                "description": "Exactly 4 answer options, in order.",
            },
            "correct_option": {
                "type": "integer",
                "minimum": 0,
                "maximum": 3,
                "description": "0-based index into options of the single correct answer.",
            },
            "explanation": {
                "type": "string",
                "description": "Brief (1-3 sentence) explanation of why the correct option is correct.",
            },
        },
        "required": ["question", "options", "correct_option", "explanation"],
    },
    # This tool definition is byte-identical on every call to this agent
    # (it never varies with target_company/topic/pattern/difficulty), so
    # it's the one genuinely cacheable static block here. Dynamic content
    # (topic, difficulty, previously-asked questions) stays in the
    # per-call user message, outside this cached block.
    "cache_control": {"type": "ephemeral"},
}

# Malformed/duplicate generations are retried up to this many times before
# the caller just accepts the last attempt, so one bad call can't loop
# indefinitely or blow past the per-session cost budget.
DEFAULT_MAX_ATTEMPTS = 3

# SequenceMatcher ratio at/above which a newly generated question is
# considered too similar to something already asked in this session.
DUPLICATE_SIMILARITY_THRESHOLD = 0.72

# Cap on how many previous-question stems are inlined into the prompt, so
# the request stays small even late in a 15-question session.
MAX_PREVIOUS_QUESTIONS_IN_PROMPT = 15


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def is_similar(a: str, b: str, threshold: float = DUPLICATE_SIMILARITY_THRESHOLD) -> bool:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio() >= threshold


class QuestionGenerationError(Exception):
    """Raised when Claude still returns a malformed question after every retry
    attempt — never returned as if it were a usable question."""


class QuestionGeneratorAgent(BaseAgent):
    """Produces one structured MCQ per call for a given topic/pattern/difficulty,
    avoiding repetition of previously asked questions within the same session."""

    name = "question_generator"
    default_tier = ModelTier.CHEAP

    def __init__(self, client: anthropic.Anthropic | None = None):
        self._client = client

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        return self._client

    def _build_prompt(
        self,
        target_company: str,
        topic: str,
        pattern: str,
        difficulty: int,
        previous_questions: list[str],
    ) -> str:
        label = DIFFICULTY_LABELS[difficulty]
        avoid = ""
        if previous_questions:
            joined = "\n".join(f"- {q}" for q in previous_questions[-MAX_PREVIOUS_QUESTIONS_IN_PROMPT:])
            avoid = (
                "\nDo not repeat, rephrase, or closely resemble any of these "
                f"questions already asked in this session:\n{joined}\n"
            )
        return (
            "You are generating one original multiple-choice aptitude question "
            f"for a placement test in the style of {target_company.replace('_', ' ')}.\n"
            f"Topic: {topic.replace('_', ' ')}. Pattern/sub-type: {pattern.replace('_', ' ')}.\n"
            f"Difficulty: {label} ({difficulty}/5).\n"
            "Requirements:\n"
            "- Exactly 4 answer options, exactly one of which is correct.\n"
            "- The question must be solvable from its own text alone (no images, no "
            "external references).\n"
            "- Keep the question and options concise.\n"
            f"{avoid}"
            f"Call the {EMIT_QUESTION_TOOL_NAME} tool with the result. Do not include any other text."
        )

    async def run(self, **kwargs) -> AgentResult:
        target_company: str = kwargs["target_company"]
        topic: str = kwargs["topic"]
        pattern: str = kwargs["pattern"]
        difficulty: int = kwargs["difficulty"]
        previous_questions: list[str] = kwargs.get("previous_questions", [])
        max_attempts: int = kwargs.get("max_attempts", DEFAULT_MAX_ATTEMPTS)
        tier: ModelTier = kwargs.get("tier", self.default_tier)
        model = resolve_model(tier)

        usage: AgentUsage | None = None
        data: dict | None = None
        malformed = True

        for _attempt in range(1, max_attempts + 1):
            prompt = self._build_prompt(target_company, topic, pattern, difficulty, previous_questions)
            response = self.client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
                tools=[EMIT_QUESTION_TOOL],
                tool_choice={"type": "tool", "name": EMIT_QUESTION_TOOL_NAME},
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

            malformed = (
                not candidate.get("question")
                or len(candidate.get("options", [])) != 4
                or not (0 <= candidate.get("correct_option", -1) <= 3)
                or not candidate.get("explanation")
            )
            duplicate = not malformed and any(
                is_similar(candidate["question"], prev) for prev in previous_questions
            )
            if not malformed and not duplicate:
                break

        if malformed:
            raise QuestionGenerationError(
                f"Claude returned a malformed question after {max_attempts} attempt(s) "
                f"(topic={topic!r}, pattern={pattern!r}): {data!r}"
            )

        data["topic"] = topic
        data["pattern"] = pattern
        data["difficulty"] = difficulty
        return AgentResult(data=data, usage=usage)
