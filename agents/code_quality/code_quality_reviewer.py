"""
CodeQualityReviewerAgent: reviews a candidate's submitted coding-round
solution for code QUALITY -- readability, naming clarity, structure/
organization, appropriate use of functions, unnecessary duplication, basic
best practices, and maintainability -- entirely separate from functional
correctness (hidden-test pass/fail, graded by agents.grader.grader).

Mirrors agents/code_problem_generator/code_problem_generator.py's shape: a
real Claude-backed agent for genuine qualitative review, forced into
structured output via a tool call, plus a deterministic
MockCodeQualityReviewerAgent (dependency-free heuristics, stdlib only) used
instead whenever MOCK_MODE is enabled (the default -- see agents/config.py).
Both satisfy the same BaseAgent contract, so app.services.coding_service
holds whichever build_code_quality_reviewer() returns without branching on
MOCK_MODE itself.

Only ever given the candidate's own code plus the problem's title/
description/function_name -- never hidden_tests/public_tests/examples, so
there is nothing test-related to leak to (or through) this agent.
"""
import re

import anthropic

from agents.base import AgentResult, AgentUsage, BaseAgent, ModelTier
from agents.config import ANTHROPIC_API_KEY, MOCK_MODE
from agents.model_router import resolve_model
from agents.pricing import estimate_cost_usd

# The seven quality dimensions this review must cover, per the coding-round
# requirement: readability, naming clarity, structure/organization,
# appropriate use of functions, unnecessary duplication, basic best
# practices, and maintainability. Single source of truth for both the real
# tool schema and the mock heuristic's output shape.
QUALITY_DIMENSIONS = [
    "readability",
    "naming_clarity",
    "structure_organization",
    "function_usage",
    "duplication",
    "best_practices",
    "maintainability",
]

EMIT_QUALITY_REVIEW_TOOL_NAME = "emit_code_quality_review"

EMIT_QUALITY_REVIEW_TOOL = {
    "name": EMIT_QUALITY_REVIEW_TOOL_NAME,
    "description": (
        "Return a structured code-quality review of a candidate's submitted solution -- independent of "
        "whether it passes the hidden tests, which are graded separately."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "quality_score": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
                "description": "Overall code-quality score, 0-100, independent of functional correctness.",
            },
            "dimensions": {
                "type": "object",
                "properties": {dim: {"type": "integer", "minimum": 0, "maximum": 100} for dim in QUALITY_DIMENSIONS},
                "required": QUALITY_DIMENSIONS,
                "description": (
                    "Per-dimension 0-100 scores: readability, naming_clarity, structure_organization, "
                    "function_usage (appropriate use of functions), duplication (100 = none found), "
                    "best_practices, maintainability."
                ),
            },
            "feedback": {
                "type": "string",
                "description": "Brief (2-4 sentence) actionable code-quality feedback, not about correctness.",
            },
        },
        "required": ["quality_score", "dimensions", "feedback"],
    },
    # Byte-identical on every call -- the one genuinely cacheable static
    # block. The candidate's code/problem context stays in the per-call
    # user message, outside this cached block.
    "cache_control": {"type": "ephemeral"},
}

DEFAULT_MAX_ATTEMPTS = 3


class CodeQualityReviewError(Exception):
    """Raised when Claude still returns a malformed review after every
    retry attempt -- never returned as if it were usable."""


def _is_malformed(candidate: dict) -> bool:
    score = candidate.get("quality_score")
    if not isinstance(score, int) or isinstance(score, bool) or not (0 <= score <= 100):
        return True
    dimensions = candidate.get("dimensions")
    if not isinstance(dimensions, dict):
        return True
    for dim in QUALITY_DIMENSIONS:
        value = dimensions.get(dim)
        if not isinstance(value, int) or isinstance(value, bool) or not (0 <= value <= 100):
            return True
    if not candidate.get("feedback"):
        return True
    return False


class CodeQualityReviewerAgent(BaseAgent):
    """Produces one structured code-quality review per call via a forced
    Claude tool call, given the candidate's code and the problem's public
    context (title/description/function_name only)."""

    name = "code_quality_reviewer"
    default_tier = ModelTier.CHEAP

    def __init__(self, client: anthropic.Anthropic | None = None):
        self._client = client

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        return self._client

    def _build_prompt(self, code: str, function_name: str, title: str, description: str) -> str:
        return (
            "You are reviewing the CODE QUALITY of a candidate's submission for a coding-round problem -- "
            "NOT whether it produces correct output, which is graded separately and you should ignore.\n"
            f"Problem title: {title}\n"
            f"Problem description: {description}\n"
            f"The candidate was asked to implement a function named `{function_name}`.\n"
            f"Candidate's submitted code:\n```python\n{code}\n```\n"
            "Evaluate ONLY: readability, naming clarity, structure/organization, appropriate use of "
            "functions, unnecessary duplication, basic best practices, and maintainability. Score each "
            "dimension 0-100, give an overall quality_score, and brief actionable feedback.\n"
            f"Call the {EMIT_QUALITY_REVIEW_TOOL_NAME} tool with the result. Do not include any other text."
        )

    async def run(self, **kwargs) -> AgentResult:
        code: str = kwargs["code"]
        function_name: str = kwargs["function_name"]
        title: str = kwargs["title"]
        description: str = kwargs["description"]
        max_attempts: int = kwargs.get("max_attempts", DEFAULT_MAX_ATTEMPTS)
        tier: ModelTier = kwargs.get("tier", self.default_tier)
        model = resolve_model(tier)

        usage: AgentUsage | None = None
        data: dict | None = None
        malformed = True

        for _attempt in range(1, max_attempts + 1):
            prompt = self._build_prompt(code, function_name, title, description)
            response = self.client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
                tools=[EMIT_QUALITY_REVIEW_TOOL],
                tool_choice={"type": "tool", "name": EMIT_QUALITY_REVIEW_TOOL_NAME},
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
            raise CodeQualityReviewError(
                f"Claude returned a malformed code-quality review after {max_attempts} attempt(s): {data!r}"
            )

        return AgentResult(
            data={
                "quality_score": data["quality_score"],
                "dimensions": {dim: data["dimensions"][dim] for dim in QUALITY_DIMENSIONS},
                "feedback": data["feedback"],
            },
            usage=usage,
        )


# --- Offline heuristic review (used when MOCK_MODE is enabled) ---

# Deliberately conservative: only flags genuinely vague/placeholder-style
# names, not ordinary short parameter names (e.g. `a`, `b`, `n`), which are
# completely normal for small functions and shouldn't be penalized.
_GENERIC_NAME_TOKENS = {"temp", "tmp", "data", "val", "foo", "bar", "thing", "stuff"}

_TOKEN_PATTERN = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b")


def _clamp_score(value: float) -> int:
    return max(0, min(100, round(value)))


def _heuristic_quality_review(code: str, function_name: str) -> dict:
    """Deterministic, dependency-free (stdlib `re` only) heuristic used by
    MockCodeQualityReviewerAgent. Never raises -- always returns a complete,
    well-formed review, even for empty or syntactically broken code."""
    lines = [line for line in code.splitlines() if line.strip()]
    line_count = len(lines) or 1
    stripped_lines = [line.strip() for line in lines]

    avg_line_length = sum(len(line) for line in lines) / line_count if lines else 0.0
    readability = _clamp_score(100 - max(0.0, avg_line_length - 60) * 1.5)

    has_docstring_or_comment = '"""' in code or "'''" in code or "#" in code
    mentions_function_name = bool(function_name) and function_name in code
    best_practices = _clamp_score(70 + (20 if has_docstring_or_comment else 0) + (10 if mentions_function_name else 0))

    function_count = code.count("def ")
    function_usage = _clamp_score(60 + min(function_count, 3) * 15)

    tokens = _TOKEN_PATTERN.findall(code)
    generic_hits = sum(1 for token in tokens if token.lower() in _GENERIC_NAME_TOKENS)
    naming_clarity = _clamp_score(100 - generic_hits * 15)

    unique_lines = len(set(stripped_lines)) if stripped_lines else 1
    duplication = _clamp_score(100 * unique_lines / line_count)

    structure_organization = _clamp_score(
        60 + (20 if function_count >= 1 else 0) + (20 if line_count <= 40 else 5)
    )

    dimensions = {
        "readability": readability,
        "naming_clarity": naming_clarity,
        "structure_organization": structure_organization,
        "function_usage": function_usage,
        "duplication": duplication,
        "best_practices": best_practices,
    }
    maintainability = _clamp_score(sum(dimensions.values()) / len(dimensions))
    dimensions["maintainability"] = maintainability

    quality_score = _clamp_score(sum(dimensions.values()) / len(dimensions))

    notes = []
    if not has_docstring_or_comment:
        notes.append("consider adding a docstring or comment")
    if avg_line_length > 60:
        notes.append("consider shortening long lines")
    if generic_hits:
        notes.append("some variable names are generic (e.g. temp, data) -- consider more descriptive names")
    if unique_lines < line_count:
        notes.append("some lines are duplicated -- consider extracting a helper")

    feedback = f"Heuristic review (MOCK_MODE): quality_score={quality_score}/100."
    if notes:
        feedback += " " + "; ".join(notes) + "."

    return {"quality_score": quality_score, "dimensions": dict(dimensions), "feedback": feedback}


class MockCodeQualityReviewerAgent(BaseAgent):
    """Deterministic, fully offline stand-in for CodeQualityReviewerAgent.

    Used when MOCK_MODE is enabled (the default -- no Anthropic API credits
    currently available): scores the candidate's code with a small,
    dependency-free heuristic (agents.code_quality.code_quality_reviewer.
    _heuristic_quality_review) instead of calling Claude. A clearly-labeled
    dev/demo fallback, not a claim of genuine code-quality judgement."""

    name = "code_quality_reviewer_mock"
    default_tier = ModelTier.CHEAP

    async def run(self, **kwargs) -> AgentResult:
        code: str = kwargs["code"]
        function_name: str = kwargs.get("function_name", "")
        data = _heuristic_quality_review(code, function_name)
        return AgentResult(data=data, usage=None)


def build_code_quality_reviewer() -> BaseAgent:
    """Return the Claude-backed reviewer, or the offline heuristic reviewer
    when MOCK_MODE is enabled (default; see agents/config.py). The caller
    holds whichever instance this returns behind the same BaseAgent
    contract and never has to branch on MOCK_MODE itself."""
    if MOCK_MODE:
        return MockCodeQualityReviewerAgent()
    return CodeQualityReviewerAgent()
