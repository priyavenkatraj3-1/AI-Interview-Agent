"""
CodeProblemGeneratorAgent: generates coding-round problems (title,
description, constraints, starter code, public/hidden test cases) on demand
via the Claude API — same "no hardcoded question bank, no scraping"
approach as agents/question_generator/question_generator.py, using a
forced tool call for structured output.

Since Anthropic API credits are currently unavailable, this module also
defines MockCodeProblemGeneratorAgent: a small, hand-verified fixture bank
of Python coding problems used instead whenever MOCK_MODE is enabled (the
default — see agents/config.py). Both classes satisfy the same BaseAgent
contract, so app.services.coding_service holds whichever
build_code_problem_generator() returns without branching on MOCK_MODE
itself. Set MOCK_MODE=false (with a valid ANTHROPIC_API_KEY) to use the
real generator instead.
"""
import copy
from difflib import SequenceMatcher

import anthropic

from agents.base import AgentResult, AgentUsage, BaseAgent, ModelTier
from agents.config import ANTHROPIC_API_KEY, MOCK_MODE
from agents.model_router import resolve_model
from agents.pricing import estimate_cost_usd

from .taxonomy import DIFFICULTY_LABELS

EMIT_PROBLEM_TOOL_NAME = "emit_coding_problem"

EMIT_PROBLEM_TOOL = {
    "name": EMIT_PROBLEM_TOOL_NAME,
    "description": "Return exactly one structured coding problem with public and hidden test cases.",
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Short problem title."},
            "description": {
                "type": "string",
                "description": "Full problem statement, self-contained and unambiguous.",
            },
            "constraints": {"type": "string", "description": "Input constraints/bounds."},
            "function_name": {
                "type": "string",
                "description": "A valid Python identifier the candidate must implement.",
            },
            "starter_code": {
                "type": "string",
                "description": "A Python function stub (signature + `pass`), stdlib-only.",
            },
            "examples": {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "input": {"type": "string"},
                        "output": {"type": "string"},
                        "explanation": {"type": "string"},
                    },
                    "required": ["input", "output"],
                },
                "description": "Human-readable example(s) shown to the candidate.",
            },
            "public_tests": {
                "type": "array",
                "minItems": 1,
                "maxItems": 2,
                "items": {
                    "type": "object",
                    "properties": {
                        "args": {
                            "type": "array",
                            "description": "JSON-serializable positional arguments for function_name.",
                        },
                        "expected": {"description": "JSON-serializable expected return value."},
                    },
                    "required": ["args", "expected"],
                },
                "description": "Sample test case(s), matching the examples, used for a non-scored 'run'.",
            },
            "hidden_tests": {
                "type": "array",
                "minItems": 3,
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "properties": {
                        "args": {
                            "type": "array",
                            "description": "JSON-serializable positional arguments for function_name.",
                        },
                        "expected": {"description": "JSON-serializable expected return value."},
                    },
                    "required": ["args", "expected"],
                },
                "description": "Full hidden test suite used for scoring — never shown to the candidate.",
            },
        },
        "required": [
            "title",
            "description",
            "constraints",
            "function_name",
            "starter_code",
            "examples",
            "public_tests",
            "hidden_tests",
        ],
    },
    # Byte-identical on every call to this agent, regardless of
    # topic/pattern/difficulty — the one genuinely cacheable static block.
    "cache_control": {"type": "ephemeral"},
}

# Mirrors question_generator.py's retry/backoff-free bound: a bad call can't
# loop indefinitely or blow past the per-session cost budget.
DEFAULT_MAX_ATTEMPTS = 3

# SequenceMatcher ratio at/above which a newly generated problem's title is
# considered too similar to a title already used in this session.
DUPLICATE_SIMILARITY_THRESHOLD = 0.72

MAX_PREVIOUS_TITLES_IN_PROMPT = 10


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def is_similar(a: str, b: str, threshold: float = DUPLICATE_SIMILARITY_THRESHOLD) -> bool:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio() >= threshold


class CodeProblemGenerationError(Exception):
    """Raised when Claude still returns a malformed problem after every
    retry attempt, or when the mock generator has no fixture for the
    requested topic/pattern — never returned as if it were usable."""


def _is_malformed(candidate: dict) -> bool:
    if not candidate.get("title") or not candidate.get("description") or not candidate.get("constraints"):
        return True
    function_name = candidate.get("function_name")
    if not function_name or not str(function_name).isidentifier():
        return True
    if not candidate.get("starter_code"):
        return True
    if not candidate.get("examples"):
        return True
    hidden_tests = candidate.get("hidden_tests") or []
    if len(hidden_tests) < 3:
        return True
    public_tests = candidate.get("public_tests") or []
    if len(public_tests) < 1:
        return True
    for test in (*public_tests, *hidden_tests):
        if not isinstance(test, dict) or "args" not in test or "expected" not in test:
            return True
        if not isinstance(test["args"], list):
            return True
    return False


class CodeProblemGeneratorAgent(BaseAgent):
    """Produces one structured coding problem per call for a given
    topic/pattern/difficulty, avoiding title repetition within a session."""

    name = "code_problem_generator"
    default_tier = ModelTier.CHEAP

    def __init__(self, client: anthropic.Anthropic | None = None):
        self._client = client

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        return self._client

    def _build_prompt(self, topic: str, pattern: str, difficulty: int, previous_titles: list[str]) -> str:
        label = DIFFICULTY_LABELS[difficulty]
        avoid = ""
        if previous_titles:
            joined = "\n".join(f"- {t}" for t in previous_titles[-MAX_PREVIOUS_TITLES_IN_PROMPT:])
            avoid = f"\nDo not reuse or closely resemble any of these problem titles already used in this session:\n{joined}\n"
        return (
            "You are generating one original Python coding-interview problem "
            f"for a placement-test coding round.\n"
            f"Topic: {topic.replace('_', ' ')}. Pattern/sub-type: {pattern.replace('_', ' ')}.\n"
            f"Difficulty: {label} ({difficulty}/5).\n"
            "Requirements:\n"
            "- The candidate implements a single Python function; give its exact name and a starter stub.\n"
            "- The function must be solvable using the Python standard library only (no I/O, no third-party packages).\n"
            "- Every test case's `args` must be a JSON-serializable list of positional arguments, and `expected` a "
            "JSON-serializable value equal to calling the function with those arguments.\n"
            "- Provide at least 3 hidden test cases covering edge cases, and at least 1 public/sample test case "
            "matching an example shown to the candidate.\n"
            "- Assume any input needed to make the answer unique (e.g. 'assume exactly one valid solution exists') "
            "where relevant.\n"
            f"{avoid}"
            f"Call the {EMIT_PROBLEM_TOOL_NAME} tool with the result. Do not include any other text."
        )

    async def run(self, **kwargs) -> AgentResult:
        topic: str = kwargs["topic"]
        pattern: str = kwargs["pattern"]
        difficulty: int = kwargs["difficulty"]
        previous_titles: list[str] = kwargs.get("previous_titles", [])
        max_attempts: int = kwargs.get("max_attempts", DEFAULT_MAX_ATTEMPTS)
        tier: ModelTier = kwargs.get("tier", self.default_tier)
        model = resolve_model(tier)

        usage: AgentUsage | None = None
        data: dict | None = None
        malformed = True

        for _attempt in range(1, max_attempts + 1):
            prompt = self._build_prompt(topic, pattern, difficulty, previous_titles)
            response = self.client.messages.create(
                model=model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
                tools=[EMIT_PROBLEM_TOOL],
                tool_choice={"type": "tool", "name": EMIT_PROBLEM_TOOL_NAME},
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
            duplicate = not malformed and any(
                is_similar(candidate["title"], prev) for prev in previous_titles
            )
            if not malformed and not duplicate:
                break

        if malformed:
            raise CodeProblemGenerationError(
                f"Claude returned a malformed coding problem after {max_attempts} attempt(s) "
                f"(topic={topic!r}, pattern={pattern!r}): {data!r}"
            )

        data["topic"] = topic
        data["pattern"] = pattern
        data["difficulty"] = difficulty
        return AgentResult(data=data, usage=usage)


# --- Offline mock fixture bank (used when MOCK_MODE is enabled) ---

_MOCK_FIXTURES: dict[tuple[str, str], dict] = {
    ("arrays", "two_sum"): {
        "title": "Two Sum",
        "description": (
            "Given an array of integers `nums` and an integer `target`, return the indices of the two "
            "numbers that add up to `target`. Assume exactly one valid answer exists, and the same element "
            "may not be used twice."
        ),
        "constraints": "2 <= len(nums) <= 10^4; exactly one valid pair exists.",
        "function_name": "two_sum",
        "starter_code": "def two_sum(nums, target):\n    pass\n",
        "examples": [
            {
                "input": "nums = [2, 7, 11, 15], target = 9",
                "output": "[0, 1]",
                "explanation": "nums[0] + nums[1] == 9",
            }
        ],
        "public_tests": [{"args": [[2, 7, 11, 15], 9], "expected": [0, 1]}],
        "hidden_tests": [
            {"args": [[2, 7, 11, 15], 9], "expected": [0, 1]},
            {"args": [[3, 2, 4], 6], "expected": [1, 2]},
            {"args": [[3, 3], 6], "expected": [0, 1]},
            {"args": [[1, 2, 3, 4, 5], 9], "expected": [3, 4]},
        ],
    },
    ("arrays", "max_subarray"): {
        "title": "Maximum Subarray",
        "description": (
            "Given an integer array `nums`, find the contiguous subarray with the largest sum and return "
            "that sum."
        ),
        "constraints": "1 <= len(nums) <= 10^5",
        "function_name": "max_subarray",
        "starter_code": "def max_subarray(nums):\n    pass\n",
        "examples": [
            {
                "input": "nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]",
                "output": "6",
                "explanation": "[4, -1, 2, 1] has the largest sum = 6",
            }
        ],
        "public_tests": [{"args": [[-2, 1, -3, 4, -1, 2, 1, -5, 4]], "expected": 6}],
        "hidden_tests": [
            {"args": [[-2, 1, -3, 4, -1, 2, 1, -5, 4]], "expected": 6},
            {"args": [[1]], "expected": 1},
            {"args": [[5, 4, -1, 7, 8]], "expected": 23},
            {"args": [[-1, -2, -3]], "expected": -1},
        ],
    },
    ("strings", "reverse_string"): {
        "title": "Reverse String",
        "description": "Given a string `s`, return it reversed.",
        "constraints": "0 <= len(s) <= 10^4",
        "function_name": "reverse_string",
        "starter_code": "def reverse_string(s):\n    pass\n",
        "examples": [{"input": 's = "hello"', "output": '"olleh"', "explanation": None}],
        "public_tests": [{"args": ["hello"], "expected": "olleh"}],
        "hidden_tests": [
            {"args": ["hello"], "expected": "olleh"},
            {"args": [""], "expected": ""},
            {"args": ["a"], "expected": "a"},
            {"args": ["racecar"], "expected": "racecar"},
        ],
    },
    ("strings", "valid_palindrome"): {
        "title": "Valid Palindrome",
        "description": (
            "Given a string `s`, return whether it is a palindrome after converting to lowercase and "
            "removing all non-alphanumeric characters."
        ),
        "constraints": "0 <= len(s) <= 2 * 10^5",
        "function_name": "is_valid_palindrome",
        "starter_code": "def is_valid_palindrome(s):\n    pass\n",
        "examples": [
            {
                "input": 's = "A man, a plan, a canal: Panama"',
                "output": "true",
                "explanation": '"amanaplanacanalpanama" reads the same forwards and backwards',
            }
        ],
        "public_tests": [{"args": ["A man, a plan, a canal: Panama"], "expected": True}],
        "hidden_tests": [
            {"args": ["A man, a plan, a canal: Panama"], "expected": True},
            {"args": ["race a car"], "expected": False},
            {"args": [""], "expected": True},
            {"args": ["0P"], "expected": False},
        ],
    },
    ("recursion", "factorial"): {
        "title": "Factorial",
        "description": "Given a non-negative integer `n`, return `n!` (n factorial).",
        "constraints": "0 <= n <= 20",
        "function_name": "factorial",
        "starter_code": "def factorial(n):\n    pass\n",
        "examples": [{"input": "n = 5", "output": "120", "explanation": "5! = 5*4*3*2*1"}],
        "public_tests": [{"args": [5], "expected": 120}],
        "hidden_tests": [
            {"args": [0], "expected": 1},
            {"args": [1], "expected": 1},
            {"args": [5], "expected": 120},
            {"args": [7], "expected": 5040},
        ],
    },
    ("recursion", "fibonacci"): {
        "title": "Nth Fibonacci Number",
        "description": (
            "Given a 0-indexed integer `n`, return the n-th Fibonacci number, where fib(0) = 0 and "
            "fib(1) = 1."
        ),
        "constraints": "0 <= n <= 30",
        "function_name": "fibonacci",
        "starter_code": "def fibonacci(n):\n    pass\n",
        "examples": [{"input": "n = 10", "output": "55", "explanation": None}],
        "public_tests": [{"args": [10], "expected": 55}],
        "hidden_tests": [
            {"args": [0], "expected": 0},
            {"args": [1], "expected": 1},
            {"args": [10], "expected": 55},
            {"args": [15], "expected": 610},
        ],
    },
}


class MockCodeProblemGeneratorAgent(BaseAgent):
    """Deterministic, fully offline stand-in for CodeProblemGeneratorAgent.

    Used when MOCK_MODE is enabled (the default — no Anthropic API credits
    currently available): serves a hand-verified fixture instead of calling
    Claude. This is a clearly-labeled dev/demo fallback, not the intended
    production path (which stays "no hardcoded question bank" via the real
    generator above)."""

    name = "code_problem_generator_mock"
    default_tier = ModelTier.CHEAP

    async def run(self, **kwargs) -> AgentResult:
        topic: str = kwargs["topic"]
        pattern: str = kwargs["pattern"]
        difficulty: int = kwargs["difficulty"]

        fixture = _MOCK_FIXTURES.get((topic, pattern))
        if fixture is None:
            raise CodeProblemGenerationError(
                f"No mock fixture for topic={topic!r} pattern={pattern!r}"
            )

        data = copy.deepcopy(fixture)
        data["topic"] = topic
        data["pattern"] = pattern
        data["difficulty"] = difficulty
        return AgentResult(data=data, usage=None)


def build_code_problem_generator() -> BaseAgent:
    """Return the Claude-backed generator, or the offline mock generator
    when MOCK_MODE is enabled (default; see agents/config.py). The caller
    holds whichever instance this returns behind the same BaseAgent
    contract and never has to branch on MOCK_MODE itself."""
    if MOCK_MODE:
        return MockCodeProblemGeneratorAgent()
    return CodeProblemGeneratorAgent()
