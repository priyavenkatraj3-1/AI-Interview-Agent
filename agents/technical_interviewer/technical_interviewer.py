"""
TechnicalInterviewerAgent: generates one free-text technical interview
question at a time (topic/pattern/difficulty-driven), for the technical
round. Mirrors agents/question_generator/question_generator.py's "no
hardcoded question bank, no scraping" approach via a forced Claude tool
call — the only difference is the question is open-ended (graded by
agents/technical_interviewer/grader.py) rather than multiple-choice.

Since Anthropic API credits are currently unavailable, this module also
defines MockTechnicalInterviewerAgent: a small, hand-verified fixture bank
used instead whenever MOCK_MODE is enabled (the default — see
agents/config.py). Both classes satisfy the same BaseAgent contract, so
app.services.technical_service holds whichever build_technical_interviewer()
returns without branching on MOCK_MODE itself. Set MOCK_MODE=false (with a
valid ANTHROPIC_API_KEY) to use the real generator instead.
"""
import copy
from difflib import SequenceMatcher

import anthropic

from agents.base import AgentResult, AgentUsage, BaseAgent, ModelTier
from agents.config import ANTHROPIC_API_KEY, MOCK_MODE
from agents.model_router import resolve_model
from agents.pricing import estimate_cost_usd

from .taxonomy import DIFFICULTY_LABELS

EMIT_QUESTION_TOOL_NAME = "emit_technical_question"

EMIT_QUESTION_TOOL = {
    "name": EMIT_QUESTION_TOOL_NAME,
    "description": "Return exactly one structured technical interview question with a grading rubric.",
    "input_schema": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "A free-text technical interview question, self-contained and unambiguous.",
            },
            "model_answer": {
                "type": "string",
                "description": (
                    "A strong reference answer used only for grading — never shown to the candidate "
                    "before they answer."
                ),
            },
            "rubric_keywords": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 8,
                "description": "Key concepts/terms a strong answer should mention, used to grade the candidate's answer.",
            },
        },
        "required": ["question", "model_answer", "rubric_keywords"],
    },
    # Byte-identical on every call to this agent, regardless of
    # topic/pattern/difficulty — the one genuinely cacheable static block.
    "cache_control": {"type": "ephemeral"},
}

# Mirrors question_generator.py's retry/backoff-free bound: a bad call can't
# loop indefinitely or blow past the per-session cost budget.
DEFAULT_MAX_ATTEMPTS = 3

# SequenceMatcher ratio at/above which a newly generated question is
# considered too similar to something already asked in this session.
DUPLICATE_SIMILARITY_THRESHOLD = 0.72

MAX_PREVIOUS_QUESTIONS_IN_PROMPT = 10


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def is_similar(a: str, b: str, threshold: float = DUPLICATE_SIMILARITY_THRESHOLD) -> bool:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio() >= threshold


class TechnicalQuestionGenerationError(Exception):
    """Raised when Claude still returns a malformed question after every
    retry attempt, or when the mock generator has no fixture for the
    requested topic/pattern — never returned as if it were usable."""


def _is_malformed(candidate: dict) -> bool:
    if not candidate.get("question") or not candidate.get("model_answer"):
        return True
    keywords = candidate.get("rubric_keywords")
    if not isinstance(keywords, list) or len(keywords) < 3:
        return True
    if not all(isinstance(k, str) and k.strip() for k in keywords):
        return True
    return False


class TechnicalInterviewerAgent(BaseAgent):
    """Produces one structured technical question per call for a given
    topic/pattern/difficulty, avoiding repetition of previously asked
    questions within the same session."""

    name = "technical_interviewer"
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
            "You are conducting the technical interview round of a placement test in the style of "
            f"{target_company.replace('_', ' ')}.\n"
            f"Topic: {topic.replace('_', ' ')}. Pattern/sub-type: {pattern.replace('_', ' ')}.\n"
            f"Difficulty: {label} ({difficulty}/5).\n"
            "Requirements:\n"
            "- Ask one open-ended technical question the candidate answers in free text (not multiple choice).\n"
            "- Provide a strong reference answer and 3-8 key concepts/terms a good answer should mention.\n"
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

            malformed = _is_malformed(candidate)
            duplicate = not malformed and any(
                is_similar(candidate["question"], prev) for prev in previous_questions
            )
            if not malformed and not duplicate:
                break

        if malformed:
            raise TechnicalQuestionGenerationError(
                f"Claude returned a malformed technical question after {max_attempts} attempt(s) "
                f"(topic={topic!r}, pattern={pattern!r}): {data!r}"
            )

        data["topic"] = topic
        data["pattern"] = pattern
        data["difficulty"] = difficulty
        return AgentResult(data=data, usage=usage)


# --- Offline mock fixture bank (used when MOCK_MODE is enabled) ---

_MOCK_FIXTURES: dict[tuple[str, str], dict] = {
    ("data_structures", "arrays_and_strings"): {
        "question": (
            "What is the time complexity of accessing an element by index in an array versus a linked "
            "list, and why?"
        ),
        "model_answer": (
            "Array access by index is O(1) because elements are stored in contiguous memory and the "
            "address can be computed directly. Linked list access is O(n) because you must traverse "
            "node by node from the head."
        ),
        "rubric_keywords": ["array", "linked list", "o(1)", "o(n)", "contiguous", "traverse"],
    },
    ("data_structures", "stacks_and_queues"): {
        "question": "Explain the difference between a stack and a queue, and give a real-world example of each.",
        "model_answer": (
            "A stack is LIFO (last in, first out) — the last element added is the first removed, like a "
            "stack of plates. A queue is FIFO (first in, first out) — the first element added is the "
            "first removed, like a line of people."
        ),
        "rubric_keywords": ["stack", "queue", "lifo", "fifo", "last in", "first in"],
    },
    ("algorithms", "sorting_and_searching"): {
        "question": "Compare binary search and linear search in terms of time complexity and the precondition each requires.",
        "model_answer": (
            "Linear search checks each element in order, O(n) time, and works on any array. Binary "
            "search repeatedly halves the search space, O(log n) time, but requires the array to be "
            "sorted first."
        ),
        "rubric_keywords": ["binary search", "linear search", "sorted", "o(log n)", "o(n)"],
    },
    ("algorithms", "recursion"): {
        "question": "What is a base case in recursion, and what happens if a recursive function doesn't have one?",
        "model_answer": (
            "The base case is the condition under which a recursive function stops calling itself and "
            "returns a value directly. Without one, the function recurses indefinitely, leading to a "
            "stack overflow."
        ),
        "rubric_keywords": ["base case", "recursion", "stack overflow", "terminate"],
    },
    ("oop", "encapsulation_and_abstraction"): {
        "question": "What is encapsulation in object-oriented programming, and why is it useful?",
        "model_answer": (
            "Encapsulation bundles data and the methods that operate on it into a single unit (a class), "
            "restricting direct access to internal state via access modifiers. It's useful because it "
            "hides implementation details and protects an object's invariants from external interference."
        ),
        "rubric_keywords": ["encapsulation", "class", "access modifier", "hide", "implementation details"],
    },
    ("oop", "inheritance_and_polymorphism"): {
        "question": "Explain polymorphism in OOP with a simple example.",
        "model_answer": (
            "Polymorphism lets objects of different classes be treated through a common interface, with "
            "each class providing its own implementation of a shared method. For example, a Shape base "
            "class with an area() method, overridden differently by Circle and Square subclasses."
        ),
        "rubric_keywords": ["polymorphism", "interface", "override", "subclass", "base class"],
    },
    ("dbms", "sql_queries"): {
        "question": "What is the difference between an INNER JOIN and a LEFT JOIN in SQL?",
        "model_answer": (
            "An INNER JOIN returns only rows with matching values in both tables. A LEFT JOIN returns "
            "all rows from the left table, plus matching rows from the right table, with NULLs where "
            "there's no match."
        ),
        "rubric_keywords": ["inner join", "left join", "matching", "null"],
    },
    ("dbms", "normalization"): {
        "question": "What is database normalization and what problem does it solve?",
        "model_answer": (
            "Normalization organizes tables to reduce data redundancy and avoid update/insert/delete "
            "anomalies, typically by decomposing tables into smaller ones linked by foreign keys "
            "according to normal forms (1NF, 2NF, 3NF)."
        ),
        "rubric_keywords": ["normalization", "redundancy", "anomaly", "normal form", "foreign key"],
    },
    ("os_and_networks", "processes_and_threads"): {
        "question": "What is the key difference between a process and a thread?",
        "model_answer": (
            "A process is an independent program in execution with its own memory space, while a "
            "thread is a lightweight unit of execution within a process that shares the process's "
            "memory with other threads."
        ),
        "rubric_keywords": ["process", "thread", "memory space", "shared", "lightweight"],
    },
    ("os_and_networks", "networking_basics"): {
        "question": "What is the difference between TCP and UDP?",
        "model_answer": (
            "TCP is connection-oriented and guarantees reliable, ordered delivery of data via "
            "acknowledgments and retransmission. UDP is connectionless, faster, and does not guarantee "
            "delivery or ordering."
        ),
        "rubric_keywords": ["tcp", "udp", "connection-oriented", "connectionless", "reliable"],
    },
}


class MockTechnicalInterviewerAgent(BaseAgent):
    """Deterministic, fully offline stand-in for TechnicalInterviewerAgent.

    Used when MOCK_MODE is enabled (the default — no Anthropic API credits
    currently available): serves a hand-verified fixture instead of calling
    Claude. This is a clearly-labeled dev/demo fallback, not the intended
    production path (which stays "no hardcoded question bank" via the real
    generator above)."""

    name = "technical_interviewer_mock"
    default_tier = ModelTier.STRONG

    async def run(self, **kwargs) -> AgentResult:
        topic: str = kwargs["topic"]
        pattern: str = kwargs["pattern"]
        difficulty: int = kwargs["difficulty"]

        fixture = _MOCK_FIXTURES.get((topic, pattern))
        if fixture is None:
            raise TechnicalQuestionGenerationError(
                f"No mock fixture for topic={topic!r} pattern={pattern!r}"
            )

        data = copy.deepcopy(fixture)
        data["topic"] = topic
        data["pattern"] = pattern
        data["difficulty"] = difficulty
        return AgentResult(data=data, usage=None)


def build_technical_interviewer() -> BaseAgent:
    """Return the Claude-backed generator, or the offline mock generator
    when MOCK_MODE is enabled (default; see agents/config.py). The caller
    holds whichever instance this returns behind the same BaseAgent
    contract and never has to branch on MOCK_MODE itself."""
    if MOCK_MODE:
        return MockTechnicalInterviewerAgent()
    return TechnicalInterviewerAgent()
