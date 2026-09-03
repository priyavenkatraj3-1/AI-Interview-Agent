"""
TechnicalInterviewerAgent: generates one free-text technical interview
question at a time (topic/pattern/difficulty-driven), for the technical
round. Mirrors agents/question_generator/question_generator.py's "no
hardcoded question bank, no scraping" approach via a forced Claude tool
call — the only difference is the question is open-ended (graded by
agents/technical_interviewer/grader.py) rather than multiple-choice.

Two additive capabilities on top of that base flow (both optional kwargs on
run(), both purely per-call dynamic prompt content — never folded into
EMIT_QUESTION_TOOL's cache_control-marked static block, so the one
genuinely cacheable block stays byte-identical across calls):

- Socratic follow-up (`previous_turn`): the immediately previous question,
  the candidate's own answer to it, and the model_answer/rubric_keywords
  this agent itself authored for that question (its own previously-emitted
  data, not a leak from the independent grader — see
  agents/technical_interviewer/grader.py's IndependentGradingInput, which
  still never sees any of this). The agent judges whether that answer was
  weak/incomplete/vague and, if so and only if `probe_eligible` (no two
  follow-ups in a row), asks ONE same-topic "why?"/clarify/justify
  follow-up instead of moving to a new topic. Reported back via
  `is_follow_up` in the returned data; app.services.technical_service
  decides topic/pattern bookkeeping from that flag, and this call still
  counts toward the fixed TOTAL_QUESTIONS turn limit either way.
- Stage-2 code probing (`coding_context`): the candidate's own submitted
  code from the coding round (title + submitted_code only — never hidden
  test cases/expected outputs, which app.services.technical_service never
  fetches in the first place; see that module's _get_coding_context).
  When present, the question must be about this specific submission.

Since Anthropic API credits are currently unavailable, this module also
defines MockTechnicalInterviewerAgent: a small, hand-verified fixture bank
(plus a deterministic word-count heuristic for the two capabilities above)
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

from .taxonomy import DIFFICULTY_LABELS, WEAK_ANSWER_WORD_THRESHOLD

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
            "is_follow_up": {
                "type": "boolean",
                "description": (
                    "true only if this question is a Socratic follow-up probing the immediately previous "
                    "answer (same topic, asks the candidate to explain/justify/clarify — e.g. 'why?') "
                    "instead of moving to a new topic; false for a normal new-topic question."
                ),
            },
        },
        "required": ["question", "model_answer", "rubric_keywords", "is_follow_up"],
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
    if not isinstance(candidate.get("is_follow_up"), bool):
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
        *,
        previous_turn: dict | None = None,
        coding_context: dict | None = None,
    ) -> str:
        label = DIFFICULTY_LABELS[difficulty]
        avoid = ""
        if previous_questions:
            joined = "\n".join(f"- {q}" for q in previous_questions[-MAX_PREVIOUS_QUESTIONS_IN_PROMPT:])
            avoid = (
                "\nDo not repeat, rephrase, or closely resemble any of these "
                f"questions already asked in this session:\n{joined}\n"
            )

        # Dynamic, per-call content only -- never folded into
        # EMIT_QUESTION_TOOL's cache_control-marked static block above.
        coding_block = ""
        if coding_context is not None:
            coding_block = (
                "\nThe candidate previously submitted this Python solution for a coding-round problem "
                f"titled \"{coding_context['title']}\":\n"
                f"```python\n{coding_context['submitted_code']}\n```\n"
                "For THIS question specifically: ignore the topic/pattern below and instead ask an "
                "open-ended technical question that references this exact submission -- e.g. its time/space "
                "complexity, an edge case it may not handle, why this particular approach was chosen, or how "
                "it could be improved. The question must be about this specific code, not a generic question "
                "about the topic. Set is_follow_up to false for this question.\n"
            )

        probe_block = ""
        if previous_turn is not None:
            if previous_turn["probe_eligible"]:
                probe_block = (
                    "\nThe candidate's immediately previous question and answer in this interview were:\n"
                    f"Previous question: {previous_turn['question']}\n"
                    f"Previous candidate answer: {previous_turn['candidate_answer']}\n"
                    f"(For reference) the model answer you wrote for that question: {previous_turn['model_answer']}\n"
                    "First, judge for yourself whether that answer is weak, incomplete, vague, or fails to "
                    "justify itself -- not merely whether it is short. If, and only if, it genuinely needs "
                    "probing, ask ONE Socratic follow-up question that stays on the SAME topic/pattern as the "
                    "previous question and asks the candidate to explain their reasoning, justify a claim, or "
                    "clarify what they meant (e.g. 'why...', 'what would happen if...', 'can you walk "
                    "through...'). Set is_follow_up to true in that case. Do not do this for every answer -- "
                    "only when it actually warrants it; a clear, complete, correct answer should NOT be "
                    "followed up. If it does not need probing, ask the next NEW question as instructed below "
                    "(topic/pattern/difficulty given below) and set is_follow_up to false.\n"
                )
            else:
                probe_block = (
                    "\nA Socratic follow-up question was already asked after the candidate's previous answer, "
                    "so do not probe again immediately -- ask the next NEW question as instructed below and "
                    "set is_follow_up to false.\n"
                )

        return (
            "You are conducting the technical interview round of a placement test in the style of "
            f"{target_company.replace('_', ' ')}.\n"
            f"Topic: {topic.replace('_', ' ')}. Pattern/sub-type: {pattern.replace('_', ' ')}.\n"
            f"Difficulty: {label} ({difficulty}/5).\n"
            f"{coding_block}"
            f"{probe_block}"
            "Requirements:\n"
            "- Ask one open-ended technical question the candidate answers in free text (not multiple choice).\n"
            "- Provide a strong reference answer and 3-8 key concepts/terms a good answer should mention.\n"
            f"{avoid}"
            f"Call the {EMIT_QUESTION_TOOL_NAME} tool with the result, including is_follow_up. Do not include "
            "any other text."
        )

    async def run(self, **kwargs) -> AgentResult:
        target_company: str = kwargs["target_company"]
        topic: str = kwargs["topic"]
        pattern: str = kwargs["pattern"]
        difficulty: int = kwargs["difficulty"]
        previous_questions: list[str] = kwargs.get("previous_questions", [])
        previous_turn: dict | None = kwargs.get("previous_turn")
        coding_context: dict | None = kwargs.get("coding_context")
        max_attempts: int = kwargs.get("max_attempts", DEFAULT_MAX_ATTEMPTS)
        tier: ModelTier = kwargs.get("tier", self.default_tier)
        model = resolve_model(tier)

        usage: AgentUsage | None = None
        data: dict | None = None
        malformed = True

        for _attempt in range(1, max_attempts + 1):
            prompt = self._build_prompt(
                target_company,
                topic,
                pattern,
                difficulty,
                previous_questions,
                previous_turn=previous_turn,
                coding_context=coding_context,
            )
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


# Offline fixture for the Stage-2 code-probing question (coding_context is
# given): a code-review-style question templated with the candidate's own
# submitted code, so the question text genuinely references their specific
# submission rather than being generic.
_CODE_REVIEW_FIXTURE_TEMPLATE = {
    "question": (
        'Looking at your own submitted solution for "{title}":\n{code}\n'
        "What is the time complexity of this solution, and can you identify one edge case it might not "
        "handle correctly?"
    ),
    "model_answer": (
        "A strong answer states the Big-O time (and ideally space) complexity of the submitted approach, "
        "and identifies at least one edge case (e.g. empty input, a single element, duplicates, or "
        "negative values) that the submitted code may not handle correctly."
    ),
    "rubric_keywords": ["complexity", "time", "space", "edge case", "big-o"],
}


def _is_weak_answer(answer: str) -> bool:
    """Deterministic weak-answer heuristic for the offline mock interviewer
    (see taxonomy.WEAK_ANSWER_WORD_THRESHOLD): an answer under that many
    words is treated as too short/vague to have adequately explained
    itself. Mirrors the *spirit* of the real Claude-backed agent's judgment
    without an API call -- not a claim that word count is a good general
    proxy for answer quality."""
    return len(answer.split()) < WEAK_ANSWER_WORD_THRESHOLD


class MockTechnicalInterviewerAgent(BaseAgent):
    """Deterministic, fully offline stand-in for TechnicalInterviewerAgent.

    Used when MOCK_MODE is enabled (the default — no Anthropic API credits
    currently available): serves a hand-verified fixture instead of calling
    Claude. This is a clearly-labeled dev/demo fallback, not the intended
    production path (which stays "no hardcoded question bank" via the real
    generator above).

    Implements the same two additive capabilities as the real agent
    (coding_context / previous_turn), via deterministic rules instead of an
    LLM call, so tests can exercise Socratic follow-up and Stage-2 code
    probing offline. See _is_weak_answer and _CODE_REVIEW_FIXTURE_TEMPLATE."""

    name = "technical_interviewer_mock"
    default_tier = ModelTier.STRONG

    async def run(self, **kwargs) -> AgentResult:
        topic: str = kwargs["topic"]
        pattern: str = kwargs["pattern"]
        difficulty: int = kwargs["difficulty"]
        previous_turn: dict | None = kwargs.get("previous_turn")
        coding_context: dict | None = kwargs.get("coding_context")

        if coding_context is not None:
            data = copy.deepcopy(_CODE_REVIEW_FIXTURE_TEMPLATE)
            data["question"] = data["question"].format(
                title=coding_context["title"], code=coding_context["submitted_code"]
            )
            data["is_follow_up"] = False
        elif (
            previous_turn is not None
            and previous_turn["probe_eligible"]
            and _is_weak_answer(previous_turn["candidate_answer"])
        ):
            data = {
                "question": (
                    f'You answered "{previous_turn["candidate_answer"]}" -- why? Can you explain your '
                    f'reasoning for "{previous_turn["question"]}" in more depth?'
                ),
                "model_answer": previous_turn["model_answer"],
                "rubric_keywords": previous_turn["rubric_keywords"],
                "is_follow_up": True,
            }
        else:
            fixture = _MOCK_FIXTURES.get((topic, pattern))
            if fixture is None:
                raise TechnicalQuestionGenerationError(
                    f"No mock fixture for topic={topic!r} pattern={pattern!r}"
                )
            data = copy.deepcopy(fixture)
            data["is_follow_up"] = False

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
