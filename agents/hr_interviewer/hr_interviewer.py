"""
HRInterviewerAgent: generates one free-text HR/behavioral interview
question at a time (topic/pattern/difficulty-driven), for the HR round.
Mirrors agents/technical_interviewer/technical_interviewer.py's "no
hardcoded question bank, no scraping" approach via a forced Claude tool
call — the difference is the topic set (behavioral/situational rather than
CS fundamentals) and that "difficulty" here tracks question depth/
complexity (a simple warm-up vs. a multi-part situational scenario) rather
than technical hardness.

Since Anthropic API credits are currently unavailable, this module also
defines MockHRInterviewerAgent: a small, hand-verified fixture bank used
instead whenever MOCK_MODE is enabled (the default — see
agents/config.py). Both classes satisfy the same BaseAgent contract, so
app.services.hr_service holds whichever build_hr_interviewer() returns
without branching on MOCK_MODE itself. Set MOCK_MODE=false (with a valid
ANTHROPIC_API_KEY) to use the real generator instead.
"""
import copy
from difflib import SequenceMatcher

import anthropic

from agents.base import AgentResult, AgentUsage, BaseAgent, ModelTier
from agents.config import ANTHROPIC_API_KEY, MOCK_MODE
from agents.model_router import resolve_model
from agents.pricing import estimate_cost_usd

from .taxonomy import DIFFICULTY_LABELS

EMIT_QUESTION_TOOL_NAME = "emit_hr_question"

EMIT_QUESTION_TOOL = {
    "name": EMIT_QUESTION_TOOL_NAME,
    "description": "Return exactly one structured HR/behavioral interview question with a grading rubric.",
    "input_schema": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "A free-text HR/behavioral interview question, self-contained and unambiguous.",
            },
            "model_answer": {
                "type": "string",
                "description": (
                    "A description of what a strong answer covers (e.g. STAR structure, relevant "
                    "qualities) — used only for grading, never shown to the candidate before they answer."
                ),
            },
            "rubric_keywords": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 8,
                "description": "Key qualities/elements a strong answer should mention, used to grade the candidate's answer.",
            },
        },
        "required": ["question", "model_answer", "rubric_keywords"],
    },
    # Byte-identical on every call to this agent, regardless of
    # topic/pattern/difficulty — the one genuinely cacheable static block.
    "cache_control": {"type": "ephemeral"},
}

# Mirrors technical_interviewer.py's retry/backoff-free bound: a bad call
# can't loop indefinitely or blow past the per-session cost budget.
DEFAULT_MAX_ATTEMPTS = 3

# SequenceMatcher ratio at/above which a newly generated question is
# considered too similar to something already asked in this session.
DUPLICATE_SIMILARITY_THRESHOLD = 0.72

MAX_PREVIOUS_QUESTIONS_IN_PROMPT = 10


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def is_similar(a: str, b: str, threshold: float = DUPLICATE_SIMILARITY_THRESHOLD) -> bool:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio() >= threshold


class HRQuestionGenerationError(Exception):
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


class HRInterviewerAgent(BaseAgent):
    """Produces one structured HR/behavioral question per call for a given
    topic/pattern/difficulty, avoiding repetition of previously asked
    questions within the same session."""

    name = "hr_interviewer"
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
            "You are conducting the HR/behavioral interview round of a placement test in the style of "
            f"{target_company.replace('_', ' ')}.\n"
            f"Topic: {topic.replace('_', ' ')}. Pattern/sub-type: {pattern.replace('_', ' ')}.\n"
            f"Depth/complexity: {label} ({difficulty}/5).\n"
            "Requirements:\n"
            "- Ask one open-ended HR/behavioral question the candidate answers in free text.\n"
            "- Describe what a strong answer covers (e.g. STAR-method structure, relevant qualities) and "
            "list 3-8 key qualities/elements a good answer should mention.\n"
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
            raise HRQuestionGenerationError(
                f"Claude returned a malformed HR question after {max_attempts} attempt(s) "
                f"(topic={topic!r}, pattern={pattern!r}): {data!r}"
            )

        data["topic"] = topic
        data["pattern"] = pattern
        data["difficulty"] = difficulty
        return AgentResult(data=data, usage=usage)


# --- Offline mock fixture bank (used when MOCK_MODE is enabled) ---

_MOCK_FIXTURES: dict[tuple[str, str], dict] = {
    ("self_introduction", "tell_me_about_yourself"): {
        "question": "Tell me about yourself and what makes you a good fit for this role.",
        "model_answer": (
            "A strong answer briefly covers relevant experience and skills, and connects them clearly "
            "to the role and the team the candidate would join."
        ),
        "rubric_keywords": ["experience", "skills", "role", "team", "learn"],
    },
    ("self_introduction", "strengths_and_weaknesses"): {
        "question": "What are your greatest strengths and weaknesses?",
        "model_answer": (
            "A strong answer names a genuine strength relevant to the job with an example, and a real "
            "weakness paired with concrete steps being taken to improve it."
        ),
        "rubric_keywords": ["strength", "weakness", "improve", "example"],
    },
    ("teamwork", "team_conflict"): {
        "question": "Describe a time you had a conflict with a teammate and how you resolved it.",
        "model_answer": (
            "A strong answer uses the STAR method: describes the situation and disagreement, the action "
            "taken to communicate and compromise, and the positive result/resolution."
        ),
        "rubric_keywords": ["conflict", "communicate", "compromise", "resolve", "team"],
    },
    ("teamwork", "collaboration_example"): {
        "question": "Give an example of when you worked successfully as part of a team.",
        "model_answer": (
            "A strong answer describes a specific project, the candidate's role and contribution, how "
            "the team collaborated, and the outcome achieved together."
        ),
        "rubric_keywords": ["team", "collaborate", "contribution", "outcome"],
    },
    ("leadership_and_ownership", "taking_initiative"): {
        "question": "Tell me about a time you took initiative or ownership of a problem without being asked.",
        "model_answer": (
            "A strong answer describes identifying a problem or opportunity, deciding to act "
            "proactively, the steps taken, and the resulting impact."
        ),
        "rubric_keywords": ["initiative", "ownership", "proactive", "impact"],
    },
    ("leadership_and_ownership", "handling_failure"): {
        "question": "Describe a time you failed at something and what you learned from it.",
        "model_answer": (
            "A strong answer honestly describes a real failure, takes responsibility rather than "
            "blaming others, and explains the concrete lesson learned and how it changed future behavior."
        ),
        "rubric_keywords": ["failure", "responsibility", "learned", "lesson"],
    },
    ("career_motivation", "why_this_company"): {
        "question": "Why do you want to work for this company?",
        "model_answer": (
            "A strong answer shows genuine research about the company's work and values, and connects "
            "them to the candidate's own career goals and interests."
        ),
        "rubric_keywords": ["company", "values", "goals", "opportunity"],
    },
    ("career_motivation", "career_goals"): {
        "question": "Where do you see yourself in the next few years, and how does this role fit into that?",
        "model_answer": (
            "A strong answer describes realistic short- and long-term career goals and explains how "
            "this specific role helps build toward them."
        ),
        "rubric_keywords": ["goals", "growth", "role", "future"],
    },
    ("pressure_and_adaptability", "handling_pressure"): {
        "question": "Describe a time you had to work under significant pressure or a tight deadline.",
        "model_answer": (
            "A strong answer describes the high-pressure situation, how the candidate prioritized and "
            "stayed organized, and the successful outcome despite the pressure."
        ),
        "rubric_keywords": ["pressure", "deadline", "prioritize", "outcome"],
    },
    ("pressure_and_adaptability", "adapting_to_change"): {
        "question": "Tell me about a time you had to quickly adapt to an unexpected change.",
        "model_answer": (
            "A strong answer describes the unexpected change, the candidate's flexible response and "
            "adjustment of plans, and a positive result from adapting quickly."
        ),
        "rubric_keywords": ["adapt", "change", "flexible", "result"],
    },
}


class MockHRInterviewerAgent(BaseAgent):
    """Deterministic, fully offline stand-in for HRInterviewerAgent.

    Used when MOCK_MODE is enabled (the default — no Anthropic API credits
    currently available): serves a hand-verified fixture instead of calling
    Claude. This is a clearly-labeled dev/demo fallback, not the intended
    production path (which stays "no hardcoded question bank" via the real
    generator above)."""

    name = "hr_interviewer_mock"
    default_tier = ModelTier.STRONG

    async def run(self, **kwargs) -> AgentResult:
        topic: str = kwargs["topic"]
        pattern: str = kwargs["pattern"]
        difficulty: int = kwargs["difficulty"]

        fixture = _MOCK_FIXTURES.get((topic, pattern))
        if fixture is None:
            raise HRQuestionGenerationError(f"No mock fixture for topic={topic!r} pattern={pattern!r}")

        data = copy.deepcopy(fixture)
        data["topic"] = topic
        data["pattern"] = pattern
        data["difficulty"] = difficulty
        return AgentResult(data=data, usage=None)


def build_hr_interviewer() -> BaseAgent:
    """Return the Claude-backed generator, or the offline mock generator
    when MOCK_MODE is enabled (default; see agents/config.py). The caller
    holds whichever instance this returns behind the same BaseAgent
    contract and never has to branch on MOCK_MODE itself."""
    if MOCK_MODE:
        return MockHRInterviewerAgent()
    return HRInterviewerAgent()
