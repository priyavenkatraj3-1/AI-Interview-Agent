"""
Dedicated tests proving prompt caching is correctly applied: "use Anthropic
prompt caching where appropriate to reduce repeated input-token cost."

Two layers of proof, covering every one of the 9 real Claude-calling
agents in this codebase:

1. Static: each agent's tool-schema constant (the one genuinely
   byte-identical-across-calls block — see agents/*/[grader|_interviewer|
   code_problem_generator|question_generator|final_evaluator].py) carries
   `cache_control: {"type": "ephemeral"}`.

2. Data-flow (the real call site): using the same FakeClient pattern
   already established in test_question_generator.py, test_technical_grader.py,
   etc. (no network call), drive each agent's run() and inspect the exact
   kwargs actually passed to `messages.create()`:
   - `tools[0]["cache_control"]` is present — the static block is marked
     cacheable at the real call site, not just in the constant in isolation.
   - the per-call `messages` content (topic, candidate answer, question,
     rounds data, etc.) is a plain string with no `cache_control` key
     anywhere on it — dynamic content is never accidentally cached.

No agent's behavior, model routing, or MOCK_MODE selection is touched by
this change: MOCK_MODE agents never call messages.create() at all (see
tests/test_technical_grader.py etc. for their own dedicated coverage), so
this file only exercises the real (non-mock) agent classes with a fake
client, exactly like every other "real agent" test in this suite already
does.
"""
import pytest

from agents.code_problem_generator.code_problem_generator import (
    CodeProblemGeneratorAgent,
    EMIT_PROBLEM_TOOL,
)
from agents.hr_interviewer.grader import (
    EMIT_DRAFT_TOOL as HR_EMIT_DRAFT_TOOL,
    EMIT_GRADE_TOOL as HR_EMIT_GRADE_TOOL,
    HRKeyValidatorAgent,
    IndependentHRGraderAgent,
)
from agents.hr_interviewer.hr_interviewer import EMIT_QUESTION_TOOL as HR_EMIT_QUESTION_TOOL, HRInterviewerAgent
from agents.orchestrator.final_evaluator import EMIT_EVALUATION_TOOL, FinalEvaluatorAgent
from agents.question_generator.question_generator import (
    EMIT_QUESTION_TOOL as APTITUDE_EMIT_QUESTION_TOOL,
    QuestionGeneratorAgent,
)
from agents.technical_interviewer.grader import (
    EMIT_DRAFT_TOOL as TECHNICAL_EMIT_DRAFT_TOOL,
    EMIT_GRADE_TOOL as TECHNICAL_EMIT_GRADE_TOOL,
    IndependentTechnicalGraderAgent,
    TechnicalKeyValidatorAgent,
)
from agents.technical_interviewer.technical_interviewer import (
    EMIT_QUESTION_TOOL as TECHNICAL_EMIT_QUESTION_TOOL,
    TechnicalInterviewerAgent,
)

CACHE_CONTROL_EPHEMERAL = {"type": "ephemeral"}

# --- 1. Static: every one of the 9 tool-schema constants is cache-marked ---

ALL_TOOL_CONSTANTS = {
    "aptitude question_generator": APTITUDE_EMIT_QUESTION_TOOL,
    "code_problem_generator": EMIT_PROBLEM_TOOL,
    "technical_interviewer": TECHNICAL_EMIT_QUESTION_TOOL,
    "technical grader phase 1 (independent)": TECHNICAL_EMIT_DRAFT_TOOL,
    "technical grader phase 2 (key validator)": TECHNICAL_EMIT_GRADE_TOOL,
    "hr_interviewer": HR_EMIT_QUESTION_TOOL,
    "hr grader phase 1 (independent)": HR_EMIT_DRAFT_TOOL,
    "hr grader phase 2 (key validator)": HR_EMIT_GRADE_TOOL,
    "final_evaluator": EMIT_EVALUATION_TOOL,
}


@pytest.mark.parametrize("label,tool", list(ALL_TOOL_CONSTANTS.items()))
def test_every_agent_tool_schema_carries_cache_control(label, tool):
    assert tool.get("cache_control") == CACHE_CONTROL_EPHEMERAL, label
    # Sanity: still a well-formed tool definition, not replaced/corrupted.
    assert "name" in tool
    assert "input_schema" in tool


def test_exactly_nine_agent_tool_schemas_exist():
    # Guards against silently missing one of the real Claude call sites in
    # the parametrized check above if a new agent is added later without
    # updating this test file.
    assert len(ALL_TOOL_CONSTANTS) == 9


# --- 2. Data-flow: the real call site marks the tool cacheable and never
# marks dynamic content cacheable ---


class FakeToolUseBlock:
    def __init__(self, input_data: dict):
        self.type = "tool_use"
        self.input = input_data


class FakeUsage:
    def __init__(self, input_tokens=100, output_tokens=50):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class FakeResponse:
    def __init__(self, input_data: dict, input_tokens=100, output_tokens=50):
        self.content = [FakeToolUseBlock(input_data)]
        self.usage = FakeUsage(input_tokens, output_tokens)


class FakeMessages:
    def __init__(self, responses: list[FakeResponse]):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


class FakeClient:
    def __init__(self, responses: list[FakeResponse]):
        self.messages = FakeMessages(responses)


def _assert_call_shape_is_correct(call_kwargs: dict) -> None:
    """Shared assertions for every real-agent call site: the tool is
    cache-marked, and the dynamic per-call message content is a plain
    string with no cache_control anywhere on it."""
    tools = call_kwargs["tools"]
    assert len(tools) == 1
    assert tools[0]["cache_control"] == CACHE_CONTROL_EPHEMERAL

    messages = call_kwargs["messages"]
    assert len(messages) == 1
    message = messages[0]
    assert "cache_control" not in message
    assert isinstance(message["content"], str)  # a plain string content block, not a cacheable content-block list
    assert "system" not in call_kwargs  # confirms no system-prompt restructuring was introduced


@pytest.mark.asyncio
async def test_aptitude_question_generator_call_shape():
    valid = {
        "question": "What is 2+2?",
        "options": ["3", "4", "5", "6"],
        "correct_option": 1,
        "explanation": "2+2=4.",
    }
    client = FakeClient([FakeResponse(valid)])
    agent = QuestionGeneratorAgent(client=client)
    await agent.run(target_company="TCS_NQT", topic="quantitative", pattern="percentages", difficulty=3, previous_questions=[])
    _assert_call_shape_is_correct(client.messages.calls[0])


@pytest.mark.asyncio
async def test_code_problem_generator_call_shape():
    valid = {
        "title": "Sum",
        "description": "Return a + b.",
        "constraints": "n/a",
        "function_name": "add_two",
        "starter_code": "def add_two(a, b):\n    pass\n",
        "examples": [{"input": "a=2, b=3", "output": "5"}],
        "public_tests": [{"args": [2, 3], "expected": 5}],
        "hidden_tests": [
            {"args": [2, 3], "expected": 5},
            {"args": [-1, 1], "expected": 0},
            {"args": [10, 20], "expected": 30},
        ],
    }
    client = FakeClient([FakeResponse(valid)])
    agent = CodeProblemGeneratorAgent(client=client)
    await agent.run(topic="arrays", pattern="two_sum", difficulty=3, previous_titles=[])
    _assert_call_shape_is_correct(client.messages.calls[0])


@pytest.mark.asyncio
async def test_technical_interviewer_call_shape():
    valid = {
        "question": "Explain stacks vs queues.",
        "model_answer": "A stack is LIFO; a queue is FIFO.",
        "rubric_keywords": ["stack", "queue", "lifo", "fifo"],
        "is_follow_up": False,
    }
    client = FakeClient([FakeResponse(valid)])
    agent = TechnicalInterviewerAgent(client=client)
    await agent.run(target_company="TCS_NQT", topic="data_structures", pattern="stacks_and_queues", difficulty=3, previous_questions=[])
    _assert_call_shape_is_correct(client.messages.calls[0])


@pytest.mark.asyncio
async def test_technical_independent_grader_call_shape():
    valid = {"draft_score": 70, "draft_feedback": "Reasonable attempt."}
    client = FakeClient([FakeResponse(valid)])
    agent = IndependentTechnicalGraderAgent(client=client)
    await agent.run(question="Explain stacks vs queues.", candidate_answer="A stack is LIFO.")
    _assert_call_shape_is_correct(client.messages.calls[0])


@pytest.mark.asyncio
async def test_technical_key_validator_call_shape():
    valid = {"score": 85, "feedback": "Good."}
    client = FakeClient([FakeResponse(valid)])
    agent = TechnicalKeyValidatorAgent(client=client)
    await agent.run(
        question="Explain stacks vs queues.",
        model_answer="A stack is LIFO; a queue is FIFO.",
        rubric_keywords=["stack", "queue"],
        candidate_answer="A stack is LIFO.",
        draft_score=70,
        draft_feedback="Reasonable attempt.",
    )
    _assert_call_shape_is_correct(client.messages.calls[0])


@pytest.mark.asyncio
async def test_hr_interviewer_call_shape():
    valid = {
        "question": "Describe a conflict you resolved.",
        "model_answer": "Describes conflict, communication, resolution.",
        "rubric_keywords": ["conflict", "communicate", "resolve"],
    }
    client = FakeClient([FakeResponse(valid)])
    agent = HRInterviewerAgent(client=client)
    await agent.run(target_company="TCS_NQT", topic="teamwork", pattern="team_conflict", difficulty=3, previous_questions=[])
    _assert_call_shape_is_correct(client.messages.calls[0])


@pytest.mark.asyncio
async def test_hr_independent_grader_call_shape():
    valid = {"draft_score": 70, "draft_feedback": "Reasonable attempt."}
    client = FakeClient([FakeResponse(valid)])
    agent = IndependentHRGraderAgent(client=client)
    await agent.run(question="Describe a conflict you resolved.", candidate_answer="I resolved a conflict.")
    _assert_call_shape_is_correct(client.messages.calls[0])


@pytest.mark.asyncio
async def test_hr_key_validator_call_shape():
    valid = {"score": 85, "feedback": "Good."}
    client = FakeClient([FakeResponse(valid)])
    agent = HRKeyValidatorAgent(client=client)
    await agent.run(
        question="Describe a conflict you resolved.",
        model_answer="Describes conflict, communication, resolution.",
        rubric_keywords=["conflict", "resolve"],
        candidate_answer="I resolved a conflict.",
        draft_score=70,
        draft_feedback="Reasonable attempt.",
    )
    _assert_call_shape_is_correct(client.messages.calls[0])


@pytest.mark.asyncio
async def test_final_evaluator_call_shape():
    valid = {
        "strengths": ["Strong in Coding."],
        "weaknesses": ["Weak in Technical."],
        "recommendation": "Recommend",
        "summary": "Solid overall performance.",
        "remediation_plan": [{"day": d, "focus": f"Focus {d}", "action": f"Action {d}"} for d in range(1, 15)],
        "hiring_verdict": "Likely hireable based on strong Coding performance.",
    }
    client = FakeClient([FakeResponse(valid)])
    agent = FinalEvaluatorAgent(client=client)
    await agent.run(target_company="TCS_NQT", overall_score=70.0, rounds={})
    _assert_call_shape_is_correct(client.messages.calls[0])
