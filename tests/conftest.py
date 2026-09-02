"""
Shared pytest fixtures.

Sets DATABASE_URL to an isolated temp SQLite file *before* importing
anything under app/, so tests never touch database/interview_agent.db.
Also patches the aptitude service's QuestionGeneratorAgent singleton with a
deterministic fake by default, so no test makes a real Claude API call
(backend/.env has no real ANTHROPIC_API_KEY configured in this environment,
and tests should be fast/offline regardless).
"""
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"

for path in (str(PROJECT_ROOT), str(BACKEND_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

_TEST_DB_FD, _TEST_DB_PATH = tempfile.mkstemp(suffix=".db", prefix="aptitude_test_")
os.close(_TEST_DB_FD)
os.environ["DATABASE_URL"] = f"sqlite:///{Path(_TEST_DB_PATH).as_posix()}"
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-placeholder")

import pytest  # noqa: E402

from app.core.database import Base, engine, SessionLocal  # noqa: E402
from app.main import app  # noqa: E402
from app.services import aptitude_service, coding_service, hr_service, technical_service  # noqa: E402
from agents.base import AgentResult, AgentUsage  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


class FakeQuestionGenerator:
    """Deterministic stand-in for QuestionGeneratorAgent: no network call,
    one distinct question per (topic, pattern, call count), correct_option
    always 0 so tests can control correctness by choosing selected_option."""

    name = "question_generator"

    def __init__(self):
        self.calls = 0

    async def run(self, **kwargs):
        self.calls += 1
        topic = kwargs["topic"]
        pattern = kwargs["pattern"]
        difficulty = kwargs["difficulty"]
        data = {
            "question": f"Fake question #{self.calls} ({topic}/{pattern}, difficulty {difficulty})",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "correct_option": 0,
            "explanation": "Option A is correct because this is a fake question.",
            "topic": topic,
            "pattern": pattern,
            "difficulty": difficulty,
        }
        usage = AgentUsage(model="fake-cheap-model", input_tokens=50, output_tokens=20, cost_usd=0.00007)
        return AgentResult(data=data, usage=usage)


@pytest.fixture(autouse=True)
def fake_question_generator(monkeypatch):
    fake = FakeQuestionGenerator()
    monkeypatch.setattr(aptitude_service, "_question_generator", fake)
    return fake


class FakeCodeProblemGenerator:
    """Deterministic stand-in for the coding round's problem generator
    (real or MockCodeProblemGeneratorAgent): no Claude call, one simple
    'add two numbers' problem per call, with a known-correct reference
    solution (see CORRECT_CODE in test_coding_api.py) so tests can control
    pass/fail by choosing what code to submit."""

    name = "code_problem_generator"

    def __init__(self):
        self.calls = 0

    async def run(self, **kwargs):
        self.calls += 1
        topic = kwargs["topic"]
        pattern = kwargs["pattern"]
        difficulty = kwargs["difficulty"]
        data = {
            "title": f"Fake Add Problem #{self.calls}",
            "description": "Return the sum of two integers a and b.",
            "constraints": "-1000 <= a, b <= 1000",
            "function_name": "add_two",
            "starter_code": "def add_two(a, b):\n    pass\n",
            "examples": [{"input": "a = 2, b = 3", "output": "5", "explanation": None}],
            "public_tests": [{"args": [2, 3], "expected": 5}],
            "hidden_tests": [
                {"args": [2, 3], "expected": 5},
                {"args": [-1, 1], "expected": 0},
                {"args": [10, 20], "expected": 30},
            ],
            "topic": topic,
            "pattern": pattern,
            "difficulty": difficulty,
        }
        usage = AgentUsage(model="fake-cheap-model", input_tokens=60, output_tokens=30, cost_usd=0.00008)
        return AgentResult(data=data, usage=usage)


@pytest.fixture(autouse=True)
def fake_code_problem_generator(monkeypatch):
    fake = FakeCodeProblemGenerator()
    monkeypatch.setattr(coding_service, "_code_problem_generator", fake)
    return fake


class FakeTechnicalInterviewer:
    """Deterministic stand-in for the technical round's question generator
    (real or MockTechnicalInterviewerAgent): no Claude call, one fixed
    stack/queue question per call with known rubric_keywords, so tests can
    control correctness deterministically via the real
    MockTechnicalKeyValidatorAgent's keyword-overlap heuristic (phase 2 of
    grading, selected automatically since MOCK_MODE defaults to true) — no
    need to fake the grader too, mirroring how the coding round's tests
    exercise the real local executor rather than faking it. See
    agents/technical_interviewer/grader.py and
    tests/test_grader_independence.py for the two-phase (independent
    draft, then key-based validation) grading pipeline."""

    name = "technical_interviewer"

    def __init__(self):
        self.calls = 0

    async def run(self, **kwargs):
        self.calls += 1
        topic = kwargs["topic"]
        pattern = kwargs["pattern"]
        difficulty = kwargs["difficulty"]
        data = {
            "question": f"Fake Technical Question #{self.calls}: explain stacks and queues.",
            "model_answer": "A stack is LIFO; a queue is FIFO.",
            "rubric_keywords": ["stack", "queue", "lifo", "fifo"],
            "topic": topic,
            "pattern": pattern,
            "difficulty": difficulty,
        }
        usage = AgentUsage(model="fake-strong-model", input_tokens=80, output_tokens=40, cost_usd=0.0002)
        return AgentResult(data=data, usage=usage)


@pytest.fixture(autouse=True)
def fake_technical_interviewer(monkeypatch):
    fake = FakeTechnicalInterviewer()
    monkeypatch.setattr(technical_service, "_technical_interviewer", fake)
    return fake


class FakeHRInterviewer:
    """Deterministic stand-in for the HR round's question generator (real
    or MockHRInterviewerAgent): no Claude call, one fixed teamwork question
    per call with known rubric_keywords, so tests can control correctness
    deterministically via the real MockHRKeyValidatorAgent's keyword-overlap
    heuristic (phase 2 of grading, selected automatically since MOCK_MODE
    defaults to true) — no need to fake the grader too, mirroring the
    technical round's tests. See agents/hr_interviewer/grader.py and
    tests/test_grader_independence.py for the two-phase (independent
    draft, then key-based validation) grading pipeline."""

    name = "hr_interviewer"

    def __init__(self):
        self.calls = 0

    async def run(self, **kwargs):
        self.calls += 1
        topic = kwargs["topic"]
        pattern = kwargs["pattern"]
        difficulty = kwargs["difficulty"]
        data = {
            "question": f"Fake HR Question #{self.calls}: describe a time you resolved a team conflict.",
            "model_answer": "A strong answer describes the conflict, how it was communicated and resolved, and the team outcome.",
            "rubric_keywords": ["conflict", "communicate", "compromise", "resolve"],
            "topic": topic,
            "pattern": pattern,
            "difficulty": difficulty,
        }
        usage = AgentUsage(model="fake-strong-model", input_tokens=80, output_tokens=40, cost_usd=0.0002)
        return AgentResult(data=data, usage=usage)


@pytest.fixture(autouse=True)
def fake_hr_interviewer(monkeypatch):
    fake = FakeHRInterviewer()
    monkeypatch.setattr(hr_service, "_hr_interviewer", fake)
    return fake


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def session_id(client) -> str:
    response = client.post("/api/sessions", json={"target_company": "TCS_NQT"})
    assert response.status_code == 200
    return response.json()["id"]
