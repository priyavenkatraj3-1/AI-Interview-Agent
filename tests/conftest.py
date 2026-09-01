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
from app.services import aptitude_service  # noqa: E402
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


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def session_id(client) -> str:
    response = client.post("/api/sessions", json={"target_company": "TCS_NQT"})
    assert response.status_code == 200
    return response.json()["id"]
