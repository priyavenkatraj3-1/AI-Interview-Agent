"""Pydantic request/response schemas for the API layer (kept separate from
the SQLAlchemy ORM models in session.py)."""
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class SessionCreateRequest(BaseModel):
    candidate_name: str | None = None
    candidate_email: str | None = None
    target_company: str = "TCS_NQT"


class SessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    candidate_name: str | None
    candidate_email: str | None
    target_company: str
    current_stage: str
    status: str
    state: dict
    created_at: datetime
    updated_at: datetime


# --- Aptitude round (Day 2) ---


class AptitudeQuestionPublic(BaseModel):
    """The question as sent to the client — never includes correct_option
    or explanation until after the answer is submitted."""

    topic: str
    pattern: str
    question: str
    options: list[str]
    difficulty: int
    presented_at: datetime


class AptitudeStateResponse(BaseModel):
    """Shared shape for the start and current-question endpoints."""

    session_id: str
    target_company: str
    total_questions: int
    current_index: int  # 1-based number of the question being served
    difficulty: int
    score: int
    completed: bool
    started_at: datetime | None
    question: AptitudeQuestionPublic | None


class AnswerSubmitRequest(BaseModel):
    selected_option: int = Field(ge=0, le=3)


class AnswerSubmitResponse(BaseModel):
    is_correct: bool
    correct_option: int
    explanation: str
    score: int
    difficulty: int
    current_index: int
    total_questions: int
    response_time_seconds: float
    completed: bool
    next_question: AptitudeQuestionPublic | None


class AptitudeHistoryItem(BaseModel):
    index: int
    topic: str
    pattern: str
    question: str
    options: list[str]
    correct_option: int
    selected_option: int
    is_correct: bool
    difficulty: int
    presented_at: datetime
    answered_at: datetime
    response_time_seconds: float | None
    explanation: str


class AptitudeResultResponse(BaseModel):
    session_id: str
    target_company: str
    total_questions: int
    score: int
    percentage: float
    difficulty_final: int
    difficulty_history: list[int]
    total_time_seconds: float
    started_at: datetime | None
    completed_at: datetime | None
    history: list[AptitudeHistoryItem]
    topic_breakdown: dict[str, dict[str, int]]


# --- Coding round (Day 3) ---


class CodingExampleCase(BaseModel):
    input: str
    output: str
    explanation: str | None = None


class CodingProblemPublic(BaseModel):
    """The problem as sent to the client — never includes public_tests or
    hidden_tests (those carry the expected outputs used for grading)."""

    topic: str
    pattern: str
    title: str
    description: str
    constraints: str
    function_name: str
    starter_code: str
    examples: list[CodingExampleCase]
    difficulty: int
    presented_at: datetime


class CodingStateResponse(BaseModel):
    """Shared shape for the start and current-problem endpoints."""

    session_id: str
    target_company: str
    total_problems: int
    current_index: int  # 1-based number of the problem being served
    difficulty: int
    score: int
    completed: bool
    started_at: datetime | None
    problem: CodingProblemPublic | None


class CodeRunRequest(BaseModel):
    code: str = Field(min_length=1)


class CodeSubmitRequest(BaseModel):
    code: str = Field(min_length=1)


class CodeTestCaseResult(BaseModel):
    input: str
    expected_output: str
    actual_output: str | None
    passed: bool
    error: str | None
    timed_out: bool
    stdout: str
    stderr: str


class CodeRunResponse(BaseModel):
    """Result of a non-scored 'run' against the problem's sample test case(s)."""

    results: list[CodeTestCaseResult]
    passed_count: int
    total_count: int


class CodeSubmitResponse(BaseModel):
    is_correct: bool
    passed_count: int
    total_count: int
    results: list[CodeTestCaseResult]
    score: int
    difficulty: int
    current_index: int
    total_problems: int
    response_time_seconds: float
    completed: bool
    next_problem: CodingProblemPublic | None


class CodingHistoryItem(BaseModel):
    index: int
    topic: str
    pattern: str
    title: str
    difficulty: int
    submitted_code: str
    passed_count: int
    total_count: int
    is_correct: bool
    presented_at: datetime
    submitted_at: datetime
    response_time_seconds: float | None


class CodingResultResponse(BaseModel):
    session_id: str
    target_company: str
    total_problems: int
    score: int
    percentage: float
    difficulty_final: int
    difficulty_history: list[int]
    total_time_seconds: float
    started_at: datetime | None
    completed_at: datetime | None
    history: list[CodingHistoryItem]
    topic_breakdown: dict[str, dict[str, int]]


# --- Technical round (Day 4) ---


class TechnicalQuestionPublic(BaseModel):
    """The question as sent to the client — never includes model_answer or
    rubric_keywords until after the answer is submitted."""

    topic: str
    pattern: str
    question: str
    difficulty: int
    presented_at: datetime


class TechnicalStateResponse(BaseModel):
    """Shared shape for the start and current-question endpoints."""

    session_id: str
    target_company: str
    total_questions: int
    current_index: int  # 1-based number of the question being served
    difficulty: int
    score: int
    completed: bool
    started_at: datetime | None
    question: TechnicalQuestionPublic | None


class TechnicalAnswerSubmitRequest(BaseModel):
    answer: str = Field(min_length=1)


class TechnicalAnswerSubmitResponse(BaseModel):
    is_correct: bool
    answer_score: int
    feedback: str
    score: int
    difficulty: int
    current_index: int
    total_questions: int
    response_time_seconds: float
    completed: bool
    next_question: TechnicalQuestionPublic | None


class TechnicalHistoryItem(BaseModel):
    index: int
    topic: str
    pattern: str
    question: str
    candidate_answer: str
    model_answer: str
    feedback: str
    answer_score: int
    is_correct: bool
    difficulty: int
    presented_at: datetime
    answered_at: datetime
    response_time_seconds: float | None


class TechnicalResultResponse(BaseModel):
    session_id: str
    target_company: str
    total_questions: int
    score: int
    percentage: float
    difficulty_final: int
    difficulty_history: list[int]
    total_time_seconds: float
    started_at: datetime | None
    completed_at: datetime | None
    history: list[TechnicalHistoryItem]
    topic_breakdown: dict[str, dict[str, int]]


# --- HR round (Day 3 continued) ---


class HRQuestionPublic(BaseModel):
    """The question as sent to the client — never includes model_answer or
    rubric_keywords until after the answer is submitted."""

    topic: str
    pattern: str
    question: str
    difficulty: int
    presented_at: datetime


class HRStateResponse(BaseModel):
    """Shared shape for the start and current-question endpoints."""

    session_id: str
    target_company: str
    total_questions: int
    current_index: int  # 1-based number of the question being served
    difficulty: int
    score: int
    completed: bool
    started_at: datetime | None
    question: HRQuestionPublic | None


class HRAnswerSubmitRequest(BaseModel):
    answer: str = Field(min_length=1)


class HRAnswerSubmitResponse(BaseModel):
    is_correct: bool
    answer_score: int
    feedback: str
    score: int
    difficulty: int
    current_index: int
    total_questions: int
    response_time_seconds: float
    completed: bool
    next_question: HRQuestionPublic | None


class HRHistoryItem(BaseModel):
    index: int
    topic: str
    pattern: str
    question: str
    candidate_answer: str
    model_answer: str
    feedback: str
    answer_score: int
    is_correct: bool
    difficulty: int
    presented_at: datetime
    answered_at: datetime
    response_time_seconds: float | None


class HRResultResponse(BaseModel):
    session_id: str
    target_company: str
    total_questions: int
    score: int
    percentage: float
    difficulty_final: int
    difficulty_history: list[int]
    total_time_seconds: float
    started_at: datetime | None
    completed_at: datetime | None
    history: list[HRHistoryItem]
    topic_breakdown: dict[str, dict[str, int]]


# --- Final evaluation (Day 3 continued) ---


class RoundScoreSummary(BaseModel):
    score: int
    total: int
    percentage: float


class RemediationDayPlan(BaseModel):
    day: int
    focus: str
    action: str


class FinalEvaluationResponse(BaseModel):
    session_id: str
    target_company: str
    overall_score: float
    aptitude: RoundScoreSummary
    coding: RoundScoreSummary
    technical: RoundScoreSummary
    hr: RoundScoreSummary
    strengths: list[str]
    weaknesses: list[str]
    recommendation: str
    summary: str
    remediation_plan: list[RemediationDayPlan]
    hiring_verdict: str
    generated_at: datetime
