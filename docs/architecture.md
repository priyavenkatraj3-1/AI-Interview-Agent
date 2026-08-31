# Architecture — Day 1 Foundation

## What this project is

An AI-driven placement interview simulator (targeting the TCS NQT / Infosys /
Wipro style hiring process). A candidate goes through four stages —
**aptitude → coding → technical → HR** — and receives a final scorecard,
remediation plan, and hire/no-hire verdict.

This document describes the Day 1 foundation only: the skeleton that later
stages are built on top of. None of the interview logic itself is
implemented yet.

## Repo layout

```
agents/       Stage-agent Python package (framework-agnostic, no FastAPI import)
backend/      FastAPI application (HTTP API, persistence)
database/     SQLite database file + migrations
docs/         Project documentation
frontend/     React (Vite) single-page app
tests/        Test suite (not yet populated)
```

## Backend

FastAPI app under `backend/app/`:

- `main.py` — app entrypoint, CORS, router registration, `init_db()` on startup.
- `core/config.py` — `Settings` (pydantic-settings), reads `backend/.env`.
- `core/database.py` — SQLAlchemy engine/session, `Base`, `get_db()` dependency.
- `models/session.py` — ORM models: `InterviewSession`, `StageProgress`, `CostLog`.
- `models/schemas.py` — Pydantic request/response schemas (kept separate from ORM models).
- `routes/health.py` — `GET /health` liveness check.
- `routes/sessions.py` — minimal session create/fetch, to prove persistence works end to end.

Run with `uvicorn app.main:app --reload --port 8000` from `backend/`.

## Database

SQLite, file lives at `database/interview_agent.db` (git-ignored, created
automatically on first run). Three tables exist on Day 1:

- **`interview_sessions`** — one row per candidate attempt. Tracks
  `current_stage` (`aptitude` / `coding` / `technical` / `hr` / `completed`),
  `status`, and a free-form `state` JSON blob so a page refresh can resume
  exactly where the candidate left off.
- **`stage_progress`** — one row per stage per session (status, score, and a
  `details` JSON blob for stage-specific results). Kept as JSON rather than
  new tables per stage so the schema doesn't need to change as each stage is
  built.
- **`cost_logs`** — per-agent-call token/cost usage, so spend can be
  aggregated per stage or session from the start.

No schema migration tool is wired up yet; `init_db()` just calls
`Base.metadata.create_all()`. `database/migrations/` is reserved for when
that's needed.

## Agents

`agents/` is a standalone Python package — it does not import FastAPI and can
be imported/tested independently of the backend.

- `base.py` — the shared contract every stage agent implements:
  `BaseAgent.run()` returns an `AgentResult` (structured `dict` + optional
  `AgentUsage`), never a raw string for downstream code to parse. Also
  defines `ModelTier` (`CHEAP` / `STRONG`).
- `config.py` — reads `ANTHROPIC_API_KEY` / `MODEL_CHEAP` / `MODEL_STRONG`
  from `backend/.env` directly (agents deliberately don't import across the
  `agents/` ↔ `backend/` boundary).
- `model_router.py` — resolves a `ModelTier` to a concrete Claude model id,
  so agents never hardcode a model.
- `orchestrator/orchestrator.py` — owns the stage state machine
  (`aptitude → coding → technical → hr → completed`) via `get_next_stage()`,
  and will delegate to the stage agents below once they're implemented.
- `question_generator/`, `grader/`, `technical_interviewer/`,
  `hr_interviewer/` — one placeholder agent class each, subclassing
  `BaseAgent`. Every `run()` currently raises `NotImplementedError` — Day 1
  only fixes the shape (class name, `name`, `default_tier`), not the
  behavior.

Explicitly **not** built yet (by design, per the Day 1 scope): actual
question generation/grading logic, a hardcoded question bank, any question
scraping, Judge0/Piston code execution, and the multi-persona HR evaluation.

## Frontend

React + Vite app in `frontend/`. On Day 1 it is intentionally minimal: a
single page that calls `GET /health` on the backend and displays whether the
API is reachable. No interview UI yet.

Run with `npm run dev` from `frontend/` (default: http://localhost:5173).

## Configuration

All secrets/environment values are read from `backend/.env` (copy from
`backend/.env.example`, never commit the real file). Key variables:

- `ANTHROPIC_API_KEY`, `MODEL_CHEAP`, `MODEL_STRONG` — Claude API access and
  model routing.
- `DATABASE_URL` — SQLite path (defaults to `database/interview_agent.db`
  relative to the project root, resolved correctly regardless of the
  directory `uvicorn` is started from).
- `CORS_ORIGINS` — origins the frontend dev server runs on.
- `CODE_EXECUTION_PROVIDER`, `JUDGE0_API_URL`, `JUDGE0_API_KEY`,
  `PISTON_API_URL` — reserved for the coding stage; not wired up yet.

## Explicitly out of scope for Day 1

- The actual aptitude / coding / technical / HR interview flows.
- A hardcoded question bank, or scraping questions from anywhere.
- Judge0/Piston code execution integration.
- The 20-persona HR evaluation approach.

These are foundation-only: routing, persistence, config, and placeholder
shapes for the pieces above.
