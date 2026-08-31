# AI Interview Agent

An AI-driven placement interview simulator (TCS NQT / Infosys / Wipro style),
built as a set of stage agents (aptitude → coding → technical → HR)
coordinated by an orchestrator, sitting behind a FastAPI backend and a React
frontend.

> **Status: Day 1 foundation.** Project scaffolding, persistence, config, and
> placeholder agent shapes only. No interview stage is implemented yet — see
> [`docs/architecture.md`](docs/architecture.md) for the full breakdown and
> what's intentionally out of scope for now.

## Repo layout

```
agents/       Stage-agent Python package (orchestrator + 4 stage agents)
backend/      FastAPI application
database/     SQLite database (git-ignored) + migrations
docs/         Documentation
frontend/     React (Vite) frontend
tests/        Test suite
```

## Prerequisites

- Python 3.11+
- Node.js 20+
- An Anthropic API key (for later stages — not required to run the Day 1 health check)

## Backend setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
copy .env.example .env        # Windows: copy, macOS/Linux: cp
```

Fill in `backend/.env` (at minimum leave the defaults; add `ANTHROPIC_API_KEY`
once you have one — not needed for Day 1).

Run the API:

```bash
uvicorn app.main:app --reload --port 8000
```

- API root: http://127.0.0.1:8000/
- Interactive docs: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

The SQLite database is created automatically at `database/interview_agent.db`
on first startup.

## Frontend setup

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 — it calls the backend's `/health` endpoint and
shows whether the API is reachable.

## Configuration

All configuration is via environment variables loaded from `backend/.env`
(see `backend/.env.example` for the full list). Never commit `backend/.env`.

## Testing

`tests/` is scaffolded but not yet populated — test coverage is added
alongside each stage as it's implemented.

## Roadmap (not yet built)

- Aptitude, coding, technical, and HR interview stages
- AI-generated questions (no hardcoded bank, no scraping)
- Judge0/Piston code execution for the coding stage
- Multi-persona HR evaluation
- Final scorecard, remediation plan, and hire/no-hire verdict
