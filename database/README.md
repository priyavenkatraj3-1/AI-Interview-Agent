# database/

Holds the local SQLite database for the project.

- `interview_agent.db` — created automatically the first time the backend
  starts (see `backend/app/core/database.py`). It is git-ignored; nothing
  here needs to be created by hand.
- `migrations/` — reserved for schema migrations (e.g. Alembic) once the
  schema needs versioned changes. Empty on Day 1 because `init_db()` just
  calls `Base.metadata.create_all()` on startup, which is sufficient while
  the schema is still evolving.

Table definitions live in `backend/app/models/session.py`
(`InterviewSession`, `StageProgress`, `CostLog`) — see `docs/architecture.md`
for how they fit together.
