"""
SQLite engine/session setup via SQLAlchemy.

Interview state must survive a page refresh, so every session and its
stage progress is persisted here rather than kept in memory. The actual
.db file lives in the top-level database/ folder (see config.py), keeping
storage out of both backend/ and frontend/.
"""
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import DATABASE_DIR, get_settings

settings = get_settings()

DATABASE_DIR.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    settings.resolved_database_url,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency that yields a DB session and closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables if they don't exist yet. Called once on app startup."""
    from app.models import session as _session_models  # noqa: F401  (registers models)

    Base.metadata.create_all(bind=engine)
