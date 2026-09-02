"""
Central configuration for the backend.

All secrets and environment-specific values are read from environment
variables (populated from backend/.env in local development). Nothing here
is hardcoded — see backend/.env.example for the full list of variables and
backend/.env for actual local values (git-ignored).
"""
import sys
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/core/config.py -> backend/app/core -> backend/app -> backend -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = PROJECT_ROOT / "backend"
DATABASE_DIR = PROJECT_ROOT / "database"

# uvicorn is run from backend/, so the project root (where the standalone
# agents/ package lives, per docs/architecture.md) isn't on sys.path by
# default. Add it once here, since this module is imported before anything
# in app/ that needs to `import agents`.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App ---
    app_env: str = "development"
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # --- Anthropic / Claude ---
    anthropic_api_key: str = ""
    model_cheap: str = "claude-haiku-4-5-20251001"
    model_strong: str = "claude-sonnet-5"

    # --- Coding sandbox (wired up in a later stage) ---
    code_execution_provider: str = "judge0"
    judge0_api_url: str = ""
    judge0_api_key: str = ""
    piston_api_url: str = ""

    # --- Mock mode (Day 3: coding round) ---
    # When true (the default — no Anthropic API credits currently
    # available), Claude-backed stage agents that support it use a
    # deterministic offline mock instead of a real API call. Mirrors
    # agents/config.py's MOCK_MODE, which is what agent construction
    # actually reads (kept separate per the agents/ <-> backend/ boundary
    # documented in docs/architecture.md).
    mock_mode: bool = True

    # --- Database ---
    database_url: str = f"sqlite:///{(DATABASE_DIR / 'interview_agent.db').as_posix()}"

    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def resolved_database_url(self) -> str:
        """
        Resolve a relative sqlite:/// URL against the project root so the app
        behaves the same whether uvicorn is started from backend/ or elsewhere.
        """
        prefix = "sqlite:///"
        if self.database_url.startswith(prefix) and not self.database_url.startswith(f"{prefix}/"):
            relative_path = self.database_url[len(prefix):]
            if not Path(relative_path).is_absolute():
                absolute_path = (PROJECT_ROOT / relative_path).resolve()
                return f"{prefix}{absolute_path.as_posix()}"
        return self.database_url


@lru_cache
def get_settings() -> Settings:
    return Settings()
