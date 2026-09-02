"""
Lightweight, standalone config for the agents package.

Agents are deliberately decoupled from the backend web framework so they
can be imported and tested (or run from a script/notebook) without FastAPI
running. This reads environment variables directly instead of importing
across the agents/ <-> backend/ package boundary. It mirrors the relevant
subset of backend/app/core/config.py — see docs/architecture.md for why the
two are kept separate.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / "backend" / ".env")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MODEL_CHEAP = os.getenv("MODEL_CHEAP", "claude-haiku-4-5-20251001")
MODEL_STRONG = os.getenv("MODEL_STRONG", "claude-sonnet-5")

# When true (the default — this project currently has no Anthropic API
# credits), Claude-backed stage agents that support it construct a
# deterministic offline mock implementation instead of a real API client
# (see agents/code_problem_generator/code_problem_generator.py). Set
# MOCK_MODE=false once credits/a valid ANTHROPIC_API_KEY are available.
MOCK_MODE = os.getenv("MOCK_MODE", "true").strip().lower() in ("1", "true", "yes", "on")
