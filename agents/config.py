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

# Coding-round sandbox provider (see agents/code_executor/dispatcher.py).
# "local" (default) keeps candidate code execution on the local subprocess
# executor; "piston" routes it through the real Piston sandbox instead.
# Only takes effect when MOCK_MODE=false -- MOCK_MODE always uses the local
# executor regardless of this value, same as every other mock fallback in
# this codebase.
CODE_EXECUTION_PROVIDER = os.getenv("CODE_EXECUTION_PROVIDER", "local").strip().lower()

# Base URL of the Piston instance to use when CODE_EXECUTION_PROVIDER=piston
# (e.g. a self-hosted instance, or the public https://emkc.org/api/v2/piston).
# No default: never hardcode a real sandbox endpoint in code.
PISTON_API_URL = os.getenv("PISTON_API_URL", "").strip()
