"""
Coding round configuration: topic/pattern taxonomy and difficulty scale.

Mirrors agents/question_generator/taxonomy.py's shape (plain data, no DB, no
Claude calls, single source of truth for "what counts as a valid
topic/pattern/difficulty") but is kept as its own module rather than shared
across stages, since each stage-agent package is self-contained per
docs/architecture.md.
"""
from __future__ import annotations

import random

# MVP scope: 2 coding problems per round (vs. aptitude's 15 — coding
# problems take much longer to read/write/debug per problem).
TOTAL_PROBLEMS = 2

MIN_DIFFICULTY = 1
MAX_DIFFICULTY = 5
START_DIFFICULTY = 3

DIFFICULTY_LABELS: dict[int, str] = {
    1: "very easy",
    2: "easy",
    3: "medium",
    4: "hard",
    5: "very hard",
}

# topic -> patterns/sub-types within it. Small on purpose for the MVP; the
# mock fixture bank (see code_problem_generator.py) has one hand-verified
# problem per pattern below.
CODING_TAXONOMY: dict[str, list[str]] = {
    "arrays": ["two_sum", "max_subarray"],
    "strings": ["reverse_string", "valid_palindrome"],
    "recursion": ["factorial", "fibonacci"],
}


def build_topic_sequence(count: int = TOTAL_PROBLEMS) -> list[str]:
    """Return `count` topic names for one coding round. No company-based
    weighting (unlike aptitude) — coding topics are general CS fundamentals,
    not company-specific. Distinct topics while `count` stays within the
    topic pool; round-robins if `count` ever exceeds it."""
    topics = list(CODING_TAXONOMY.keys())
    return [topics[i % len(topics)] for i in range(count)]


def pick_pattern(topic: str, used_patterns: set[str], rng: random.Random) -> str:
    """Pick a pattern for `topic` that hasn't been used yet in this session
    if possible, falling back to reuse once every pattern has been used."""
    candidates = [p for p in CODING_TAXONOMY[topic] if p not in used_patterns]
    if not candidates:
        candidates = CODING_TAXONOMY[topic]
    return rng.choice(candidates)


def session_rng(session_id: str, index: int) -> random.Random:
    """Deterministic-per-problem RNG so pattern selection is reproducible
    for a given session without needing a stored random state."""
    return random.Random(f"coding:{session_id}:{index}")


def clamp_difficulty(level: int) -> int:
    return max(MIN_DIFFICULTY, min(MAX_DIFFICULTY, level))
