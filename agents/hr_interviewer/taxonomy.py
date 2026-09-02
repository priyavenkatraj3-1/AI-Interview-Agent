"""
HR round configuration: behavioral topic/pattern taxonomy, difficulty
scale, and per-company topic mix.

Mirrors agents/technical_interviewer/taxonomy.py's shape (plain data, no
DB, no Claude calls, single source of truth for "what counts as a valid
topic/pattern/difficulty") but is kept as its own module rather than
shared across stages, since each stage-agent package is self-contained per
docs/architecture.md.
"""
from __future__ import annotations

import random

# MVP scope: 5 free-text HR/behavioral questions per round.
TOTAL_QUESTIONS = 5

MIN_DIFFICULTY = 1
MAX_DIFFICULTY = 5
START_DIFFICULTY = 3

# For HR, "difficulty" tracks question depth/complexity — e.g. a simple
# warm-up question at 1 vs. a multi-part situational/conflict scenario at 5
# — rather than technical hardness.
DIFFICULTY_LABELS: dict[int, str] = {
    1: "very easy",
    2: "easy",
    3: "medium",
    4: "hard",
    5: "very hard",
}

# A free-text answer scoring at/above this (0-100) counts as correct for
# scoring/difficulty-progression purposes (see agents/hr_interviewer/grader.py).
PASS_THRESHOLD = 60

SUPPORTED_COMPANIES = ["TCS_NQT", "INFOSYS", "WIPRO"]

# topic -> patterns/sub-types within it. Common HR/behavioral interview
# categories, kept small (2 patterns/topic) so the mock fixture bank (see
# hr_interviewer.py) can cover every combination.
HR_TAXONOMY: dict[str, list[str]] = {
    "self_introduction": ["tell_me_about_yourself", "strengths_and_weaknesses"],
    "teamwork": ["team_conflict", "collaboration_example"],
    "leadership_and_ownership": ["taking_initiative", "handling_failure"],
    "career_motivation": ["why_this_company", "career_goals"],
    "pressure_and_adaptability": ["handling_pressure", "adapting_to_change"],
}

# How many of the TOTAL_QUESTIONS come from each topic, per company. Rough
# approximation of each company's published HR-round emphasis — not exact,
# just enough to make company selection mean something (mirrors
# agents/technical_interviewer/taxonomy.py's COMPANY_TOPIC_COUNTS).
COMPANY_TOPIC_COUNTS: dict[str, dict[str, int]] = {
    "TCS_NQT": {
        "self_introduction": 1,
        "teamwork": 1,
        "leadership_and_ownership": 1,
        "career_motivation": 1,
        "pressure_and_adaptability": 1,
    },
    "INFOSYS": {
        "self_introduction": 1,
        "teamwork": 2,
        "leadership_and_ownership": 1,
        "career_motivation": 1,
        "pressure_and_adaptability": 0,
    },
    "WIPRO": {
        "self_introduction": 1,
        "teamwork": 1,
        "leadership_and_ownership": 0,
        "career_motivation": 1,
        "pressure_and_adaptability": 2,
    },
}

for _company, _counts in COMPANY_TOPIC_COUNTS.items():
    assert sum(_counts.values()) == TOTAL_QUESTIONS, _company
    assert set(_counts.keys()) == set(HR_TAXONOMY.keys()), _company


def build_topic_sequence(target_company: str) -> list[str]:
    """Return a length-TOTAL_QUESTIONS list of topic names for one session,
    interleaved (not blocked) so e.g. all teamwork questions aren't asked
    back-to-back."""
    counts = dict(COMPANY_TOPIC_COUNTS.get(target_company, COMPANY_TOPIC_COUNTS["TCS_NQT"]))
    order = list(counts.keys())
    sequence: list[str] = []
    while sum(counts.values()) > 0:
        for topic in order:
            if counts[topic] > 0:
                sequence.append(topic)
                counts[topic] -= 1
    return sequence


def pick_pattern(topic: str, used_patterns: set[str], rng: random.Random) -> str:
    """Pick a pattern for `topic` that hasn't been used yet in this session
    if possible, falling back to reuse once every pattern has been used."""
    candidates = [p for p in HR_TAXONOMY[topic] if p not in used_patterns]
    if not candidates:
        candidates = HR_TAXONOMY[topic]
    return rng.choice(candidates)


def session_rng(session_id: str, index: int) -> random.Random:
    """Deterministic-per-question RNG so pattern selection is reproducible
    for a given session without needing a stored random state."""
    return random.Random(f"hr:{session_id}:{index}")


def clamp_difficulty(level: int) -> int:
    return max(MIN_DIFFICULTY, min(MAX_DIFFICULTY, level))
