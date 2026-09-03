"""
Technical round configuration: topic/pattern taxonomy, difficulty scale,
and per-company topic mix.

Mirrors agents/question_generator/taxonomy.py's shape (plain data, no DB,
no Claude calls, single source of truth for "what counts as a valid
topic/pattern/difficulty") but is kept as its own module rather than shared
across stages, since each stage-agent package is self-contained per
docs/architecture.md.
"""
from __future__ import annotations

import random

# MVP scope: 5 free-text technical questions per round.
TOTAL_QUESTIONS = 5

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

# A free-text answer scoring at/above this (0-100) counts as correct for
# scoring/difficulty-progression purposes (see agents/technical_interviewer/grader.py).
PASS_THRESHOLD = 60

# Offline/mock Socratic-probing heuristic (see MockTechnicalInterviewerAgent
# in technical_interviewer.py): a previous answer under this many words is
# treated as too short/vague to have adequately explained itself, and may
# trigger one same-topic "why?" follow-up question instead of moving on.
# Not used by the real Claude-backed interviewer, which judges weakness
# itself from the actual answer content rather than a word count.
WEAK_ANSWER_WORD_THRESHOLD = 6

SUPPORTED_COMPANIES = ["TCS_NQT", "INFOSYS", "WIPRO"]

# topic -> patterns/sub-types within it. Mirrors the broad shape of a
# TCS NQT / Infosys / Wipro technical round without hardcoding any actual
# questions. Kept small (2 patterns/topic) so the mock fixture bank (see
# technical_interviewer.py) can cover every combination.
TECHNICAL_TAXONOMY: dict[str, list[str]] = {
    "data_structures": ["arrays_and_strings", "stacks_and_queues"],
    "algorithms": ["sorting_and_searching", "recursion"],
    "oop": ["encapsulation_and_abstraction", "inheritance_and_polymorphism"],
    "dbms": ["sql_queries", "normalization"],
    "os_and_networks": ["processes_and_threads", "networking_basics"],
}

# How many of the TOTAL_QUESTIONS come from each topic, per company. Rough
# approximation of each company's published technical-round emphasis — not
# exact, just enough to make company selection mean something (mirrors
# agents/question_generator/taxonomy.py's COMPANY_TOPIC_COUNTS).
COMPANY_TOPIC_COUNTS: dict[str, dict[str, int]] = {
    "TCS_NQT": {"data_structures": 1, "algorithms": 1, "oop": 1, "dbms": 1, "os_and_networks": 1},
    "INFOSYS": {"data_structures": 1, "algorithms": 1, "oop": 1, "dbms": 2, "os_and_networks": 0},
    "WIPRO": {"data_structures": 2, "algorithms": 1, "oop": 0, "dbms": 1, "os_and_networks": 1},
}

for _company, _counts in COMPANY_TOPIC_COUNTS.items():
    assert sum(_counts.values()) == TOTAL_QUESTIONS, _company
    assert set(_counts.keys()) == set(TECHNICAL_TAXONOMY.keys()), _company


def build_topic_sequence(target_company: str) -> list[str]:
    """Return a length-TOTAL_QUESTIONS list of topic names for one session,
    interleaved (not blocked) so e.g. all DBMS questions aren't asked
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
    candidates = [p for p in TECHNICAL_TAXONOMY[topic] if p not in used_patterns]
    if not candidates:
        candidates = TECHNICAL_TAXONOMY[topic]
    return rng.choice(candidates)


def session_rng(session_id: str, index: int) -> random.Random:
    """Deterministic-per-question RNG so pattern selection is reproducible
    for a given session without needing a stored random state."""
    return random.Random(f"technical:{session_id}:{index}")


def clamp_difficulty(level: int) -> int:
    return max(MIN_DIFFICULTY, min(MAX_DIFFICULTY, level))
