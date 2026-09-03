"""
Aptitude round configuration: topic/pattern taxonomy, difficulty scale, and
per-company topic mix.

Kept as plain data (no DB, no Claude calls) so the backend service and the
generator agent share a single source of truth for "what counts as a valid
topic/pattern/difficulty" without duplicating literals.
"""
from __future__ import annotations

import random

TOTAL_QUESTIONS = 15

MIN_DIFFICULTY = 1
MAX_DIFFICULTY = 5
START_DIFFICULTY = 3

# Backend-enforced per-question time limit (seconds). Checked server-side in
# app.services.aptitude_service.submit_answer() against `presented_at` (set
# by the backend when the question was served, never client-supplied) vs.
# the wall-clock time the answer is actually submitted -- not merely a
# frontend display value the client could ignore. Exceeding this marks the
# question timed out and incorrect regardless of the submitted
# selected_option, exactly like a wrong answer for scoring/difficulty
# purposes.
MAX_TIME_PER_QUESTION_SECONDS = 90

DIFFICULTY_LABELS: dict[int, str] = {
    1: "very easy",
    2: "easy",
    3: "medium",
    4: "hard",
    5: "very hard",
}

SUPPORTED_COMPANIES = ["TCS_NQT", "INFOSYS", "WIPRO"]

# topic -> patterns/sub-types within it. Mirrors the broad shape of the
# TCS NQT / Infosys / Wipro aptitude sections (quant, logical reasoning,
# verbal ability) without hardcoding any actual questions.
APTITUDE_TAXONOMY: dict[str, list[str]] = {
    "quantitative": [
        "percentages",
        "profit_and_loss",
        "simple_and_compound_interest",
        "time_and_work",
        "time_speed_and_distance",
        "ratio_and_proportion",
        "averages",
        "number_series",
        "permutations_and_combinations",
        "probability",
        "ages",
        "mixtures_and_alligation",
    ],
    "logical_reasoning": [
        "series_completion",
        "coding_decoding",
        "blood_relations",
        "direction_sense",
        "syllogism",
        "seating_arrangement",
        "puzzles",
        "data_sufficiency",
        "statement_and_conclusion",
    ],
    "verbal_ability": [
        "synonyms_antonyms",
        "sentence_correction",
        "reading_comprehension",
        "para_jumbles",
        "fill_in_the_blanks",
        "error_spotting",
    ],
}

# How many of the 15 questions come from each topic, per company. Rough
# approximation of each company's published aptitude-section weighting —
# not exact, just enough to make company selection mean something.
COMPANY_TOPIC_COUNTS: dict[str, dict[str, int]] = {
    "TCS_NQT": {"quantitative": 6, "logical_reasoning": 5, "verbal_ability": 4},
    "INFOSYS": {"quantitative": 7, "logical_reasoning": 4, "verbal_ability": 4},
    "WIPRO": {"quantitative": 5, "logical_reasoning": 6, "verbal_ability": 4},
}

for _company, _counts in COMPANY_TOPIC_COUNTS.items():
    assert sum(_counts.values()) == TOTAL_QUESTIONS, _company
    assert set(_counts.keys()) == set(APTITUDE_TAXONOMY.keys()), _company


def build_topic_sequence(target_company: str) -> list[str]:
    """Return a length-TOTAL_QUESTIONS list of topic names for one session,
    interleaved (not blocked) so e.g. quant questions aren't all first."""
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
    if possible, so consecutive questions on the same topic don't repeat
    the same sub-type. Falls back to reuse once every pattern for the
    topic has been used (content-level duplicate prevention still applies
    on the generated question text itself)."""
    candidates = [p for p in APTITUDE_TAXONOMY[topic] if p not in used_patterns]
    if not candidates:
        candidates = APTITUDE_TAXONOMY[topic]
    return rng.choice(candidates)


def session_rng(session_id: str, index: int) -> random.Random:
    """Deterministic-per-question RNG so pattern selection is reproducible
    for a given session without needing a stored random state."""
    return random.Random(f"{session_id}:{index}")


def clamp_difficulty(level: int) -> int:
    return max(MIN_DIFFICULTY, min(MAX_DIFFICULTY, level))
