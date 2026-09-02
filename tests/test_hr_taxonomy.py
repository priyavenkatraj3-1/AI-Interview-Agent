"""Unit tests for agents/hr_interviewer/taxonomy.py — pure logic, no DB/API."""
import random

from agents.hr_interviewer.taxonomy import (
    COMPANY_TOPIC_COUNTS,
    HR_TAXONOMY,
    MAX_DIFFICULTY,
    MIN_DIFFICULTY,
    SUPPORTED_COMPANIES,
    TOTAL_QUESTIONS,
    build_topic_sequence,
    clamp_difficulty,
    pick_pattern,
    session_rng,
)


def test_topic_sequence_has_exactly_total_questions_for_every_company():
    for company in SUPPORTED_COMPANIES:
        sequence = build_topic_sequence(company)
        assert len(sequence) == TOTAL_QUESTIONS


def test_topic_sequence_only_uses_known_topics():
    for company in SUPPORTED_COMPANIES:
        sequence = build_topic_sequence(company)
        assert set(sequence) <= set(HR_TAXONOMY.keys())


def test_topic_sequence_matches_company_counts():
    for company in SUPPORTED_COMPANIES:
        sequence = build_topic_sequence(company)
        for topic, expected_count in COMPANY_TOPIC_COUNTS[company].items():
            assert sequence.count(topic) == expected_count


def test_unknown_company_falls_back_to_tcs_nqt():
    fallback = build_topic_sequence("UNKNOWN_CO")
    tcs = build_topic_sequence("TCS_NQT")
    assert fallback == tcs


def test_pick_pattern_avoids_used_patterns_when_possible():
    rng = random.Random(42)
    topic = "teamwork"
    all_patterns = set(HR_TAXONOMY[topic])
    used = set(list(all_patterns)[:-1])  # all but one used
    pattern = pick_pattern(topic, used, rng)
    assert pattern not in used


def test_pick_pattern_falls_back_to_reuse_once_exhausted():
    rng = random.Random(1)
    topic = "career_motivation"
    used = set(HR_TAXONOMY[topic])  # every pattern already used
    pattern = pick_pattern(topic, used, rng)
    assert pattern in HR_TAXONOMY[topic]


def test_session_rng_is_deterministic_per_session_and_index():
    rng1 = session_rng("session-abc", 3)
    rng2 = session_rng("session-abc", 3)
    assert rng1.random() == rng2.random()


def test_session_rng_differs_across_index():
    a = session_rng("session-abc", 0).random()
    b = session_rng("session-abc", 1).random()
    assert a != b


def test_clamp_difficulty_stays_within_bounds():
    assert clamp_difficulty(MIN_DIFFICULTY - 5) == MIN_DIFFICULTY
    assert clamp_difficulty(MAX_DIFFICULTY + 5) == MAX_DIFFICULTY
    assert clamp_difficulty(3) == 3
