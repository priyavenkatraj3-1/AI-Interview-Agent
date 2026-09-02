"""Unit tests for agents/code_problem_generator/taxonomy.py — pure logic, no DB/API."""
import random

from agents.code_problem_generator.taxonomy import (
    CODING_TAXONOMY,
    MAX_DIFFICULTY,
    MIN_DIFFICULTY,
    TOTAL_PROBLEMS,
    build_topic_sequence,
    clamp_difficulty,
    pick_pattern,
    session_rng,
)


def test_topic_sequence_has_exactly_total_problems():
    sequence = build_topic_sequence(TOTAL_PROBLEMS)
    assert len(sequence) == TOTAL_PROBLEMS


def test_topic_sequence_only_uses_known_topics():
    sequence = build_topic_sequence(TOTAL_PROBLEMS)
    assert set(sequence) <= set(CODING_TAXONOMY.keys())


def test_topic_sequence_has_no_repeats_when_count_within_topic_pool():
    sequence = build_topic_sequence(TOTAL_PROBLEMS)
    assert len(set(sequence)) == len(sequence)


def test_topic_sequence_round_robins_when_count_exceeds_topic_pool():
    topics = list(CODING_TAXONOMY.keys())
    count = len(topics) + 2
    sequence = build_topic_sequence(count)
    assert len(sequence) == count
    assert sequence == [topics[i % len(topics)] for i in range(count)]


def test_pick_pattern_avoids_used_patterns_when_possible():
    rng = random.Random(42)
    topic = "arrays"
    all_patterns = set(CODING_TAXONOMY[topic])
    used = set(list(all_patterns)[:-1])  # all but one used
    pattern = pick_pattern(topic, used, rng)
    assert pattern not in used


def test_pick_pattern_falls_back_to_reuse_once_exhausted():
    rng = random.Random(1)
    topic = "strings"
    used = set(CODING_TAXONOMY[topic])  # every pattern already used
    pattern = pick_pattern(topic, used, rng)
    assert pattern in CODING_TAXONOMY[topic]


def test_session_rng_is_deterministic_per_session_and_index():
    rng1 = session_rng("session-abc", 1)
    rng2 = session_rng("session-abc", 1)
    assert rng1.random() == rng2.random()


def test_session_rng_differs_across_index():
    a = session_rng("session-abc", 0).random()
    b = session_rng("session-abc", 1).random()
    assert a != b


def test_clamp_difficulty_stays_within_bounds():
    assert clamp_difficulty(MIN_DIFFICULTY - 5) == MIN_DIFFICULTY
    assert clamp_difficulty(MAX_DIFFICULTY + 5) == MAX_DIFFICULTY
    assert clamp_difficulty(3) == 3
