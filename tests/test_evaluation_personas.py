"""Unit tests for evaluation/personas.py — pure data validation, no DB/API."""
from evaluation.personas import PERSONAS, SUPPORTED_COMPANIES


def test_there_are_exactly_twenty_personas():
    assert len(PERSONAS) == 20


def test_persona_names_are_unique():
    names = [p.name for p in PERSONAS]
    assert len(names) == len(set(names))


def test_every_ability_is_in_valid_range():
    for p in PERSONAS:
        for ability in (p.aptitude_ability, p.coding_ability, p.technical_ability, p.hr_ability):
            assert 0.0 <= ability <= 1.0


def test_every_persona_targets_a_supported_company():
    for p in PERSONAS:
        assert p.target_company in SUPPORTED_COMPANIES


def test_profiles_are_clearly_different_not_all_identical():
    profiles = {(p.aptitude_ability, p.coding_ability, p.technical_ability, p.hr_ability) for p in PERSONAS}
    # 20 distinct profiles -> genuinely different strength profiles, not copies.
    assert len(profiles) == 20


def test_at_least_one_single_skill_specialist_per_round():
    # A "specialist": high ability in exactly one round, low in the other three.
    def is_specialist_in(p, round_name):
        abilities = p.ability_by_round
        target = abilities[round_name]
        others = [v for k, v in abilities.items() if k != round_name]
        return target >= 0.8 and all(o <= 0.3 for o in others)

    for round_name in ("aptitude", "coding", "technical", "hr"):
        assert any(is_specialist_in(p, round_name) for p in PERSONAS), round_name
