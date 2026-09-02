"""
20 synthetic student personas for the mandatory evaluation.

Each persona is a simple ability profile — one probability-of-correctness
value per round, in [0, 1] — used by evaluation/harness.py to decide what
answer to submit at each question. This is a deliberate simplification: a
real IRT model would let difficulty interact with ability; here each
question is treated as an independent trial at the persona's fixed
ability. This simplification is documented in docs/evaluation_report.md,
not hidden.

`target_company` rotates through the three supported companies so the
evaluation also exercises each company's topic-mix taxonomy.
"""
from __future__ import annotations

from dataclasses import dataclass

SUPPORTED_COMPANIES = ["TCS_NQT", "INFOSYS", "WIPRO"]


@dataclass(frozen=True)
class Persona:
    name: str
    description: str
    aptitude_ability: float
    coding_ability: float
    technical_ability: float
    hr_ability: float
    target_company: str

    @property
    def ability_by_round(self) -> dict[str, float]:
        return {
            "aptitude": self.aptitude_ability,
            "coding": self.coding_ability,
            "technical": self.technical_ability,
            "hr": self.hr_ability,
        }


def _company_for(index: int) -> str:
    return SUPPORTED_COMPANIES[index % len(SUPPORTED_COMPANIES)]


PERSONAS: list[Persona] = [
    Persona("all_rounder_high", "Strong across all four rounds.", 0.95, 0.95, 0.95, 0.95, _company_for(0)),
    Persona("all_rounder_low", "Weak across all four rounds.", 0.10, 0.10, 0.10, 0.10, _company_for(1)),
    Persona("balanced_medium", "Consistently average across all rounds.", 0.55, 0.55, 0.55, 0.55, _company_for(2)),
    Persona("aptitude_specialist", "Excels at aptitude only; weak elsewhere.", 0.90, 0.20, 0.20, 0.20, _company_for(3)),
    Persona("coding_specialist", "Excels at coding only; weak elsewhere.", 0.20, 0.90, 0.20, 0.20, _company_for(4)),
    Persona("technical_specialist", "Excels at technical only; weak elsewhere.", 0.20, 0.20, 0.90, 0.20, _company_for(5)),
    Persona("hr_specialist", "Excels at HR/behavioral only; weak elsewhere.", 0.20, 0.20, 0.20, 0.90, _company_for(6)),
    Persona("coder_not_communicator", "Strong technical/coding, weak HR/soft-skills.", 0.70, 0.90, 0.80, 0.15, _company_for(7)),
    Persona("communicator_not_coder", "Strong HR/aptitude, weak coding/technical.", 0.75, 0.15, 0.25, 0.90, _company_for(8)),
    Persona("crammer", "Strong rote aptitude, weak applied coding/technical.", 0.85, 0.30, 0.35, 0.50, _company_for(9)),
    Persona("late_bloomer", "Weak aptitude, strong coding/technical/HR.", 0.25, 0.80, 0.80, 0.70, _company_for(10)),
    Persona("steady_improver", "Moderate-to-good across the board.", 0.50, 0.60, 0.65, 0.70, _company_for(11)),
    Persona("front_loaded", "Strong aptitude/coding, fades on technical/HR.", 0.85, 0.70, 0.40, 0.20, _company_for(12)),
    Persona("consistently_good", "High accuracy across every round.", 0.88, 0.85, 0.82, 0.80, _company_for(0)),
    Persona("guesser", "Near-chance performance everywhere.", 0.25, 0.25, 0.25, 0.25, _company_for(1)),
    Persona("technical_and_hr_strong", "Strong technical + HR, average aptitude/coding.", 0.55, 0.55, 0.85, 0.85, _company_for(2)),
    Persona("aptitude_and_coding_strong", "Strong aptitude + coding, average technical/HR.", 0.85, 0.85, 0.55, 0.55, _company_for(3)),
    Persona("moderately_capable", "Ability around 0.65 across all rounds.", 0.65, 0.65, 0.65, 0.65, _company_for(4)),
    Persona("moderately_weak", "Ability around 0.45 across all rounds.", 0.45, 0.45, 0.45, 0.45, _company_for(5)),
    Persona("exceptional", "Near-perfect across every round.", 0.99, 0.99, 0.99, 0.99, _company_for(6)),
]

assert len(PERSONAS) == 20
assert len({p.name for p in PERSONAS}) == 20  # names must be unique
