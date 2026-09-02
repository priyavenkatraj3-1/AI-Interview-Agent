"""
Difficulty-vs-ability scatter plot for the mandatory evaluation (see
evaluation/harness.py for how the underlying data is collected).

Matplotlib only (no seaborn), saved as a plain PNG. This is the only place
in the whole project that depends on matplotlib — see
backend/requirements-dev.txt, where it's listed as an evaluation-only
dependency, never imported by the running application (agents/, backend/,
frontend/).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display/GUI backend needed to save a PNG
import matplotlib.pyplot as plt

ROUND_COLORS = {
    "aptitude": "tab:blue",
    "coding": "tab:orange",
    "technical": "tab:green",
    "hr": "tab:red",
}


def plot_difficulty_vs_ability(points: list[dict], output_path: Path) -> Path:
    """`points`: a list of {"round": str, "ability": float, "difficulty": float}
    (one entry per answered question). Saves a scatter plot (ability on the
    X-axis, difficulty on the Y-axis) to `output_path` as a PNG and returns
    the path."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for round_name, color in ROUND_COLORS.items():
        round_points = [p for p in points if p["round"] == round_name]
        if not round_points:
            continue
        ax.scatter(
            [p["ability"] for p in round_points],
            [p["difficulty"] for p in round_points],
            label=round_name.capitalize(),
            color=color,
            alpha=0.6,
        )

    ax.set_xlabel("Persona ability score (0-1)")
    ax.set_ylabel("Question difficulty (1-5)")
    ax.set_title("Question Difficulty vs Persona Ability")
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
