"""
Orchestrator: drives an InterviewSession through its stages and delegates
actual work to the stage agents.

STAGE_SEQUENCE and get_next_stage() are real (Day 1: this is just
architecture, no AI calls). Everything that requires calling an agent or
compiling final results is a placeholder for later days.

Responsibilities (see docs/architecture.md for the full flow):
- Own the stage state machine: aptitude -> coding -> technical -> hr -> completed
- Decide which agent to invoke for the session's current stage
- Persist stage transitions via StageProgress rows
- Aggregate per-stage results into the final scorecard + remediation plan + verdict
"""
from agents.base import AgentResult

STAGE_SEQUENCE = ["aptitude", "coding", "technical", "hr", "completed"]


def get_next_stage(current_stage: str) -> str:
    """Pure state-machine step. Returns 'completed' once HR is done."""
    if current_stage not in STAGE_SEQUENCE:
        raise ValueError(f"Unknown stage: {current_stage!r}")
    idx = STAGE_SEQUENCE.index(current_stage)
    return STAGE_SEQUENCE[min(idx + 1, len(STAGE_SEQUENCE) - 1)]


class Orchestrator:
    """Coordinates the four stage agents across the lifetime of one session."""

    def __init__(self):
        # Wiring to concrete agent instances happens once those agents are
        # implemented (question generator, grader, technical/HR interviewers).
        pass

    async def run_current_stage(self, session_state: dict) -> AgentResult:
        """
        Invoke whichever agent owns session_state['current_stage'] and return
        its structured result.

        Not implemented yet — placeholder for Day 1.
        """
        raise NotImplementedError("Orchestrator.run_current_stage() is a Day 1 placeholder.")

    async def compile_final_report(self, session_state: dict) -> AgentResult:
        """
        Aggregate all stage results into: scorecard, 14-day remediation plan,
        and the hire/no-hire verdict paragraph.

        Not implemented yet — placeholder for Day 1.
        """
        raise NotImplementedError("Orchestrator.compile_final_report() is a Day 1 placeholder.")
