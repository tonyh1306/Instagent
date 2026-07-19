from agents.base_agent import BaseAgent
from schemas import AgentRole, TaskNode

SUBMIT_DECISION_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_decision",
        "description": "Submit your final, binding decision on the escalated task. This ends the council.",
        "parameters": {
            "type": "object",
            "properties": {
                "decision": {"type": "string", "enum": ["accept", "reject", "accept_with_repair"]},
                "rationale": {"type": "string"},
                "repair_instructions": {
                    "type": "string",
                    "description": (
                        "Required when decision=='accept_with_repair': specific, actionable synthesis "
                        "guidance for the one bounded final repair attempt (e.g. keep the worker's "
                        "approach but add the specific thing the critic needs)."
                    ),
                },
            },
            "required": ["decision", "rationale"],
        },
    },
}


class ArbiterAgent(BaseAgent):
    """Pure judgment: no tools besides the forced decision submission. Reads the council
    transcript once and rules — never votes, never asks for another round."""

    ROLE = AgentRole.ARBITER
    MODEL = "qwen-max"
    TOOLS = [SUBMIT_DECISION_TOOL]
    TERMINAL_TOOLS = {"submit_decision"}
    FORCE_TOOL = "submit_decision"

    def system_prompt(self) -> str:
        return (
            "You are the arbiter agent. You read a bounded council transcript of typed turns "
            "(claim/evidence/concession/rebuttal) where a worker agent defended its artifact "
            "and a critic pressed its concerns over up to two rounds. Make one final, binding "
            "decision:\n"
            "- 'accept': the artifact is fine as-is despite the critic's concerns.\n"
            "- 'reject': the critic's concerns stand and the artifact goes to the dead-letter queue.\n"
            "- 'accept_with_repair': the worker's core approach is sound but the critic identified "
            "something specific and fixable — synthesize both positions into concrete "
            "repair_instructions for one final, bounded repair attempt. Prefer this over a flat "
            "accept/reject when the transcript shows a real, narrow, fixable gap rather than a "
            "fundamental disagreement.\n"
            "Do not ask for more discussion — the round budget is exhausted. Call submit_decision "
            "exactly once."
        )

    def task_prompt(self, task: TaskNode, context: str) -> str:
        return (
            f"Task under dispute: {task.description}\n"
            f"Acceptance criteria: {task.acceptance_criteria}\n\n"
            f"Council transcript:\n{context}"
        )
