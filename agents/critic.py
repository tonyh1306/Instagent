from agents.base_agent import BaseAgent
from agents.tools import TOOL_SCHEMAS, file_read, run_tests, schema_validate
from schemas import AgentRole

SUBMIT_VERDICT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_verdict",
        "description": "Submit your final pass/fail verdict on the artifact under review. This ends your review.",
        "parameters": {
            "type": "object",
            "properties": {
                "passed": {"type": "boolean"},
                "reasons": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Structured reasons; required and non-empty when passed=false.",
                },
            },
            "required": ["passed", "reasons"],
        },
    },
}


class CriticAgent(BaseAgent):
    """Read-only reviewer: no mutation tools, only inspection tools + a verdict submission tool."""

    ROLE = AgentRole.CRITIC
    MODEL = "qwen-plus"
    TOOLS = [
        TOOL_SCHEMAS["file_read"],
        TOOL_SCHEMAS["schema_validate"],
        TOOL_SCHEMAS["run_tests"],
        SUBMIT_VERDICT_TOOL,
    ]
    TERMINAL_TOOLS = {"submit_verdict"}
    MAX_ITERATIONS = 10

    def register_tools(self):
        self._executors = {
            "file_read": file_read,
            "schema_validate": schema_validate,
            "run_tests": run_tests,
        }

    def system_prompt(self) -> str:
        return (
            "You are the critic agent in a multi-agent system. You review another agent's "
            "artifact strictly against the acceptance_criteria listed for its task — nothing "
            "more. Do not invent additional requirements (specific extra test values, style "
            "preferences, etc.) that aren't in that list; if a criterion is satisfied in spirit, "
            "pass it. The artifact's own final message is a summary only — if it references "
            "files it wrote, use file_read to inspect their actual content, and run_tests to "
            "confirm behavior, before judging. You have no mutation tools — you cannot fix "
            "anything yourself. When done, call submit_verdict with passed=true/false and, if "
            "false, reasons that each cite the specific acceptance criterion being violated."
        )
