from agents.base_agent import BaseAgent
from agents.tools import TOOL_SCHEMAS, file_write, lint, run_tests
from schemas import AgentRole


class CoderAgent(BaseAgent):
    ROLE = AgentRole.CODER
    MODEL = "qwen-plus"
    TOOLS = [TOOL_SCHEMAS["file_write"], TOOL_SCHEMAS["run_tests"], TOOL_SCHEMAS["lint"]]
    MAX_ITERATIONS = 10

    def register_tools(self):
        self._executors = {
            "file_write": file_write,
            "run_tests": run_tests,
            "lint": lint,
        }
