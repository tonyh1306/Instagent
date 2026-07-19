from agents.base_agent import BaseAgent
from agents.tools import TOOL_SCHEMAS, doc_format, file_write
from schemas import AgentRole


class WriterAgent(BaseAgent):
    ROLE = AgentRole.WRITER
    MODEL = "qwen-plus"
    TOOLS = [TOOL_SCHEMAS["file_write"], TOOL_SCHEMAS["doc_format"]]

    def register_tools(self):
        self._executors = {
            "file_write": file_write,
            "doc_format": doc_format,
        }
