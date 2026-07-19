from agents.base_agent import BaseAgent
from agents.tools import TOOL_SCHEMAS, doc_fetch, web_search
from schemas import AgentRole


class ResearcherAgent(BaseAgent):
    ROLE = AgentRole.RESEARCHER
    MODEL = "qwen-plus"
    TOOLS = [TOOL_SCHEMAS["web_search"], TOOL_SCHEMAS["doc_fetch"]]

    def register_tools(self):
        self._executors = {
            "web_search": web_search,
            "doc_fetch": doc_fetch,
        }
