"""Single-agent baseline: one agent, one broad toolset, no decomposition, no parallelism,
no critic gate. Used purely as a comparison point against the multi-agent pipeline so the
efficiency/quality claims in the dashboard are measured against something real, not asserted.
"""

import json
import time

from agents.base_agent import BaseAgent
from agents.tools import TOOL_SCHEMAS, doc_fetch, doc_format, file_write, lint, run_tests, web_search
from budget import RunBudget
from schemas import AgentRole, TaskNode

MAX_ITERATIONS = 20


class BaselineAgent(BaseAgent):
    ROLE = AgentRole.BASELINE
    MODEL = "qwen-plus"  # same tier as the multi-agent workers, for a fair comparison
    TOOLS = [
        TOOL_SCHEMAS["web_search"],
        TOOL_SCHEMAS["doc_fetch"],
        TOOL_SCHEMAS["file_write"],
        TOOL_SCHEMAS["run_tests"],
        TOOL_SCHEMAS["lint"],
        TOOL_SCHEMAS["doc_format"],
    ]
    MAX_ITERATIONS = MAX_ITERATIONS

    def register_tools(self):
        self._executors = {
            "web_search": web_search,
            "doc_fetch": doc_fetch,
            "file_write": file_write,
            "run_tests": run_tests,
            "lint": lint,
            "doc_format": doc_format,
        }

    def system_prompt(self) -> str:
        return (
            "You are a single agent responsible for the entire request end to end - research, "
            "implementation, testing, and documentation. There is no one else to delegate to and "
            "no reviewer checking your work, so verify your own output (e.g. run_tests) before "
            "declaring done."
        )

    def task_prompt(self, task: TaskNode, context: str) -> str:
        return task.description


def _last_run_tests_success(transcript: list[dict]) -> bool | None:
    for msg in reversed(transcript):
        if msg.get("role") != "tool":
            continue
        try:
            result = json.loads(msg["content"])
        except (json.JSONDecodeError, TypeError):
            continue
        if "returncode" in result and "stdout" in result:
            return result["returncode"] == 0
    return None


def run_baseline(user_task: str, run_id: str = "baseline", max_llm_calls: int = 60, max_tool_calls: int = 150) -> dict:
    """Runs the whole task through a single agent and returns timing + usage + a best-effort
    success signal (did its own last test run report returncode 0, if it ran tests at all)."""
    import redis as redis_lib

    r = redis_lib.from_url("redis://localhost:6379/0", decode_responses=True)
    budget = RunBudget(r, run_id, max_llm_calls=max_llm_calls, max_tool_calls=max_tool_calls)

    agent = BaselineAgent(budget=budget)
    task = TaskNode(id="__baseline__", description=user_task, assigned_agent=AgentRole.BASELINE)

    start = time.monotonic()
    try:
        result = agent.run(task)
        final_content = result.content or ""
        error = None
    except Exception as e:
        final_content = ""
        error = str(e)
    duration_s = time.monotonic() - start

    usage = budget.usage()
    success = _last_run_tests_success(result.transcript) if error is None else False

    return {
        "duration_s": round(duration_s, 1),
        "llm_calls": usage["llm_calls"],
        "tool_calls": usage["tool_calls"],
        "final_content": final_content,
        "success": success,
        "error": error,
    }
