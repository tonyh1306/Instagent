"""Shared Qwen tool-calling loop: call -> parse tool_calls -> execute -> feed back -> repeat.

Each subclass declares its own TOOLS list (only the tools that agent is allowed to
call) and registers executor functions for them. A subclass may also mark some of
its tools as "terminal": calling one of those doesn't execute anything, it ends the
loop immediately and returns the tool's parsed arguments as structured output. This
is how the orchestrator/critic/arbiter get schema-locked JSON out of a tool-calling
model without ad-hoc prose parsing.
"""

import concurrent.futures
import json
from dataclasses import dataclass
from typing import Callable

from budget import RunBudget
from circuit_breaker import CircuitBreaker
from qwen_client import call
from schemas import AgentRole, TaskNode


class MaxIterationsExceeded(Exception):
    pass


class AgentCallTimeout(Exception):
    pass


@dataclass
class AgentRunResult:
    content: str | None  # free-text final answer, if the loop ended without a terminal tool
    structured: dict | None  # parsed arguments of a terminal tool call, if one ended the loop
    terminal_tool: str | None  # name of the terminal tool that was called, if any
    transcript: list[dict]  # full message history, for logging to the blackboard decision log


class BaseAgent:
    ROLE: AgentRole
    MODEL: str = "qwen-plus"
    TOOLS: list[dict] = []
    TERMINAL_TOOLS: set[str] = set()
    FORCE_TOOL: str | None = None  # if set, every call forces this tool (for agents with no inspection tools)
    MAX_ITERATIONS: int = 6
    CALL_TIMEOUT_S: float = 60.0
    TOOL_TIMEOUT_S: float = 30.0

    def __init__(self, circuit_breaker: CircuitBreaker | None = None, budget: RunBudget | None = None):
        self._executors: dict[str, Callable[..., object]] = {}
        self.circuit_breaker = circuit_breaker
        self.budget = budget
        self.register_tools()

    def register_tools(self) -> None:
        """Subclasses populate self._executors with {tool_name: callable} here."""

    def system_prompt(self) -> str:
        return (
            f"You are the {self.ROLE.value} agent in a multi-agent system. "
            "Use only the tools you've been given. Stay within your role."
        )

    def task_prompt(self, task: TaskNode, context: str) -> str:
        criteria = "\n".join(f"- {c}" for c in task.acceptance_criteria) or "(none specified)"
        return (
            f"Task: {task.description}\n\n"
            f"Acceptance criteria:\n{criteria}\n\n"
            f"Context:\n{context or '(none)'}"
        )

    def _call_qwen(self, messages: list[dict]):
        if self.circuit_breaker:
            self.circuit_breaker.check(self.ROLE)  # raises CircuitOpen if tripped
        if self.budget:
            self.budget.charge_llm_call()  # raises BudgetExceeded if run is over budget

        tool_choice = "auto"
        if self.FORCE_TOOL:
            tool_choice = {"type": "function", "function": {"name": self.FORCE_TOOL}}

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(
                    call,
                    model=self.MODEL,
                    messages=messages,
                    tools=self.TOOLS or None,
                    tool_choice=tool_choice,
                )
                try:
                    response = future.result(timeout=self.CALL_TIMEOUT_S)
                except concurrent.futures.TimeoutError as e:
                    raise AgentCallTimeout(
                        f"{self.ROLE.value}: Qwen call exceeded {self.CALL_TIMEOUT_S}s"
                    ) from e
        except Exception:
            if self.circuit_breaker:
                self.circuit_breaker.record_failure(self.ROLE)
            raise

        if self.circuit_breaker:
            self.circuit_breaker.record_success(self.ROLE)
        return response

    def _execute_tool(self, name: str, arguments_json: str) -> str:
        if self.budget:
            self.budget.charge_tool_call()  # raises BudgetExceeded if run is over budget
        fn = self._executors.get(name)
        if fn is None:
            return json.dumps({"error": f"tool {name!r} is not available to {self.ROLE.value}"})
        try:
            args = json.loads(arguments_json) if arguments_json else {}
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"invalid tool arguments JSON: {e}"})
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(lambda: fn(**args))
            try:
                result = future.result(timeout=self.TOOL_TIMEOUT_S)
            except concurrent.futures.TimeoutError:
                return json.dumps({"error": f"tool {name!r} exceeded {self.TOOL_TIMEOUT_S}s timeout"})
            except Exception as e:  # noqa: BLE001 - tool failures are reported back to the model, not raised
                return json.dumps({"error": f"tool {name!r} raised: {e}"})
        return result if isinstance(result, str) else json.dumps(result)

    def run(self, task: TaskNode, context: str = "", system_prompt_suffix: str = "") -> AgentRunResult:
        system_content = self.system_prompt()
        if system_prompt_suffix:
            system_content = f"{system_content}\n\n{system_prompt_suffix}"
        messages: list[dict] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": self.task_prompt(task, context)},
        ]

        for _ in range(self.MAX_ITERATIONS):
            response = self._call_qwen(messages)
            msg = response.choices[0].message

            if not msg.tool_calls:
                messages.append({"role": "assistant", "content": msg.content})
                return AgentRunResult(
                    content=msg.content, structured=None, terminal_tool=None, transcript=messages
                )

            for tc in msg.tool_calls:
                if tc.function.name in self.TERMINAL_TOOLS:
                    structured = json.loads(tc.function.arguments)
                    messages.append(
                        {
                            "role": "assistant",
                            "content": msg.content,
                            "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
                        }
                    )
                    return AgentRunResult(
                        content=None,
                        structured=structured,
                        terminal_tool=tc.function.name,
                        transcript=messages,
                    )

            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
                }
            )
            for tc in msg.tool_calls:
                result = self._execute_tool(tc.function.name, tc.function.arguments)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        raise MaxIterationsExceeded(
            f"{self.ROLE.value} exceeded {self.MAX_ITERATIONS} iterations on task {task.id}"
        )
