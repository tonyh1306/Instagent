"""Global run budget: a hard ceiling on LLM calls and tool calls for one pipeline run,
shared across worker threads via Redis so it can't be blown past by concurrency.
"""

import redis


class BudgetExceeded(Exception):
    pass


class RunBudget:
    def __init__(
        self,
        redis_client: redis.Redis,
        run_id: str,
        max_llm_calls: int = 100,
        max_tool_calls: int = 300,
    ):
        self.redis = redis_client
        self.run_id = run_id
        self.max_llm_calls = max_llm_calls
        self.max_tool_calls = max_tool_calls

    def _llm_key(self) -> str:
        return f"budget:{self.run_id}:llm_calls"

    def _tool_key(self) -> str:
        return f"budget:{self.run_id}:tool_calls"

    def charge_llm_call(self) -> None:
        total = self.redis.incr(self._llm_key())
        if total > self.max_llm_calls:
            raise BudgetExceeded(f"run {self.run_id} exceeded max_llm_calls={self.max_llm_calls}")

    def charge_tool_call(self) -> None:
        total = self.redis.incr(self._tool_key())
        if total > self.max_tool_calls:
            raise BudgetExceeded(f"run {self.run_id} exceeded max_tool_calls={self.max_tool_calls}")

    def usage(self) -> dict:
        return {
            "llm_calls": int(self.redis.get(self._llm_key()) or 0),
            "tool_calls": int(self.redis.get(self._tool_key()) or 0),
        }

    def exhausted(self) -> bool:
        u = self.usage()
        return u["llm_calls"] >= self.max_llm_calls or u["tool_calls"] >= self.max_tool_calls
