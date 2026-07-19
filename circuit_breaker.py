"""Redis-backed circuit breaker: opens per-agent-role after N consecutive failures,
so a persistently broken agent (bad prompt, dead tool, API outage) stops burning
calls/time instead of getting retried forever by every worker thread.
"""

import time

import redis

from schemas import AgentRole


class CircuitOpen(Exception):
    pass


class CircuitBreaker:
    def __init__(
        self,
        redis_client: redis.Redis,
        failure_threshold: int = 3,
        cooldown_s: float = 60.0,
    ):
        self.redis = redis_client
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s

    @staticmethod
    def _key(role: AgentRole) -> str:
        return f"circuit:{role.value}"

    def check(self, role: AgentRole) -> None:
        """Raises CircuitOpen if the breaker for this role is currently open."""
        state = self.redis.hgetall(self._key(role))
        if not state:
            return
        opened_until = float(state.get("opened_until", 0))
        if time.time() < opened_until:
            raise CircuitOpen(
                f"circuit open for {role.value}: {int(opened_until - time.time())}s remaining"
            )

    def record_success(self, role: AgentRole) -> None:
        self.redis.delete(self._key(role))

    def record_failure(self, role: AgentRole) -> None:
        key = self._key(role)
        failures = self.redis.hincrby(key, "failures", 1)
        if failures >= self.failure_threshold:
            self.redis.hset(key, mapping={"opened_until": time.time() + self.cooldown_s})
