"""Redis-backed reputation: a trust score per subject (an AgentRole, or a composite
"role:variant" string — see diversity.py) that accumulates across runs rather than
resetting per-run like budget.py or per-cooldown-window like circuit_breaker.py.

Built from committed-vs-rejected outcomes (critic verdicts, arbiter accept/reject
rulings). Unlike the circuit breaker, which only cares about a recent failure streak,
reputation is the long-run signal: it's what lets bidding (bidding.py) and variant
selection (diversity.py) prefer subjects with a track record, without needing any
agent to remember anything itself.
"""

import redis

from schemas import AgentRole


class Reputation:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    @staticmethod
    def _key(subject: AgentRole | str) -> str:
        s = subject.value if isinstance(subject, AgentRole) else subject
        return f"reputation:{s}"

    def record_outcome(self, subject: AgentRole | str, success: bool) -> None:
        key = self._key(subject)
        self.redis.hincrby(key, "attempts", 1)
        if success:
            self.redis.hincrby(key, "successes", 1)

    def score(self, subject: AgentRole | str) -> float:
        """Laplace-smoothed success rate (Beta(1,1) prior): 0.5 for a subject with no
        history yet, converging toward the true rate as attempts accumulate. Smoothing
        keeps one early failure/success from swinging the score to 0.0/1.0."""
        state = self.redis.hgetall(self._key(subject))
        successes = int(state.get("successes", 0))
        attempts = int(state.get("attempts", 0))
        return (successes + 1) / (attempts + 2)

    def stats(self, subject: AgentRole | str) -> dict:
        state = self.redis.hgetall(self._key(subject))
        return {
            "successes": int(state.get("successes", 0)),
            "attempts": int(state.get("attempts", 0)),
            "score": self.score(subject),
        }
