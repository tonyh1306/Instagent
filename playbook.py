"""Redis-backed playbook: a capped, per-role list of first-attempt successes, fed back
to that role as few-shot context on future tasks.

This is a prompt-level stand-in for a fine-tuning/distillation loop: there's no
training infra here (agents call a hosted Qwen Cloud endpoint, not a model we can
re-weight), so "the society teaches its next generation" has to happen in-context
instead of in-weights. Only first-try successes (TaskNode.attempts == 0 at commit
time) are distilled in - a task that needed repair or arbiter intervention taught the
worker something, but the trajectory itself isn't a clean example to imitate.

Capped at MAX_ENTRIES per role (oldest evicted first) so the playbook can't grow
without bound or let one dominant task shape crowd out everything else - a crude but
auditable guard against the "sudden drift" failure mode of an unbounded experience
store.
"""

import json

import redis

from schemas import AgentRole, TaskNode

MAX_ENTRIES = 5
SUMMARY_CHARS = 400
EXEMPLARS_PER_CONTEXT = 2


class Playbook:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    @staticmethod
    def _key(role: AgentRole) -> str:
        return f"playbook:{role.value}"

    def record_success(self, role: AgentRole, task: TaskNode, artifact_content: str) -> None:
        entry = {
            "description": task.description,
            "acceptance_criteria": task.acceptance_criteria,
            "summary": (artifact_content or "")[:SUMMARY_CHARS],
        }
        key = self._key(role)
        self.redis.lpush(key, json.dumps(entry))
        self.redis.ltrim(key, 0, MAX_ENTRIES - 1)

    def count(self, role: AgentRole) -> int:
        return self.redis.llen(self._key(role))

    def exemplars(self, role: AgentRole, limit: int = EXEMPLARS_PER_CONTEXT) -> list[dict]:
        raw = self.redis.lrange(self._key(role), 0, limit - 1)
        return [json.loads(r) for r in raw]

    def as_context(self, role: AgentRole, limit: int = EXEMPLARS_PER_CONTEXT) -> str:
        exemplars = self.exemplars(role, limit)
        if not exemplars:
            return ""
        lines = ["Past first-attempt successes on similar tasks, for reference:"]
        for e in exemplars:
            lines.append(f"- Task: {e['description']!r}\n  What worked: {e['summary']}")
        return "\n".join(lines)
