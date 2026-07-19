"""Redis-backed shared state: task graph, artifacts, and the decision log.

Optimistic versioning: to update a key, callers must pass the version they last
read via `expected_version`. If another writer already bumped the version, `set`
raises `VersionConflict` instead of silently clobbering the other write. The
check-and-set happens inside a single Lua script, so it's atomic even with many
concurrent workers.
"""

from __future__ import annotations

import json
import os

import redis

from schemas import AgentRole, BlackboardEntry, utcnow

_KEY_PREFIX = "bb:"
_LOG_KEY = "bb:log"

_CAS_SCRIPT = """
local current = redis.call('GET', KEYS[1])
local expected = ARGV[1]
if expected == '' then
    if current then
        return 0
    end
else
    if not current then
        return 0
    end
    local current_tbl = cjson.decode(current)
    if tostring(current_tbl['version']) ~= expected then
        return 0
    end
end
redis.call('SET', KEYS[1], ARGV[2])
return 1
"""


class VersionConflict(Exception):
    """Raised when `set` is called with a stale `expected_version`."""


class Blackboard:
    def __init__(self, redis_client: redis.Redis | None = None, url: str | None = None):
        self.redis = redis_client or redis.from_url(
            url or os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
            decode_responses=True,
        )
        self._cas = self.redis.register_script(_CAS_SCRIPT)

    @staticmethod
    def _rk(key: str) -> str:
        return f"{_KEY_PREFIX}{key}"

    def get(self, key: str) -> BlackboardEntry | None:
        raw = self.redis.get(self._rk(key))
        if raw is None:
            return None
        return BlackboardEntry.model_validate_json(raw)

    def set(
        self,
        key: str,
        value: object,
        written_by: AgentRole,
        expected_version: int | None = None,
    ) -> BlackboardEntry:
        """Create (expected_version=None) or update (expected_version=<last-read version>) a key.

        Raises VersionConflict if the key's current version doesn't match expected_version
        (or, for creation, if the key already exists).
        """
        new_version = 1 if expected_version is None else expected_version + 1
        expected_arg = "" if expected_version is None else str(expected_version)

        entry = BlackboardEntry(
            key=key,
            value=value,
            written_by=written_by,
            timestamp=utcnow(),
            version=new_version,
        )

        ok = self._cas(keys=[self._rk(key)], args=[expected_arg, entry.model_dump_json()])
        if not ok:
            raise VersionConflict(
                f"key {key!r}: expected_version={expected_version} is stale or key state mismatched"
            )
        return entry

    def list(self, prefix: str) -> list[BlackboardEntry]:
        pattern = f"{self._rk(prefix)}*"
        keys = list(self.redis.scan_iter(match=pattern))
        if not keys:
            return []
        raws = self.redis.mget(keys)
        return [BlackboardEntry.model_validate_json(r) for r in raws if r is not None]

    def log(self, event: dict) -> None:
        """Append an event to the append-only decision log (task/council/arbiter decisions)."""
        record = {"timestamp": utcnow().isoformat(), **event}
        self.redis.rpush(_LOG_KEY, json.dumps(record))

    def get_log(self) -> list[dict]:
        return [json.loads(r) for r in self.redis.lrange(_LOG_KEY, 0, -1)]
