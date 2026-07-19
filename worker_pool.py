"""Redis-backed worker-pool queue: reliable claim/ack pattern (BRPOPLPUSH) plus a
generic worker-loop runner. This file knows nothing about tasks/agents — it just
moves opaque string ids between pending/processing/dead-letter lists so it stays
reusable across whatever the orchestrator decides "a unit of work" is.
"""

import threading
import time
from typing import Callable

import redis

QUEUE_PENDING = "queue:pending"
QUEUE_PROCESSING = "queue:processing"
QUEUE_DEAD_LETTER = "queue:dead_letter"


class WorkerPool:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    def enqueue(self, item_id: str) -> None:
        self.redis.lpush(QUEUE_PENDING, item_id)

    def claim(self, timeout: float = 2.0) -> str | None:
        """Blocking pop from pending -> processing. Returns None on timeout (no work available)."""
        result = self.redis.brpoplpush(QUEUE_PENDING, QUEUE_PROCESSING, timeout=timeout)
        return result

    def ack(self, item_id: str) -> None:
        """Work succeeded (or was handed off elsewhere) — remove from the in-flight list."""
        self.redis.lrem(QUEUE_PROCESSING, 1, item_id)

    def dead_letter(self, item_id: str) -> None:
        self.redis.lrem(QUEUE_PROCESSING, 1, item_id)
        self.redis.lpush(QUEUE_DEAD_LETTER, item_id)

    def requeue_orphaned(self) -> int:
        """Move everything still sitting in `processing` back to `pending`.

        Call this at startup to recover from a crash that happened mid-claim (worker died
        after BRPOPLPUSH but before ack/dead_letter).
        """
        moved = 0
        while True:
            item = self.redis.rpoplpush(QUEUE_PROCESSING, QUEUE_PENDING)
            if item is None:
                break
            moved += 1
        return moved

    def depth(self) -> dict:
        return {
            "pending": self.redis.llen(QUEUE_PENDING),
            "processing": self.redis.llen(QUEUE_PROCESSING),
            "dead_letter": self.redis.llen(QUEUE_DEAD_LETTER),
        }


def run_worker_loop(
    pool: WorkerPool,
    executor_fn: Callable[[str], None],
    stop_event: threading.Event,
    poll_timeout: float = 2.0,
    on_error: Callable[[str, Exception], None] | None = None,
) -> None:
    """Claim item ids and hand each to executor_fn until stop_event is set.

    executor_fn is responsible for the full lifecycle of an item (including calling
    pool.ack / pool.dead_letter itself) so it can make idempotency and retry decisions.
    """
    while not stop_event.is_set():
        item_id = pool.claim(timeout=poll_timeout)
        if item_id is None:
            continue
        try:
            executor_fn(item_id)
        except Exception as e:  # noqa: BLE001 - a worker thread must never die silently
            if on_error:
                on_error(item_id, e)
            else:
                pool.dead_letter(item_id)


def start_workers(
    pool: WorkerPool,
    executor_fn: Callable[[str], None],
    num_workers: int,
    on_error: Callable[[str, Exception], None] | None = None,
) -> tuple[list[threading.Thread], threading.Event]:
    stop_event = threading.Event()
    threads = []
    for i in range(num_workers):
        t = threading.Thread(
            target=run_worker_loop,
            args=(pool, executor_fn, stop_event),
            kwargs={"on_error": on_error},
            name=f"worker-{i}",
            daemon=True,
        )
        t.start()
        threads.append(t)
    return threads, stop_event


def stop_workers(threads: list[threading.Thread], stop_event: threading.Event, timeout: float = 5.0) -> None:
    stop_event.set()
    for t in threads:
        t.join(timeout=timeout)
