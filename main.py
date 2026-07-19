"""CLI entry point for the multi-agent pipeline.

    uv run main.py "Write a prime checker with a pytest test and a README."

By default each run starts from a clean slate (blackboard, queues, budget counters,
circuit-breaker state in Redis are cleared first); pass --keep-state to resume/inspect
prior state instead.
"""

import argparse
import sys

from blackboard import Blackboard
from orchestrator import run_pipeline
from schemas import TaskStatus

_REDIS_PATTERNS = ("bb:*", "budget:*", "circuit:*")
_QUEUES = ("queue:pending", "queue:processing", "queue:dead_letter")
# reputation:* and playbook:* are deliberately excluded from the default reset - they're the
# system's cross-run "society memory" (trust scores, distilled successful trajectories), not
# per-run scratch state. Clear them explicitly with --reset-learned-state.
_LEARNED_STATE_PATTERNS = ("reputation:*", "playbook:*")


def reset_state(bb: Blackboard, include_learned_state: bool = False) -> None:
    patterns = _REDIS_PATTERNS + (_LEARNED_STATE_PATTERNS if include_learned_state else ())
    for pattern in patterns:
        for key in bb.redis.scan_iter(match=pattern):
            bb.redis.delete(key)
    bb.redis.delete(*_QUEUES)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the multi-agent pipeline on a task.",
        epilog='Example: uv run main.py "Write a prime checker with a pytest test."',
    )
    parser.add_argument("task", nargs="?", help="the task to run")
    parser.add_argument("--workers", type=int, default=3, help="worker threads (default: 3)")
    parser.add_argument(
        "--timeout", type=float, default=200.0, help="max wall-clock seconds (default: 200)"
    )
    parser.add_argument(
        "--max-llm-calls", type=int, default=60, help="run budget: LLM calls (default: 60)"
    )
    parser.add_argument(
        "--max-tool-calls", type=int, default=150, help="run budget: tool calls (default: 150)"
    )
    parser.add_argument(
        "--keep-state",
        action="store_true",
        help="don't clear prior run state from Redis before starting",
    )
    parser.add_argument(
        "--reset-only", action="store_true", help="clear run state from Redis and exit"
    )
    parser.add_argument(
        "--reset-learned-state",
        action="store_true",
        help="also clear reputation scores and the playbook (persistent across runs by default)",
    )
    parser.add_argument(
        "--show-log", action="store_true", help="print the full decision log after the run"
    )
    args = parser.parse_args()

    bb = Blackboard()

    if args.reset_only:
        reset_state(bb, include_learned_state=args.reset_learned_state)
        print("Run state cleared.")
        return 0

    if not args.task:
        parser.error("a task is required (or use --reset-only)")

    if not args.keep_state:
        reset_state(bb, include_learned_state=args.reset_learned_state)

    final_nodes = run_pipeline(
        args.task,
        num_workers=args.workers,
        max_wall_clock_s=args.timeout,
        max_llm_calls=args.max_llm_calls,
        max_tool_calls=args.max_tool_calls,
    )

    print()
    for node_id, node in sorted(final_nodes.items()):
        artifact = f" | artifact: {node.artifact_ref}" if node.artifact_ref else ""
        print(f"{node_id}: {node.status.value}{artifact}")

    if args.show_log:
        print("\n--- decision log ---")
        for event in bb.get_log():
            print(event)

    failed = [n for n in final_nodes.values() if n.status != TaskStatus.COMMITTED]
    if failed:
        print(f"\n{len(failed)}/{len(final_nodes)} task(s) did not commit.")
        return 1
    print(f"\nAll {len(final_nodes)} task(s) committed. Artifacts are in workspace/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
