"""Live control panel: submit any task and watch the pipeline run against real Redis
state - task DAG with per-node status, streaming decision log, reputation/diversity/
playbook stats - instead of only the fixed baseline-vs-pipeline comparison in
dashboard.py.

Runs one pipeline at a time, matching the rest of the system's single-run-at-a-time
Redis state (task keys aren't run_id-namespaced): a submission spawns run_pipeline in
a background thread, and the frontend polls /api/status on a 1s timer rather than
needing a websocket - every value it needs (task nodes, the decision log, reputation,
playbook, budget, queue depth) is already just a Redis read away, live, regardless of
which thread is driving the run.
"""

import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from blackboard import Blackboard
from diversity import VARIANTS, variant_subject
from main import reset_state
from orchestrator import run_pipeline
from playbook import Playbook
from reputation import Reputation
from schemas import AgentRole, TaskNode
from worker_pool import WorkerPool
from budget import RunBudget

app = FastAPI()
_bb = Blackboard()
_lock = threading.Lock()

ROLE_ORDER = [
    AgentRole.ORCHESTRATOR,
    AgentRole.RESEARCHER,
    AgentRole.CODER,
    AgentRole.WRITER,
    AgentRole.CRITIC,
    AgentRole.ARBITER,
]

_state = {
    "run_id": None,
    "task": None,
    "running": False,
    "error": None,
    "started_at": None,
    "finished_at": None,
}


def _run_in_background(task: str, workers: int, timeout: float, max_llm_calls: int, max_tool_calls: int) -> None:
    try:
        run_pipeline(
            task,
            num_workers=workers,
            max_wall_clock_s=timeout,
            max_llm_calls=max_llm_calls,
            max_tool_calls=max_tool_calls,
        )
    except Exception as e:  # noqa: BLE001 - surface any pipeline-level failure to the UI
        with _lock:
            _state["error"] = str(e)
    finally:
        with _lock:
            _state["running"] = False
            _state["finished_at"] = time.time()


@app.post("/api/run")
async def start_run(request: Request):
    payload = await request.json()
    task = (payload.get("task") or "").strip()
    if not task:
        raise HTTPException(400, "task is required")

    with _lock:
        if _state["running"]:
            raise HTTPException(409, "a run is already in progress")
        # Only bb:*/budget:*/circuit:*/queues - reputation and the playbook are
        # deliberately left alone, since they're meant to accumulate across runs.
        reset_state(_bb)
        run_id = uuid.uuid4().hex[:8]
        _state.update(
            run_id=run_id, task=task, running=True, error=None, started_at=time.time(), finished_at=None
        )

    workers = int(payload.get("workers", 3))
    timeout = float(payload.get("timeout", 200))
    max_llm_calls = int(payload.get("max_llm_calls", 60))
    max_tool_calls = int(payload.get("max_tool_calls", 150))
    thread = threading.Thread(
        target=_run_in_background,
        args=(task, workers, timeout, max_llm_calls, max_tool_calls),
        daemon=True,
    )
    thread.start()
    return {"run_id": run_id, "status": "started"}


def _reputation_snapshot() -> dict:
    rep = Reputation(_bb.redis)
    roles = {role.value: rep.stats(role) for role in ROLE_ORDER}
    variants = {
        role.value: [
            {"variant": i, "suffix": texts[i], **rep.stats(variant_subject(role, i))}
            for i in range(len(texts))
        ]
        for role, texts in VARIANTS.items()
    }
    return {"roles": roles, "variants": variants}


def _playbook_snapshot() -> dict:
    pb = Playbook(_bb.redis)
    return {role.value: pb.count(role) for role in VARIANTS}


def _budget_snapshot(log: list[dict]) -> dict:
    run_started = next((e for e in log if e["event"] == "run_started"), None)
    if not run_started:
        return {"llm_calls": 0, "tool_calls": 0, "max_llm_calls": 0, "max_tool_calls": 0}
    usage = RunBudget(_bb.redis, run_started["run_id"]).usage()
    return {**usage, "max_llm_calls": run_started["max_llm_calls"], "max_tool_calls": run_started["max_tool_calls"]}


@app.get("/api/status")
def status(log_since: int = 0):
    entries = _bb.list("task:")
    nodes = sorted((TaskNode.model_validate(e.value) for e in entries), key=lambda n: n.id)
    log = _bb.get_log()
    with _lock:
        run_info = dict(_state)
    return JSONResponse(
        {
            "run": run_info,
            "nodes": [n.model_dump(mode="json") for n in nodes],
            "log": log[log_since:],
            "log_total": len(log),
            "reputation": _reputation_snapshot(),
            "playbook": _playbook_snapshot(),
            "queue_depth": WorkerPool(_bb.redis).depth(),
            "budget": _budget_snapshot(log),
        }
    )


@app.get("/", response_class=HTMLResponse)
def index():
    return _PAGE


_PAGE = (Path(__file__).parent / "control_panel.html").read_text()
