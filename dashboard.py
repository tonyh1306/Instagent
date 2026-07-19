"""Live demo dashboard: runs the single-agent baseline and the multi-agent pipeline on the
same task, then renders a side-by-side comparison - wall-clock time, LLM/tool call counts,
negotiation overhead (% of calls spent on bidding + council, not core work), and the full
decision-log timeline so the conflict + resolution flow is visible, not just asserted.
"""

import html
import json
import time

import redis as redis_lib
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from baseline import run_baseline
from blackboard import Blackboard
from budget import RunBudget
from orchestrator import run_pipeline

DEFAULT_TASK = (
    "Write a Python function that checks if a number is prime, with a pytest test, "
    "and a short README documenting it."
)
NEGOTIATION_EVENTS = {"bid", "council_turn", "arbiter_decision"}

app = FastAPI()


def _clear_run_state(bb: Blackboard) -> None:
    # reputation:* and playbook:* are intentionally left alone here - see main.py's
    # reset_state for why they're cross-run state rather than per-comparison scratch state.
    for pattern in ("bb:*", "budget:*", "circuit:*"):
        for key in bb.redis.scan_iter(match=pattern):
            bb.redis.delete(key)
    bb.redis.delete("queue:pending", "queue:processing", "queue:dead_letter")


def run_comparison(task: str) -> dict:
    bb = Blackboard()

    _clear_run_state(bb)
    t0 = time.monotonic()
    baseline_result = run_baseline(task)
    baseline_duration = round(time.monotonic() - t0, 1)

    _clear_run_state(bb)
    t0 = time.monotonic()
    final_nodes = run_pipeline(task, num_workers=3, max_wall_clock_s=280)
    multi_duration = round(time.monotonic() - t0, 1)

    log = bb.get_log()
    run_started = next((e for e in log if e["event"] == "run_started"), None)
    usage = {"llm_calls": 0, "tool_calls": 0}
    if run_started:
        usage = RunBudget(bb.redis, run_started["run_id"]).usage()

    negotiation_calls = sum(1 for e in log if e["event"] in NEGOTIATION_EVENTS)
    total_calls = usage["llm_calls"] or 1
    overhead_pct = round(100 * negotiation_calls / total_calls, 1)

    committed = sum(1 for n in final_nodes.values() if n.status.value == "committed")
    dead_lettered = sum(1 for n in final_nodes.values() if n.status.value == "dead_letter")

    comparison = {
        "task": task,
        "baseline": {**baseline_result, "duration_s": baseline_duration},
        "multi_agent": {
            "duration_s": multi_duration,
            "llm_calls": usage["llm_calls"],
            "tool_calls": usage["tool_calls"],
            "negotiation_calls": negotiation_calls,
            "negotiation_overhead_pct": overhead_pct,
            "tasks_committed": committed,
            "tasks_dead_lettered": dead_lettered,
            "success": dead_lettered == 0 and committed > 0,
        },
        "decision_log": log,
    }
    bb.redis.set("comparison:latest", json.dumps(comparison, default=str))
    return comparison


def _fmt_bool(b) -> str:
    if b is None:
        return "<span class=\"unknown\">unknown</span>"
    return "<span class=\"pass\">yes</span>" if b else "<span class=\"fail\">no</span>"


def _render_html(comparison: dict) -> str:
    b = comparison["baseline"]
    m = comparison["multi_agent"]
    speedup = round(b["duration_s"] / m["duration_s"], 2) if m["duration_s"] else 0

    log_rows = "\n".join(
        f'<tr><td class="ts">{html.escape(e.get("timestamp", ""))}</td>'
        f'<td class="ev">{html.escape(e.get("event", ""))}</td>'
        f'<td class="detail">{html.escape(json.dumps({k: v for k, v in e.items() if k not in ("timestamp", "event")}))[:200]}</td></tr>'
        for e in comparison["decision_log"]
    )

    return f"""
<html>
<head>
<title>Multi-agent vs single-agent: {html.escape(comparison['task'][:60])}</title>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 1100px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }}
  h1 {{ font-size: 1.3rem; }}
  .task {{ color: #555; margin-bottom: 1.5rem; }}
  .columns {{ display: flex; gap: 1.5rem; margin-bottom: 2rem; }}
  .col {{ flex: 1; border: 1px solid #ddd; border-radius: 8px; padding: 1rem 1.25rem; }}
  .col h2 {{ margin-top: 0; font-size: 1.05rem; }}
  .metric {{ display: flex; justify-content: space-between; padding: 0.3rem 0; border-bottom: 1px solid #f0f0f0; }}
  .metric .label {{ color: #666; }}
  .metric .value {{ font-weight: 600; }}
  .headline {{ background: #f5f5f7; border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem; font-size: 1.05rem; }}
  .pass {{ color: #1a7f37; font-weight: 600; }}
  .fail {{ color: #cf222e; font-weight: 600; }}
  .unknown {{ color: #9a6700; font-weight: 600; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
  th {{ text-align: left; padding: 0.4rem; border-bottom: 2px solid #ddd; }}
  td {{ padding: 0.35rem 0.4rem; border-bottom: 1px solid #f0f0f0; vertical-align: top; }}
  .ts {{ color: #999; white-space: nowrap; }}
  .ev {{ font-weight: 600; white-space: nowrap; }}
  .detail {{ color: #444; font-family: monospace; font-size: 0.78rem; }}
</style>
</head>
<body>
<h1>Single-agent vs multi-agent</h1>
<div class="task">Task: {html.escape(comparison['task'])}</div>

<div class="headline">
  Multi-agent finished in <strong>{m['duration_s']}s</strong> vs baseline's <strong>{b['duration_s']}s</strong>
  ({speedup}x). Negotiation (bidding + council) accounted for
  <strong>{m['negotiation_overhead_pct']}%</strong> of its {m['llm_calls']} LLM calls.
</div>

<div class="columns">
  <div class="col">
    <h2>Baseline (single agent)</h2>
    <div class="metric"><span class="label">Wall-clock time</span><span class="value">{b['duration_s']}s</span></div>
    <div class="metric"><span class="label">LLM calls</span><span class="value">{b['llm_calls']}</span></div>
    <div class="metric"><span class="label">Tool calls</span><span class="value">{b['tool_calls']}</span></div>
    <div class="metric"><span class="label">Self-reported success</span><span class="value">{_fmt_bool(b['success'])}</span></div>
    <div class="metric"><span class="label">Error</span><span class="value">{html.escape(str(b.get('error') or '-'))}</span></div>
  </div>
  <div class="col">
    <h2>Multi-agent pipeline</h2>
    <div class="metric"><span class="label">Wall-clock time</span><span class="value">{m['duration_s']}s</span></div>
    <div class="metric"><span class="label">LLM calls</span><span class="value">{m['llm_calls']}</span></div>
    <div class="metric"><span class="label">Tool calls</span><span class="value">{m['tool_calls']}</span></div>
    <div class="metric"><span class="label">Negotiation calls (bid/council)</span><span class="value">{m['negotiation_calls']} ({m['negotiation_overhead_pct']}%)</span></div>
    <div class="metric"><span class="label">Tasks committed / dead-lettered</span><span class="value">{m['tasks_committed']} / {m['tasks_dead_lettered']}</span></div>
    <div class="metric"><span class="label">Success (no dead-letters)</span><span class="value">{_fmt_bool(m['success'])}</span></div>
  </div>
</div>

<h2>Multi-agent decision log</h2>
<table>
<tr><th>Time</th><th>Event</th><th>Detail</th></tr>
{log_rows}
</table>
</body>
</html>
"""


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(task: str = DEFAULT_TASK, rerun: bool = True):
    bb = Blackboard()
    if rerun:
        comparison = run_comparison(task)
    else:
        raw = bb.redis.get("comparison:latest")
        comparison = json.loads(raw) if raw else run_comparison(task)
    return _render_html(comparison)
