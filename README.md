# qwen-agents

A multi-agent task-execution system built on Qwen Cloud (DashScope's OpenAI-compatible
endpoint). An orchestrator decomposes a user request into a dependency DAG, dispatches it to
a pool of specialist agents (researcher/coder/writer) running in parallel, gates every output
through a critic before it's accepted, and escalates unresolved disagreements to a bounded
council + arbiter. Contested role assignments go through a contract-net bidding round instead
of being dictated. Includes a single-agent baseline and a live dashboard for comparing the two.

## Requirements

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/)
- Redis running locally (`brew install redis`, then `brew services start redis` or
  `redis-server`)
- A DashScope (Qwen Cloud) API key

## Setup

```bash
uv sync
cp .env.example .env
# edit .env and set DASHSCOPE_API_KEY (and DASHSCOPE_REGION=intl or cn)
```

Verify Redis is up:

```bash
redis-cli ping   # -> PONG
```

Verify the Qwen client + tool-calling works:

```bash
uv run smoke_test_tools.py
```

## Running the multi-agent pipeline

```bash
uv run main.py "Write a Python function that checks if a number is prime, with a pytest test, and a short README documenting it."
```

Prior run state in Redis is cleared automatically before each run (pass `--keep-state` to
skip that). Other knobs — `--workers`, `--timeout`, `--max-llm-calls`, `--max-tool-calls`,
`--show-log` (print the decision log after the run) — are described in
`uv run main.py --help`.

Output artifacts (files written by the coder/writer agents) land in `workspace/`. The full
run trace (decompose, bids, critic verdicts, repairs, council turns, arbiter decisions) is in
the blackboard's decision log — pass `--show-log` to print it after the run, or read it
programmatically:

```python
from blackboard import Blackboard
for event in Blackboard().get_log():
    print(event)
```

### Resetting state between runs

The blackboard, work queues, budget counters, and circuit-breaker state all live in Redis;
`main.py` clears them automatically before each run. To clear them manually (e.g. after a
dashboard run):

```bash
uv run main.py --reset-only
rm -f workspace/*.py workspace/*.md   # if you also want to drop prior artifacts
```

Reputation scores and the playbook (`reputation:*`, `playbook:*` — see CLAUDE.md) are
deliberately **not** cleared by the above, since they're meant to accumulate across runs, not
reset per-run like the rest. Clear them explicitly if you want a fresh start:

```bash
uv run main.py --reset-only --reset-learned-state
```

## Running the comparison dashboard

Runs a single-agent baseline and the full multi-agent pipeline back-to-back on the same task
and renders a side-by-side comparison (wall-clock time, LLM/tool call counts, negotiation
overhead, and the full decision-log timeline).

```bash
uv run uvicorn dashboard:app --port 8756
```

Then open `http://127.0.0.1:8756/dashboard` (add `?task=...` to use a different prompt than
the default). Each load triggers a fresh comparison run and takes a few minutes end to end,
since it runs both systems live against the API.

## Running the live control panel

Submit any task and watch the multi-agent pipeline run against live Redis state: the task
DAG with per-node status (pending/in-progress/repair/escalated/committed/dead-letter), the
decision log streaming in as it's written, and live reputation/diversity/playbook/budget
panels.

```bash
uv run uvicorn control_panel:app --port 8760
```

Open `http://127.0.0.1:8760/`, type a task, and hit Run. The page polls `/api/status` once a
second - no websocket required, since every value it shows is already just a Redis read via
the same blackboard/reputation/playbook/budget/worker-pool objects the pipeline itself uses.
It runs one pipeline at a time (a second submission while one is in flight gets rejected),
matching the rest of the system's single-run-at-a-time Redis state.

## Cross-run learning: reputation, diversity, playbook

Three mechanisms make the system's behavior improve across runs, not just within one:

- **Reputation** — every critic verdict and arbiter accept/reject ruling updates a trust score
  per role (and, see below, per prompt variant). Contract-net bidding uses it as a small nudge
  when awarding contested tasks.
- **Diversity** — each worker role (researcher/coder/writer) has two prompt "stances" (e.g.
  thorough-vs-simple); which one gets used for a given task is picked epsilon-greedily from
  their reputations, so the system keeps trying both instead of collapsing onto one style.
- **Playbook** — a role's first-attempt successes (no repair needed) get distilled into a
  capped list of exemplars, fed back to that role as few-shot context on future tasks.

All three persist across runs by design (see "Resetting state" above for how to clear them).
Watch them working via the decision log:

```python
from blackboard import Blackboard
for event in Blackboard().get_log():
    if event["event"] in ("reputation_update", "variant_selected", "playbook_recorded"):
        print(event)
```

See CLAUDE.md for the full rationale, including why this stops short of giving agents a
persistent memory graph.

## Project layout

| File | Purpose |
|---|---|
| `main.py` | CLI entry point: reset state, run the pipeline, print results |
| `qwen_client.py` | Thin OpenAI-SDK wrapper pointed at DashScope, with retry-with-backoff |
| `schemas.py` | All pydantic wire-format types (see CLAUDE.md) |
| `blackboard.py` | Redis-backed shared state: versioned key/value store + decision log |
| `worker_pool.py` | Redis-backed reliable work queue (claim/ack/dead-letter) |
| `orchestrator.py` | Decompose → DAG → dispatch → critic gate → repair → escalate |
| `bidding.py` | Contract-net protocol for contested role assignments |
| `council.py` | Bounded worker↔critic negotiation before arbitration |
| `budget.py` / `circuit_breaker.py` | Global run budget and per-role circuit breaker |
| `reputation.py` | Cross-run trust score per role (and per prompt-variant) |
| `diversity.py` | Epsilon-greedy prompt-variant selection per worker role |
| `playbook.py` | Cross-run few-shot exemplars distilled from first-attempt successes |
| `baseline.py` | Single-agent comparison runner |
| `dashboard.py` | FastAPI live comparison dashboard |
| `control_panel.py` / `control_panel.html` | FastAPI live control panel: submit a task, watch it run |
| `agents/` | `base_agent.py` (tool-calling loop) + one file per role |

See `CLAUDE.md` for the full architecture and design rationale.
