# Architecture

Multi-agent task-execution system on Qwen Cloud (DashScope OpenAI-compatible endpoint).
Hub-and-spoke: an orchestrator decomposes work, specialist agents execute in parallel via a
Redis-backed worker pool, a critic gates every output, and a bounded council + arbiter resolve
what the critic can't. Agents never share raw conversation state — they only read/write typed
pydantic models through a shared Redis blackboard.

## Wire formats (`schemas.py`)

Every cross-agent handoff is a pydantic model with `extra="forbid"` — no agent ever passes a
raw string to another; if it doesn't fit a schema, it doesn't cross a boundary.

- **`AgentRole`** — `orchestrator | researcher | coder | writer | critic | arbiter | baseline`
- **`TaskStatus`** — `pending → (repair) → committed | dead_letter`, with `in_progress`,
  `escalated` as transient states. See the state machine below.
- **`TaskNode`** — a DAG node: `id`, `description`, `assigned_agent`, `dependencies`,
  `acceptance_criteria`, `status`, `attempts`, `artifact_ref`, `candidate_agents` (non-empty
  triggers bidding), `arbiter_ruled` (bounds re-escalation after a compromise ruling).
- **`BlackboardEntry`** — `key`, `value`, `written_by`, `version` (optimistic concurrency).
- **`CriticVerdict`** — `passed: bool`, `reasons: list[str]`.
- **`ArbiterDecision`** — `decision: accept | reject | accept_with_repair`, `rationale`,
  `repair_instructions` (required for `accept_with_repair`).
- **`Bid`** — `agent_role`, `confidence`, `approach`, `estimated_steps` (contract-net bidding).
- **`CouncilTurn`** — `round`, `speaker`, `turn_type: claim|evidence|concession|rebuttal`, `text`.
- **`ConflictEvent`** — `key`, `attempted_by`, `expected_version`, `actual_version`,
  `resolution: reread_and_merge | escalated_to_council`.

## Blackboard (`blackboard.py`)

Thin Redis wrapper: `get(key)`, `set(key, value, written_by, expected_version=None)`,
`list(prefix)`, plus an append-only `log(event)` / `get_log()` decision log.

`set` is optimistic-concurrency-controlled via a single atomic Lua script (check-version +
write in one round trip): pass `expected_version=None` to create a key that must not already
exist, or the version you last read to update it. A stale or conflicting write raises
`VersionConflict` rather than silently clobbering the other writer. Every call site in
`orchestrator.py` that could plausibly race catches this, logs a `ConflictEvent`, and backs off
(`_safe_set_task` / `_log_conflict`) — see "Fault tolerance" below.

## Agents (`agents/`)

`base_agent.py` implements one shared tool-calling loop used by every role:

```
call Qwen → parse tool_calls → execute each (or return, if terminal) → feed results back → repeat
```

- **Per-agent tool scoping**: each subclass declares its own `TOOLS` (schemas) and
  `register_tools()` (name → callable). This is enforced, not just organized — e.g. `CriticAgent`
  has zero mutation tools (`file_read`, `schema_validate`, `run_tests` only), so it is
  structurally incapable of "fixing" code instead of reporting on it.
- **Terminal tools**: a tool name in `TERMINAL_TOOLS` ends the loop immediately and returns its
  parsed arguments as structured output instead of executing anything. This is how
  `Orchestrator.decompose`, `CriticAgent`'s `submit_verdict`, and `ArbiterAgent`'s
  `submit_decision` get schema-locked JSON out of a tool-calling model with no prose parsing.
- **`FORCE_TOOL`**: when an agent has nothing to inspect and must always emit structured output
  (Orchestrator, Arbiter), `tool_choice` is forced to that one tool from the first turn.
- Hard `MAX_ITERATIONS` (loop) and `CALL_TIMEOUT_S`/`TOOL_TIMEOUT_S` (per-call, via a
  single-worker `ThreadPoolExecutor`) bound every agent.
- Roles: `ResearcherAgent` (web_search, doc_fetch), `CoderAgent` (file_write, run_tests, lint),
  `WriterAgent` (file_write, doc_format), `CriticAgent` (file_read, schema_validate, run_tests,
  + forced `submit_verdict`), `ArbiterAgent` (only the forced `submit_decision`).
- `agents/tools.py` holds every concrete tool implementation. All file I/O is sandboxed to
  `workspace/` (`_resolve_in_workspace` rejects path traversal and normalizes a redundant
  leading `workspace/` segment agents are prone to echoing).

## Task lifecycle (`orchestrator.py`)

```
PENDING --[contract-net bid, if candidate_agents]--> PENDING (assigned_agent resolved)
PENDING --[dispatch]--> IN_PROGRESS --[worker output + critic pass]--> COMMITTED
                                    \-[critic fail, attempts==0]--> REPAIR --> IN_PROGRESS
                                    \-[critic fail, attempts>0]--> ESCALATED
ESCALATED --[council + arbiter: accept]--> COMMITTED
ESCALATED --[council + arbiter: reject]--> DEAD_LETTER
ESCALATED --[council + arbiter: accept_with_repair]--> REPAIR (arbiter_ruled=True) --> IN_PROGRESS
REPAIR (arbiter_ruled=True) --[fails again]--> DEAD_LETTER   # no re-escalation; one compromise, bounded
```

1. **Decompose**: `Orchestrator` (qwen-max, `FORCE_TOOL="submit_plan"`) turns the user request
   into 2-5 `TaskNode`s with dependencies and acceptance criteria. It's told to flag genuine
   role overlap (e.g. "write a summary" could be researcher or writer) via `candidate_agents`;
   everything else gets a direct `assigned_agent` guess with no bidding overhead.
2. **Dispatch**: nodes with all dependencies `COMMITTED` are pushed onto the Redis worker pool
   (`worker_pool.py`: `BRPOPLPUSH pending→processing`, `ack`/`dead_letter` to resolve). N worker
   threads claim and run `make_executor`'s `execute(task_id)`.
3. **Contract-net bidding** (`bidding.py`): if a node has 2+ `candidate_agents`, each bids in
   parallel (one forced-tool call each) with confidence/approach/estimated_steps; the
   orchestrator scores (`confidence - 0.02*estimated_steps`) and awards, logging the full
   call-for-proposals → bids → award to the decision log. Skipped entirely for unambiguous
   tasks — this is the expensive path, used only where roles genuinely overlap.
4. **Critic gate**: every worker artifact is reviewed by `CriticAgent` before being committed.
   The critic can `file_read`/`run_tests` the actual workspace files (not just trust the
   worker's prose summary — an earlier bug) and returns a typed `CriticVerdict`. Its prompt is
   explicitly scoped to *only* the task's `acceptance_criteria` — it does not invent extra
   requirements.
5. **Repair**: one repair attempt is allowed, with the critic's `reasons` injected into the
   worker's next context (`_gather_context`). A second failure escalates.
6. **Council** (`council.py`): bounded, cross-visible negotiation. The worker and critic each
   see the other's most recent turn and must respond to its *specific* points via typed
   `CouncilTurn`s (`claim`/`evidence`/`concession`/`rebuttal`) — not just restate their opening
   position. Hard-capped at `MAX_ROUNDS=2`.
7. **Arbiter**: single qwen-max call reading the full transcript, rules once —
   `accept` / `reject` / `accept_with_repair`. The last is a genuine synthesis path: the arbiter
   grants one more bounded repair attempt with specific `repair_instructions`
   (`TaskNode.arbiter_ruled=True` ensures a second failure goes straight to `DEAD_LETTER`
   instead of re-escalating — no infinite negotiation).
8. `_resolve_escalations` runs this synchronously in the orchestrator's poll loop (governance,
   not a retryable queue item) each iteration between wall-clock timeout checks.

## Fault tolerance

- **Retry-with-backoff** (`qwen_client.py`): transient errors (connection, timeout, rate-limit,
  5xx) retried with exponential backoff, `MAX_RETRIES=3`.
- **Circuit breaker** (`circuit_breaker.py`): Redis-backed, opens per-`AgentRole` after
  `failure_threshold` consecutive failures, cools down after `cooldown_s`. Checked/updated
  around every `BaseAgent._call_qwen`.
- **Run budget** (`budget.py`): Redis-backed hard ceiling on total LLM calls and tool calls per
  run (`run_id`-scoped), charged from `BaseAgent` and from council's raw calls. The
  orchestrator's poll loop stops the run if exhausted.
- **Idempotency / crash recovery**: `WorkerPool.requeue_orphaned()` moves anything still in
  `processing` back to `pending` at startup. `execute()`'s guard treats `PENDING`/`REPAIR`/
  `IN_PROGRESS` as re-claimable (an `IN_PROGRESS` node reaching the queue again only happens via
  orphan-recovery, since the queue itself prevents double-claiming a live task) and anything
  else as already-handled — verified for both the "orphan gets redone" and "finished task stays
  skipped" cases.
- **Dead-letter queue**: `queue:dead_letter` in Redis + `TaskStatus.DEAD_LETTER`, reached via
  repair exhaustion, escalation rejection, a failed post-compromise repair, or any uncaught
  worker/critic exception.
- **ConflictEvent**: every versioned blackboard write in the executor that could race
  (`_safe_set_task`) catches `VersionConflict`, logs a `ConflictEvent` (honestly labeled
  `resolution="reread_and_merge"` — there's nothing to merge, so this means "reread and back
  off"), and dead-letters rather than proceeding from stale state. `resolution=
  "escalated_to_council"` exists in the schema but is intentionally unused: this architecture
  makes true semantic dependency conflicts structurally impossible (a dependency's artifact is
  immutable once `COMMITTED`), so nothing fabricates that path.

## Reputation, diversity, and the playbook

Three small mechanisms, layered on top of the task lifecycle, that let the system's behavior
improve across runs without any agent carrying private state:

- **Reputation** (`reputation.py`) — a Redis-backed trust score per subject (an `AgentRole`, or
  a composite `"role:vN"` string — see diversity below), built from committed-vs-rejected
  outcomes: every critic verdict and every arbiter `accept`/`reject` ruling records an outcome
  (`accept_with_repair` doesn't record immediately — its outcome is still pending the follow-up
  repair attempt's own critic verdict). Score is a Beta(1,1)-smoothed success rate
  (`(successes+1)/(attempts+2)`), so a subject with no history starts neutral at 0.5 rather than
  being penalized or favored by default, and one early result can't swing it to 0 or 1.
  Unlike `budget.py`/`circuit_breaker.py`, reputation is **not** `run_id`-scoped — it's meant to
  persist and accumulate across runs, so it lives under its own `reputation:*` Redis prefix that
  the default reset in `main.py`/`dashboard.py` never touches (clear it explicitly with
  `--reset-learned-state`). Contract-net bidding (`bidding.py`) folds a subject's reputation
  into its award score as a gentle ±0.1 nudge centered on 0.5 — confidence still dominates, a
  proven track record just breaks close calls.
- **Diversity preservation** (`diversity.py`) — each worker role's `agent_registry` entry is a
  single instance reused for the whole run, so without this every dispatch of a role would run
  the exact same system prompt: a population of one. Each worker role (researcher/coder/writer)
  has 2 short stance-variant prompt suffixes (still bound by the same acceptance criteria — these
  change approach, not requirements). Before dispatch, `execute()` picks a variant epsilon-greedily
  (`EPSILON=0.2`) using that role's per-variant reputation (`"role:vN"` subjects): mostly exploit
  whichever stance has been passing the critic more, but keep sampling the other one so it isn't
  permanently starved out by an early bad run. `BaseAgent.run` takes an optional
  `system_prompt_suffix` to carry the chosen variant's text; both the variant choice and its
  resulting reputation update are logged (`variant_selected`, `reputation_update`).
- **Playbook** (`playbook.py`) — a capped (`MAX_ENTRIES=5`), per-role Redis list of first-attempt
  successes (`TaskNode.attempts == 0` at commit — a task that needed repair or arbitration taught
  the worker something, but isn't a clean trajectory to imitate), fed back to that role as
  few-shot context (`_gather_context`) on future tasks. This is the prompt-level stand-in for a
  fine-tuning/distillation loop: there's no training infra here (agents call a hosted Qwen Cloud
  endpoint, not a model this system can re-weight), so "the society teaches its next generation"
  happens in-context instead of in-weights. Also not `run_id`-scoped, for the same reason as
  reputation.
  - **Why not a persistent per-agent memory graph instead of these three:** roles here are
    interchangeable job titles, not individuals — `agent_registry` builds a fresh instance per
    role per run, and the critic's statelessness (see above) is load-bearing, not incidental. A
    full episodic memory store would let behavior drift on private history nothing else in the
    system can see or audit, cutting against the "every handoff is a typed, inspectable model"
    discipline everywhere else in this codebase. Reputation already gives the useful slice of
    "persistent identity" (a durable, auditable trust score, visible in the decision log); the
    playbook gives the useful slice of "learns over time" (a capped, auditable exemplar list).
    Revisit a real memory graph only if a role needs to recall something about a specific
    external entity across runs (e.g. a particular user's conventions) — a materially different
    problem than generic task-shape learning, and nothing here currently demonstrates that need.

## Model tiering

| Role | Model | Rationale |
|---|---|---|
| Orchestrator, Arbiter | `qwen-max` | low call volume, high-stakes single decisions |
| Researcher, Coder, Writer, Baseline | `qwen-plus` | high volume, parallelizable |
| Critic | `qwen-plus` | needs to catch subtle errors; stateless per call |
| Council turns | worker's own model / `qwen-plus` for critic's turns | matches the role speaking |

## Baseline + dashboard (`baseline.py`, `dashboard.py`)

`BaselineAgent` is a single `qwen-plus` agent with the union of every worker tool, no
decomposition, no critic, no parallelism — a genuine comparison point, not a strawman.
`dashboard.py` (FastAPI, `GET /dashboard`) runs the baseline and the full pipeline back-to-back
on the same task and renders wall-clock time, LLM/tool call counts, negotiation overhead % (
`bid`/`council_turn`/`arbiter_decision` events as a fraction of total LLM calls, derived
directly from the decision log — no separate tracking needed), and the full decision-log
timeline.

**Honest empirical finding, not papered over**: on both a sequential (code→test→docs) and a
breadth-heavy (3 independent paragraphs) demo task, the multi-agent pipeline was *slower* than
the single-agent baseline (~0.44-0.53x), even with real parallelism and bidding firing
correctly. Per-subtask critic-review overhead outweighed the parallelism gain at this task
scale. The multi-agent run did catch a real bug (a filename mismatch) the ungated baseline
couldn't be checked for. If reporting an "efficiency gain," the defensible claim from actual
runs is reliability/quality at a latency cost, not a wall-clock win — that would need either
much larger/more parallel tasks or cheaper critic passes to reverse.

## Local dev

Redis must be running (`brew services start redis`). All run state (blackboard, queues, budget
counters, circuit-breaker state) lives in Redis and persists across runs — see README.md for
the reset snippet. `workspace/` is the sandboxed root all agent file I/O is confined to.
