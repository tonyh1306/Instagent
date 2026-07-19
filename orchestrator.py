"""Decomposes a user task into a DAG of TaskNodes, writes it to the blackboard, and
dispatches dependency-satisfied ("ready") nodes to the Redis worker pool for
parallel execution by the right specialist agent.
"""

import time
import uuid

import bidding
import council
import diversity
from agents.base_agent import BaseAgent
from agents.coder import CoderAgent
from agents.critic import CriticAgent
from agents.researcher import ResearcherAgent
from agents.writer import WriterAgent
from blackboard import Blackboard, VersionConflict
from budget import BudgetExceeded, RunBudget
from circuit_breaker import CircuitBreaker
from playbook import Playbook
from reputation import Reputation
from schemas import AgentRole, ConflictEvent, CriticVerdict, TaskNode, TaskStatus, utcnow
from worker_pool import WorkerPool, start_workers, stop_workers

MAX_REPAIR_ATTEMPTS = 1

SUBMIT_PLAN_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_plan",
        "description": "Submit the decomposed task DAG for the user's request.",
        "parameters": {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "short unique id, e.g. 't1'"},
                            "description": {"type": "string"},
                            "assigned_agent": {
                                "type": "string",
                                "enum": ["researcher", "coder", "writer"],
                            },
                            "dependencies": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "ids of other tasks in this same plan that must commit first",
                            },
                            "acceptance_criteria": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "concrete, checkable criteria",
                            },
                            "candidate_agents": {
                                "type": "array",
                                "items": {"type": "string", "enum": ["researcher", "coder", "writer"]},
                                "description": (
                                    "Only set this if 2+ roles could plausibly do this subtask well "
                                    "(e.g. 'summarize findings' fits researcher or writer). Leave empty "
                                    "for subtasks with one obvious owner - contested tasks trigger a "
                                    "bidding round, which costs extra calls, so don't overuse this."
                                ),
                            },
                        },
                        "required": ["id", "description", "assigned_agent", "acceptance_criteria"],
                    },
                }
            },
            "required": ["tasks"],
        },
    },
}

TERMINAL_STATUSES = {TaskStatus.COMMITTED, TaskStatus.DEAD_LETTER}


class Orchestrator(BaseAgent):
    ROLE = AgentRole.ORCHESTRATOR
    MODEL = "qwen-max"
    TOOLS = [SUBMIT_PLAN_TOOL]
    TERMINAL_TOOLS = {"submit_plan"}
    FORCE_TOOL = "submit_plan"

    def system_prompt(self) -> str:
        return (
            "You are the orchestrator agent. Decompose the user's task into a small DAG of "
            "subtasks (prefer 2-5). Each subtask must be assigned to exactly one of: "
            "researcher, coder, writer (set assigned_agent to your best default). Use "
            "'dependencies' to reference other task ids in this same plan that must commit "
            "first. Every subtask needs concrete, checkable acceptance_criteria.\n\n"
            "Actively look for at least one subtask whose deliverable is a written artifact "
            "with no strict dependency on code (a summary, explanation, or report) - both "
            "researcher and writer can genuinely produce that, so list both in "
            "candidate_agents for it and let them bid. Don't do this for subtasks that clearly "
            "need code execution (coder) or that clearly need fresh information gathering "
            "before anything can be written (researcher alone) - only flag real overlap."
        )

    def task_prompt(self, task: TaskNode, context: str) -> str:
        return f"User task to decompose:\n\n{task.description}"

    def decompose(self, user_task: str) -> list[TaskNode]:
        wrapper = TaskNode(id="__decompose__", description=user_task, assigned_agent=self.ROLE)
        result = self.run(wrapper)
        raw_tasks = result.structured["tasks"]
        ids = {t["id"] for t in raw_tasks}
        nodes = []
        for t in raw_tasks:
            deps = [d for d in t.get("dependencies", []) if d in ids and d != t["id"]]
            candidates = [AgentRole(r) for r in t.get("candidate_agents", []) if r]
            # Contract-net bidding overwrites assigned_agent before dispatch anyway, so if the
            # model left it blank while listing candidates, just seed it with the first candidate.
            assigned_raw = t.get("assigned_agent") or (candidates[0].value if candidates else None)
            nodes.append(
                TaskNode(
                    id=t["id"],
                    description=t["description"],
                    assigned_agent=AgentRole(assigned_raw),
                    dependencies=deps,
                    acceptance_criteria=t.get("acceptance_criteria", []),
                    candidate_agents=candidates if len(candidates) >= 2 else [],
                )
            )
        return nodes


def _load_all_task_nodes(bb: Blackboard) -> dict[str, TaskNode]:
    entries = bb.list("task:")
    return {e.value["id"]: TaskNode.model_validate(e.value) for e in entries}


def _is_ready(node: TaskNode, all_nodes: dict[str, TaskNode]) -> bool:
    return node.status in (TaskStatus.PENDING, TaskStatus.REPAIR) and all(
        all_nodes[d].status == TaskStatus.COMMITTED for d in node.dependencies if d in all_nodes
    )


def _gather_context(
    bb: Blackboard, node: TaskNode, all_nodes: dict[str, TaskNode], playbook: Playbook | None = None
) -> str:
    parts = []
    for dep_id in node.dependencies:
        dep = all_nodes.get(dep_id)
        if dep and dep.artifact_ref:
            art = bb.get(dep.artifact_ref)
            if art:
                parts.append(f"[from {dep_id}] {art.value.get('content', '')}")
    if playbook is not None:
        playbook_context = playbook.as_context(node.assigned_agent)
        if playbook_context:
            parts.append(playbook_context)
    if node.status == TaskStatus.REPAIR:
        verdict_entry = bb.get(f"verdict:{node.id}")
        if verdict_entry:
            reasons = verdict_entry.value.get("reasons", [])
            reason_lines = "\n".join(f"- {r}" for r in reasons)
            parts.append(
                "Your previous attempt was rejected by the critic for these reasons:\n"
                f"{reason_lines}\nAddress them in this attempt."
            )
    return "\n\n".join(parts)


def _log_conflict(bb: Blackboard, key: str, attempted_by: AgentRole, expected_version: int | None) -> ConflictEvent:
    """Logs a structured ConflictEvent when a versioned write loses its race.

    Our queue mechanics guarantee only one worker ever claims a given pending task_id, so the
    only conflicts that actually occur here are mechanical (two writers touching the same key
    at the same version, e.g. a crash-recovery re-claim racing a still-live claim). There's no
    real content to merge in that case - "reread_and_merge" concretely means "reread current
    state and back off," which is what the caller does after logging this. A genuine semantic
    conflict (two workers producing contradictory artifacts for the same dependency) can't occur
    in this architecture since a dependency's artifact is immutable once committed - so
    resolution="escalated_to_council" is defined in the schema but intentionally unused here
    rather than faked.
    """
    current = bb.get(key)
    event = ConflictEvent(
        key=key,
        attempted_by=attempted_by,
        expected_version=expected_version,
        actual_version=current.version if current else None,
        resolution="reread_and_merge",
    )
    bb.log({"event": "conflict", **event.model_dump(mode="json")})
    return event


def _run_critic(critic_agent: CriticAgent, node: TaskNode, artifact_content: str) -> CriticVerdict:
    critic_task = TaskNode(
        id=node.id,
        description=f"Review this output for task: {node.description}",
        assigned_agent=AgentRole.CRITIC,
        acceptance_criteria=node.acceptance_criteria,
    )
    context = (
        f"Artifact produced by {node.assigned_agent.value}:\n\n{artifact_content}\n\n"
        "Relevant files (if any) are already present in the shared workspace; "
        "use run_tests to verify if applicable."
    )
    result = critic_agent.run(critic_task, context)
    return CriticVerdict.model_validate(result.structured)


def _safe_set_task(
    bb: Blackboard,
    pool: WorkerPool,
    task_id: str,
    node: TaskNode,
    expected_version: int,
    written_by: AgentRole = AgentRole.ORCHESTRATOR,
) -> bool:
    """Writes the task node; on VersionConflict, logs a ConflictEvent and dead-letters as a
    safe fallback (something changed underneath us in a place we assumed we owned exclusively -
    we don't know what, so we can't safely keep negotiating from stale state)."""
    try:
        bb.set(f"task:{node.id}", node.model_dump(mode="json"), written_by=written_by, expected_version=expected_version)
        return True
    except VersionConflict:
        _log_conflict(bb, f"task:{node.id}", written_by, expected_version)
        pool.dead_letter(task_id)
        return False


def make_executor(
    bb: Blackboard,
    pool: WorkerPool,
    agent_registry: dict[AgentRole, BaseAgent],
    budget: RunBudget | None = None,
    reputation: Reputation | None = None,
    playbook: Playbook | None = None,
):
    """Returns the per-task-id function the worker pool calls for each claimed item.

    Every worker artifact passes through the critic before being committed. On a fail
    verdict, the task gets one repair attempt (routed back to the same worker with the
    critic's reasons as context); a second fail escalates instead of retrying forever.
    """

    def execute(task_id: str) -> None:
        entry = bb.get(f"task:{task_id}")
        if entry is None:
            pool.ack(task_id)
            return
        node = TaskNode.model_validate(entry.value)
        # PENDING/REPAIR is the normal case; IN_PROGRESS is only re-claimable because
        # pool.requeue_orphaned() moves crashed-worker items back to pending at startup -
        # if it's here, no live worker still owns it. Anything else really is already handled.
        if node.status not in (TaskStatus.PENDING, TaskStatus.REPAIR, TaskStatus.IN_PROGRESS):
            pool.ack(task_id)  # already handled by someone else - idempotency guard
            return

        if node.status == TaskStatus.PENDING and len(node.candidate_agents) >= 2:
            model_by_role = {r: agent_registry[r].MODEL for r in node.candidate_agents}
            winner = bidding.run_contract_net(
                node, node.candidate_agents, model_by_role, bb, budget=budget, reputation=reputation
            )
            node.assigned_agent = winner
            node.updated_at = utcnow()
            current = bb.set(
                f"task:{node.id}", node.model_dump(mode="json"), written_by=AgentRole.ORCHESTRATOR,
                expected_version=entry.version,
            )
            entry = current

        all_nodes = _load_all_task_nodes(bb)
        context = _gather_context(bb, node, all_nodes, playbook=playbook)

        node.status = TaskStatus.IN_PROGRESS
        node.updated_at = utcnow()
        try:
            current = bb.set(
                f"task:{node.id}", node.model_dump(mode="json"), written_by=AgentRole.ORCHESTRATOR,
                expected_version=entry.version,
            )
        except VersionConflict:
            _log_conflict(bb, f"task:{node.id}", AgentRole.ORCHESTRATOR, entry.version)
            pool.ack(task_id)  # lost a race with another claim of this same version - back off
            return

        agent = agent_registry[node.assigned_agent]
        variant_idx = None
        system_prompt_suffix = ""
        if reputation is not None and diversity.has_variants(node.assigned_agent):
            variant_idx = diversity.select_variant(node.assigned_agent, reputation)
            system_prompt_suffix = diversity.variant_suffix(node.assigned_agent, variant_idx)
            bb.log(
                {
                    "event": "variant_selected",
                    "task_id": node.id,
                    "role": node.assigned_agent.value,
                    "variant_idx": variant_idx,
                }
            )
        try:
            result = agent.run(node, context, system_prompt_suffix=system_prompt_suffix)

            artifact_key = f"artifact:{node.id}"
            existing_artifact = bb.get(artifact_key)
            bb.set(
                artifact_key, {"content": result.content}, written_by=node.assigned_agent,
                expected_version=existing_artifact.version if existing_artifact else None,
            )

            current = bb.get(f"task:{node.id}")
            node = TaskNode.model_validate(current.value)
            node.artifact_ref = artifact_key

            verdict = _run_critic(agent_registry[AgentRole.CRITIC], node, result.content)
        except Exception as e:
            bb.log({"event": "worker_error", "task_id": node.id, "error": str(e)})
            node.status = TaskStatus.DEAD_LETTER
            node.updated_at = utcnow()
            if _safe_set_task(bb, pool, task_id, node, current.version):
                pool.dead_letter(task_id)
            return

        bb.log(
            {"event": "critic_verdict", "task_id": node.id, "passed": verdict.passed, "reasons": verdict.reasons}
        )
        if reputation is not None:
            reputation.record_outcome(node.assigned_agent, verdict.passed)
            bb.log(
                {
                    "event": "reputation_update",
                    "subject": node.assigned_agent.value,
                    **reputation.stats(node.assigned_agent),
                }
            )
            if variant_idx is not None:
                variant_subject = diversity.variant_subject(node.assigned_agent, variant_idx)
                reputation.record_outcome(variant_subject, verdict.passed)
                bb.log(
                    {
                        "event": "reputation_update",
                        "subject": variant_subject,
                        **reputation.stats(variant_subject),
                    }
                )

        if verdict.passed:
            node.status = TaskStatus.COMMITTED
            node.updated_at = utcnow()
            if not _safe_set_task(bb, pool, task_id, node, current.version):
                return
            bb.log({"event": "committed", "task_id": node.id})
            if playbook is not None and node.attempts == 0:
                # Only distill in clean first-attempt successes - repaired/arbitrated
                # trajectories taught the worker something, but aren't a clean pattern to imitate.
                playbook.record_success(node.assigned_agent, node, result.content)
                bb.log({"event": "playbook_recorded", "task_id": node.id, "role": node.assigned_agent.value})
            pool.ack(task_id)

            all_nodes = _load_all_task_nodes(bb)
            for n in all_nodes.values():
                if _is_ready(n, all_nodes):
                    pool.enqueue(n.id)
            return

        existing_verdict = bb.get(f"verdict:{node.id}")
        bb.set(
            f"verdict:{node.id}", verdict.model_dump(), written_by=AgentRole.CRITIC,
            expected_version=existing_verdict.version if existing_verdict else None,
        )
        node.attempts += 1
        node.updated_at = utcnow()

        if node.arbiter_ruled:
            # This was the one bounded repair attempt the arbiter granted as a compromise.
            # Failing it again means no further negotiation - straight to dead-letter.
            node.status = TaskStatus.DEAD_LETTER
            if not _safe_set_task(bb, pool, task_id, node, current.version):
                return
            bb.log({"event": "dead_letter", "task_id": node.id, "reason": "arbiter-mandated repair also failed"})
            pool.dead_letter(task_id)
        elif node.attempts <= MAX_REPAIR_ATTEMPTS:
            node.status = TaskStatus.REPAIR
            if not _safe_set_task(bb, pool, task_id, node, current.version):
                return
            bb.log({"event": "repair", "task_id": node.id, "attempt": node.attempts})
            pool.ack(task_id)
            pool.enqueue(node.id)
        else:
            node.status = TaskStatus.ESCALATED
            if not _safe_set_task(bb, pool, task_id, node, current.version):
                return
            bb.log({"event": "escalated", "task_id": node.id})
            pool.ack(task_id)

    return execute


def _resolve_escalations(
    bb: Blackboard,
    pool: WorkerPool,
    agent_registry: dict[AgentRole, BaseAgent],
    budget: RunBudget | None = None,
    reputation: Reputation | None = None,
) -> None:
    """Runs the bounded council + arbiter for any ESCALATED node and applies the ruling.

    Council is a governance step, not a retryable unit of work, so it runs synchronously
    in the orchestrator's poll loop rather than through the worker pool queue.
    """
    any_committed = False
    all_nodes = _load_all_task_nodes(bb)
    for node in all_nodes.values():
        if node.status != TaskStatus.ESCALATED:
            continue

        artifact_entry = bb.get(node.artifact_ref) if node.artifact_ref else None
        artifact_content = artifact_entry.value.get("content", "") if artifact_entry else ""
        verdict_entry = bb.get(f"verdict:{node.id}")
        critic_reasons = verdict_entry.value.get("reasons", []) if verdict_entry else []
        worker_model = agent_registry[node.assigned_agent].MODEL

        try:
            outcome = council.resolve(node, artifact_content, critic_reasons, worker_model, budget=budget, bb=bb)
            decision = outcome["decision"]
            bb.log({"event": "council_transcript", "task_id": node.id, "transcript": outcome["transcript"]})
            bb.log(
                {
                    "event": "arbiter_decision",
                    "task_id": node.id,
                    "decision": decision.decision,
                    "rationale": decision.rationale,
                    "repair_instructions": decision.repair_instructions,
                }
            )
            new_status = {
                "accept": TaskStatus.COMMITTED,
                "reject": TaskStatus.DEAD_LETTER,
                "accept_with_repair": TaskStatus.REPAIR,
            }[decision.decision]
            # accept_with_repair doesn't get recorded here - its outcome is still pending
            # the follow-up repair attempt, which will record its own critic verdict.
            if reputation is not None and decision.decision in ("accept", "reject"):
                reputation.record_outcome(node.assigned_agent, decision.decision == "accept")
        except Exception as e:
            bb.log({"event": "council_error", "task_id": node.id, "error": str(e)})
            decision = None
            new_status = TaskStatus.DEAD_LETTER

        current = bb.get(f"task:{node.id}")
        fresh_node = TaskNode.model_validate(current.value)
        fresh_node.status = new_status
        fresh_node.updated_at = utcnow()

        if new_status == TaskStatus.REPAIR:
            fresh_node.arbiter_ruled = True
            existing_verdict = bb.get(f"verdict:{node.id}")
            bb.set(
                f"verdict:{node.id}",
                {"passed": False, "reasons": [f"[arbiter compromise ruling] {decision.repair_instructions}"]},
                written_by=AgentRole.ARBITER,
                expected_version=existing_verdict.version if existing_verdict else None,
            )

        if not _safe_set_task(bb, pool, node.id, fresh_node, current.version, written_by=AgentRole.ARBITER):
            continue
        bb.log({"event": new_status.value, "task_id": node.id})

        if new_status == TaskStatus.REPAIR:
            pool.enqueue(node.id)
        any_committed = any_committed or new_status == TaskStatus.COMMITTED

    if any_committed:
        fresh_nodes = _load_all_task_nodes(bb)
        for n in fresh_nodes.values():
            if _is_ready(n, fresh_nodes):
                pool.enqueue(n.id)


def run_pipeline(
    user_task: str,
    num_workers: int = 3,
    max_wall_clock_s: float = 180.0,
    poll_interval_s: float = 1.0,
    max_llm_calls: int = 60,
    max_tool_calls: int = 150,
    circuit_failure_threshold: int = 3,
    circuit_cooldown_s: float = 60.0,
) -> dict[str, TaskNode]:
    bb = Blackboard()
    pool = WorkerPool(bb.redis)
    pool.requeue_orphaned()

    run_id = uuid.uuid4().hex[:12]
    budget = RunBudget(bb.redis, run_id, max_llm_calls=max_llm_calls, max_tool_calls=max_tool_calls)
    circuit_breaker = CircuitBreaker(
        bb.redis, failure_threshold=circuit_failure_threshold, cooldown_s=circuit_cooldown_s
    )
    # Unlike budget/circuit_breaker, reputation is deliberately NOT run_id-scoped - it's the
    # persistent, cross-run trust signal (see reputation.py), so it lives under its own
    # `reputation:*` Redis prefix that main.py/dashboard.py's reset never touches.
    reputation = Reputation(bb.redis)
    # Also not run_id-scoped, for the same reason as reputation - the playbook is meant to
    # accumulate across runs (see playbook.py).
    playbook = Playbook(bb.redis)
    bb.log({"event": "run_started", "run_id": run_id, "max_llm_calls": max_llm_calls, "max_tool_calls": max_tool_calls})

    orchestrator = Orchestrator(circuit_breaker=circuit_breaker, budget=budget)
    nodes = orchestrator.decompose(user_task)
    for node in nodes:
        bb.set(f"task:{node.id}", node.model_dump(mode="json"), written_by=AgentRole.ORCHESTRATOR)
    bb.log({"event": "decompose", "task_ids": [n.id for n in nodes]})

    all_nodes = _load_all_task_nodes(bb)
    for n in all_nodes.values():
        if _is_ready(n, all_nodes):
            pool.enqueue(n.id)

    agent_registry = {
        AgentRole.RESEARCHER: ResearcherAgent(circuit_breaker=circuit_breaker, budget=budget),
        AgentRole.CODER: CoderAgent(circuit_breaker=circuit_breaker, budget=budget),
        AgentRole.WRITER: WriterAgent(circuit_breaker=circuit_breaker, budget=budget),
        AgentRole.CRITIC: CriticAgent(circuit_breaker=circuit_breaker, budget=budget),
    }
    executor = make_executor(bb, pool, agent_registry, budget=budget, reputation=reputation, playbook=playbook)
    threads, stop_event = start_workers(pool, executor, num_workers)

    deadline = time.monotonic() + max_wall_clock_s
    while time.monotonic() < deadline:
        _resolve_escalations(bb, pool, agent_registry, budget=budget, reputation=reputation)
        all_nodes = _load_all_task_nodes(bb)
        if all(n.status in TERMINAL_STATUSES for n in all_nodes.values()):
            break
        if budget.exhausted():
            bb.log({"event": "budget_exhausted", "run_id": run_id, "usage": budget.usage()})
            break
        time.sleep(poll_interval_s)

    stop_workers(threads, stop_event)
    return _load_all_task_nodes(bb)
