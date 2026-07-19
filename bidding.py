"""Contract-net protocol: when a subtask could plausibly go to more than one agent role
(orchestrator flags this via TaskNode.candidate_agents), broadcast a call-for-proposals,
collect structured Bids from each candidate in parallel, and award the task to the best
bid. Logged to the blackboard decision log so the negotiation is visible, not implicit.

Skipped entirely for unambiguous subtasks (candidate_agents empty/single) - bidding only
earns its keep where roles genuinely overlap.
"""

import concurrent.futures
import json

from blackboard import Blackboard
from budget import RunBudget
from qwen_client import call
from reputation import Reputation
from schemas import AgentRole, Bid, TaskNode

SUBMIT_BID_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_bid",
        "description": "Submit your bid for this subtask.",
        "parameters": {
            "type": "object",
            "properties": {
                "confidence": {"type": "number", "description": "0.0-1.0: how confident you are you can do this well"},
                "approach": {"type": "string", "description": "one or two sentences: how you'd tackle it"},
                "estimated_steps": {"type": "integer", "description": "rough number of tool calls you'd expect to need"},
            },
            "required": ["confidence", "approach", "estimated_steps"],
        },
    },
}


def _collect_bid(role: AgentRole, model: str, task: TaskNode, budget: RunBudget | None) -> Bid:
    if budget:
        budget.charge_llm_call()
    messages = [
        {
            "role": "system",
            "content": (
                f"You are the {role.value} agent. A subtask is open for bidding among agents "
                "who could plausibly handle it. Submit a bid via submit_bid: your confidence "
                "(0-1), a one-to-two sentence approach, and an estimated number of tool-call "
                "steps. Be honest — overclaiming confidence you can't back up hurts the team."
            ),
        },
        {
            "role": "user",
            "content": f"Subtask: {task.description}\nAcceptance criteria: {task.acceptance_criteria}",
        },
    ]
    response = call(
        model=model,
        messages=messages,
        tools=[SUBMIT_BID_TOOL],
        tool_choice={"type": "function", "function": {"name": "submit_bid"}},
    )
    args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
    return Bid(agent_role=role, **args)


def _score(bid: Bid, reputation: Reputation | None) -> float:
    # Confidence dominates; a small penalty for agents estimating a lot of steps
    # (proxy for approach complexity/cost) breaks ties in favor of the simpler plan.
    # Reputation is a gentle nudge, not a gate: centered on 0.5 (a role with no track
    # record yet gets zero adjustment), it shifts the score by at most +/-0.1 so a
    # strong bid from an unproven role can still win, but a role with a poor long-run
    # track record needs a genuinely better bid to beat one with a good one.
    reputation_term = 0.0
    if reputation is not None:
        reputation_term = 0.2 * (reputation.score(bid.agent_role) - 0.5)
    return bid.confidence - 0.02 * bid.estimated_steps + reputation_term


def run_contract_net(
    task: TaskNode,
    candidate_agents: list[AgentRole],
    model_by_role: dict[AgentRole, str],
    bb: Blackboard,
    budget: RunBudget | None = None,
    reputation: Reputation | None = None,
) -> AgentRole:
    """Runs one bounded bidding round among candidate_agents and returns the winning role.

    Logs the call-for-proposals, each bid, and the award + rationale to the decision log.
    """
    bb.log({"event": "call_for_proposals", "task_id": task.id, "candidates": [r.value for r in candidate_agents]})

    bids: list[Bid] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(candidate_agents)) as ex:
        futures = {
            ex.submit(_collect_bid, role, model_by_role[role], task, budget): role for role in candidate_agents
        }
        for future in concurrent.futures.as_completed(futures):
            role = futures[future]
            try:
                bid = future.result()
                bids.append(bid)
                bb.log({"event": "bid", "task_id": task.id, **bid.model_dump()})
            except Exception as e:
                bb.log({"event": "bid_error", "task_id": task.id, "role": role.value, "error": str(e)})

    if not bids:
        winner = candidate_agents[0]
        rationale = "no bids received (all bidders errored); defaulted to first candidate"
    else:
        best = max(bids, key=lambda b: _score(b, reputation))
        winner = best.agent_role
        others = [f"{b.agent_role.value}(confidence={b.confidence}, steps={b.estimated_steps})" for b in bids if b is not best]
        rep_note = f", reputation={reputation.score(winner):.2f}" if reputation is not None else ""
        rationale = (
            f"awarded to {winner.value} (confidence={best.confidence}, steps={best.estimated_steps}{rep_note}, "
            f"approach: {best.approach!r}) over {others or 'no other bidders'}"
        )

    bb.log({"event": "award", "task_id": task.id, "winner": winner.value, "rationale": rationale})
    return winner
