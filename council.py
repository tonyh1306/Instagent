"""Bounded conflict resolution: invoked only when the critic gate can't deterministically
resolve a task (repair attempt already exhausted). The worker and critic exchange typed,
cross-visible turns (claim/evidence/concession/rebuttal) for at most MAX_ROUNDS rounds -
each party sees the other's most recent turn and must respond to its specific points, not
just restate their own position. The arbiter then reads the full transcript and rules once:
accept, reject, or accept-with-repair (a bounded final synthesis attempt).
"""

import json

from agents.arbiter import ArbiterAgent
from blackboard import Blackboard
from budget import RunBudget
from qwen_client import call
from schemas import AgentRole, ArbiterDecision, CouncilTurn, TaskNode

MAX_ROUNDS = 2

SUBMIT_TURN_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_turn",
        "description": "Submit your turn in this council round.",
        "parameters": {
            "type": "object",
            "properties": {
                "turn_type": {
                    "type": "string",
                    "enum": ["claim", "evidence", "concession", "rebuttal"],
                    "description": (
                        "claim: a position you're asserting. evidence: concrete support for a "
                        "claim. concession: you agree with a specific point the other party made. "
                        "rebuttal: you specifically counter a point the other party made."
                    ),
                },
                "text": {"type": "string", "description": "1-3 sentences. Address the other party's specific last point."},
            },
            "required": ["turn_type", "text"],
        },
    },
}


def _speak_turn(
    role: AgentRole,
    model: str,
    system_prompt: str,
    transcript: list[dict],
    round_num: int,
    budget: RunBudget | None = None,
) -> CouncilTurn:
    if budget:
        budget.charge_llm_call()
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": _render_transcript(transcript)
            + "\n\nSubmit your turn now via submit_turn. If the other party has spoken, your "
            "text must directly address their most recent specific point - don't just restate "
            "your earlier position.",
        },
    ]
    response = call(
        model=model,
        messages=messages,
        tools=[SUBMIT_TURN_TOOL],
        tool_choice={"type": "function", "function": {"name": "submit_turn"}},
    )
    args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
    return CouncilTurn(round=round_num, speaker=role, turn_type=args["turn_type"], text=args["text"])


def _render_transcript(entries: list[dict]) -> str:
    return "\n\n".join(
        f"[{e['speaker']}" + (f" - {e['turn_type']}]" if e.get("turn_type") else "]") + f" {e['text']}"
        for e in entries
    )


def resolve(
    node: TaskNode,
    artifact_content: str,
    critic_reasons: list[str],
    worker_model: str,
    budget: RunBudget | None = None,
    bb: Blackboard | None = None,
) -> dict:
    """Runs the bounded worker<->critic dialogue, then has the arbiter rule.

    Returns {"decision": ArbiterDecision, "transcript": list[dict]} - the caller (orchestrator)
    is responsible for applying the decision to the task's status. If bb is given, each turn is
    also logged to the decision log as it happens (visible negotiation, not just a final blob).
    """
    worker_role = node.assigned_agent
    worker_system = (
        f"You are the {worker_role.value} agent defending your artifact in a bounded, {MAX_ROUNDS}-round "
        "council review against the critic's rejection. State your case with claim/evidence turns, "
        "but concede specific points that are actually valid - the arbiter penalizes stonewalling."
    )
    critic_system = (
        f"You are the critic agent in a bounded, {MAX_ROUNDS}-round council review. Press the worker "
        "on the specific acceptance criteria at issue with claim/evidence/rebuttal turns. If the "
        "worker concedes or clarifies something that resolves a concern, acknowledge it - the "
        "arbiter penalizes relitigating settled points."
    )

    transcript: list[dict] = [
        {
            "speaker": "system",
            "turn_type": None,
            "text": (
                f"Task: {node.description}\n"
                f"Acceptance criteria: {node.acceptance_criteria}\n"
                f"Worker's artifact:\n{artifact_content}\n"
                f"Critic's rejection reasons (after a repair attempt already failed):\n"
                + "\n".join(f"- {r}" for r in critic_reasons)
            ),
        }
    ]

    for round_num in range(1, MAX_ROUNDS + 1):
        worker_turn = _speak_turn(worker_role, worker_model, worker_system, transcript, round_num, budget=budget)
        transcript.append(worker_turn.model_dump(mode="json"))
        if bb:
            bb.log({"event": "council_turn", "task_id": node.id, **worker_turn.model_dump(mode="json")})

        critic_turn = _speak_turn(AgentRole.CRITIC, "qwen-plus", critic_system, transcript, round_num, budget=budget)
        transcript.append(critic_turn.model_dump(mode="json"))
        if bb:
            bb.log({"event": "council_turn", "task_id": node.id, **critic_turn.model_dump(mode="json")})

    arbiter = ArbiterAgent(budget=budget)
    wrapper_task = TaskNode(
        id=node.id,
        description=node.description,
        assigned_agent=AgentRole.ARBITER,
        acceptance_criteria=node.acceptance_criteria,
    )
    result = arbiter.run(wrapper_task, _render_transcript(transcript))
    decision = ArbiterDecision.model_validate(result.structured)

    return {"decision": decision, "transcript": transcript}
