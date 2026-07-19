"""Wire-format contracts shared by every agent and the blackboard.

No agent should ever pass raw strings between each other — orchestrator, workers,
critic, and arbiter all read/write these typed models exclusively.
"""

from datetime import datetime, timezone
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AgentRole(StrEnum):
    ORCHESTRATOR = "orchestrator"
    RESEARCHER = "researcher"
    CODER = "coder"
    WRITER = "writer"
    CRITIC = "critic"
    ARBITER = "arbiter"
    BASELINE = "baseline"  # single-agent comparison runner; not part of the DAG pipeline


class TaskStatus(StrEnum):
    PENDING = "pending"  # blocked on unmet dependencies
    READY = "ready"  # dependencies satisfied, eligible for dispatch
    IN_PROGRESS = "in_progress"  # dispatched to a worker
    IN_REVIEW = "in_review"  # worker output awaiting critic verdict
    REPAIR = "repair"  # critic failed it once, sent back for one repair attempt
    ESCALATED = "escalated"  # repair failed too; routed to council/arbiter
    COMMITTED = "committed"  # critic-approved, terminal success state
    DEAD_LETTER = "dead_letter"  # repair + escalation exhausted, terminal failure state


class TaskNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    description: str
    assigned_agent: AgentRole
    dependencies: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    acceptance_criteria: list[str] = Field(default_factory=list)
    attempts: int = Field(default=0, description="repair attempts consumed so far")
    artifact_ref: str | None = Field(
        default=None, description="blackboard key holding this task's committed output"
    )
    candidate_agents: list[AgentRole] = Field(
        default_factory=list,
        description="2+ roles that could plausibly do this task; non-empty triggers contract-net bidding",
    )
    arbiter_ruled: bool = Field(
        default=False, description="true once a council/arbiter compromise ruling has been applied - bounds re-escalation"
    )
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class BlackboardEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    value: object
    written_by: AgentRole
    timestamp: datetime = Field(default_factory=utcnow)
    version: int = Field(default=1, description="optimistic-concurrency version; bump on every write")


class AgentMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    from_agent: AgentRole
    to_agent_or_broadcast: AgentRole | str = Field(
        description="a specific AgentRole, or the literal string 'broadcast'"
    )
    task_id: str
    content: str
    artifact_ref: str | None = Field(
        default=None, description="blackboard key holding a referenced artifact, if any"
    )


class CriticVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    reasons: list[str] = Field(default_factory=list, description="structured reasons; required when passed=False")


class ArbiterDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["accept", "reject", "accept_with_repair"]
    rationale: str
    repair_instructions: str | None = Field(
        default=None,
        description="required when decision=='accept_with_repair': synthesis guidance for the one bounded final repair attempt",
    )


class Bid(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_role: AgentRole
    confidence: float = Field(ge=0.0, le=1.0)
    approach: str
    estimated_steps: int = Field(ge=1)


class CouncilTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    round: int
    speaker: AgentRole
    turn_type: Literal["claim", "evidence", "concession", "rebuttal"]
    text: str


class ConflictEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    attempted_by: AgentRole
    expected_version: int | None
    actual_version: int | None
    resolution: Literal["reread_and_merge", "escalated_to_council"]
    timestamp: datetime = Field(default_factory=utcnow)
