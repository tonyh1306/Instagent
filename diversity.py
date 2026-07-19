"""Prompt-variant pool per worker role, selected epsilon-greedily so a role doesn't
collapse onto one fixed style across many tasks/runs.

Without this, every dispatch of e.g. CoderAgent runs the exact same system prompt -
a monoculture where the "population" of one role is really a population of one. Each
role gets a small set of stance variants (still satisfying the same acceptance
criteria - these change approach, not requirements); a composite "role:vN" reputation
score (reputation.py) tracks which stances tend to pass the critic, and selection is
epsilon-greedy: mostly exploit the best-performing variant, but keep sampling the
others at rate `epsilon` so a variant that's merely unlucky early on (or one that
suits a task shape not yet seen) doesn't get permanently starved out.
"""

import random

from reputation import Reputation
from schemas import AgentRole

EPSILON = 0.2

VARIANTS: dict[AgentRole, list[str]] = {
    AgentRole.RESEARCHER: [
        "Approach: prioritize breadth - check multiple sources before concluding.",
        "Approach: prioritize speed - settle on the first solid, credible source and move on.",
    ],
    AgentRole.CODER: [
        "Approach: favor a thorough, defensive implementation - handle edge cases explicitly.",
        "Approach: favor the simplest implementation that satisfies the acceptance criteria.",
    ],
    AgentRole.WRITER: [
        "Approach: favor a detailed, comprehensive draft.",
        "Approach: favor a concise, minimal draft.",
    ],
}


def has_variants(role: AgentRole) -> bool:
    return role in VARIANTS


def variant_subject(role: AgentRole, variant_idx: int) -> str:
    return f"{role.value}:v{variant_idx}"


def select_variant(role: AgentRole, reputation: Reputation, epsilon: float = EPSILON) -> int:
    """Returns the index of the variant to use for this dispatch."""
    variants = VARIANTS[role]
    if random.random() < epsilon:
        return random.randrange(len(variants))
    scores = [reputation.score(variant_subject(role, i)) for i in range(len(variants))]
    return max(range(len(variants)), key=lambda i: scores[i])


def variant_suffix(role: AgentRole, variant_idx: int) -> str:
    return VARIANTS[role][variant_idx]
