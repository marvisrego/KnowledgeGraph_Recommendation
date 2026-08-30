"""Shared state schema for the LangGraph career advisor workflow."""

from __future__ import annotations

from typing import Any, TypedDict


class CareerAgentState(TypedDict, total=False):
    """Full state passed between LangGraph nodes.

    Each node reads what it needs and writes its outputs.
    Fields are optional (total=False) so nodes can run incrementally.
    """
    # Input
    query: str
    messages: list[dict[str, str]]

    # Intent routing
    intent: dict[str, Any]
    user_type: str
    has_context: bool

    # Skill/role evidence
    skill_evidence: dict[str, Any]
    current_role_id: str | None
    owned_skill_ids: set[str]

    # Retrieval
    candidates: list[dict]
    ranked_candidates: list[dict]
    anchor_ids: list[str]

    # Graph traversal
    triples: list[tuple]

    # Qualification (fraction of essential skills owned per role)
    qualification_scores: dict[str, float]

    # Effort
    effort_scores: dict[str, dict]

    # Skill gap analysis (priority missing skills per role)
    skill_gap_analysis: list[dict]

    # Learning plan (phased roadmap per role)
    learning_plan: list[dict]

    # Generation
    response: str

    # Courses
    courses: list[dict]

    # Path data
    path_data: dict

    # Explore (partial context)
    explore_data: dict

    # Faithfulness
    faithfulness: dict | None

    # Explanations
    explanations: list[dict]

    # Evidence for frontend
    public_evidence: dict

    # Errors
    errors: list[str]

    # Metadata (timings, token usage)
    metadata: dict[str, Any]
