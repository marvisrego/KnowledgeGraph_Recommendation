"""Ranking node — Cohere rerank + skill-gap ordering."""

from __future__ import annotations

import time
from typing import Any

from agents.state import CareerAgentState


def ranking_node(state: CareerAgentState, *, settings, G, **kwargs) -> dict[str, Any]:
    """Rerank candidates via Cohere, then sort by skill-gap accessibility."""
    from src.inference_pipeline import rerank_candidates
    from src.skill_gap import rank_roles_by_gap

    t0 = time.time()
    query = state["query"]
    candidates = state.get("candidates", [])
    skill_evidence = state.get("skill_evidence", {})
    current_role_id = state.get("current_role_id")

    if not candidates:
        return {"ranked_candidates": [], "anchor_ids": []}

    try:
        top_candidates = rerank_candidates(query, candidates, settings)
    except RuntimeError:
        top_candidates = candidates[: settings.rerank_top_n]

    top_candidates = rank_roles_by_gap(
        skill_evidence.get("skill_ids", set()),
        top_candidates,
        G,
        current_role_id=current_role_id,
        onet_importance_threshold=settings.onet_importance_threshold,
    )

    anchor_ids = [c["id"] for c in top_candidates]

    return {
        "ranked_candidates": top_candidates,
        "anchor_ids": anchor_ids,
        "metadata": {**state.get("metadata", {}), "ranking_ms": int((time.time() - t0) * 1000)},
    }
