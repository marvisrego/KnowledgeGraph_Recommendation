"""Effort scoring node — compute TES for each shortlisted role."""

from __future__ import annotations

import time
from typing import Any

from agents.state import CareerAgentState


def effort_node(state: CareerAgentState, *, settings, G, transition_smoother=None, **kwargs) -> dict[str, Any]:
    """Compute Transition Effort Score for each role in the path."""
    from src.transition_effort import transition_effort_score, get_idf_map, EffortWeights
    from src.inference_pipeline import _build_path_data

    t0 = time.time()
    anchor_ids = state.get("anchor_ids", [])
    triples = state.get("triples", [])
    ranked_candidates = state.get("ranked_candidates", [])
    current_role_id = state.get("current_role_id")
    owned_skill_ids = state.get("owned_skill_ids", set())

    path_data = _build_path_data(anchor_ids, triples, G, ranked_candidates=ranked_candidates)

    if current_role_id and path_data.get("roles"):
        idf_map = get_idf_map(G)
        weights = EffortWeights(
            skill_gap=settings.effort_weight_skill_gap,
            domain=settings.effort_weight_domain,
            empirical=settings.effort_weight_empirical,
            transferability=settings.effort_weight_transferability,
        )
        for role_entry in path_data["roles"]:
            target_id = role_entry["id"]
            if target_id == current_role_id:
                continue
            try:
                effort = transition_effort_score(
                    current_role_id,
                    target_id,
                    owned_skill_ids,
                    G,
                    idf_map,
                    weights,
                    transition_smoother,
                    settings.onet_importance_threshold,
                )
                role_entry["effort_score"] = round(effort.score, 4)
                role_entry["effort_band"] = effort.band
            except Exception:
                pass

    return {
        "path_data": path_data,
        "metadata": {**state.get("metadata", {}), "effort_ms": int((time.time() - t0) * 1000)},
    }
