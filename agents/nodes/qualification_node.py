"""Qualification scoring node — fraction of essential skills the user already has per candidate."""

from __future__ import annotations

import time
from typing import Any

from agents.state import CareerAgentState


def qualification_node(state: CareerAgentState, *, settings, G, **kwargs) -> dict[str, Any]:
    """Compute IDF-weighted qualification score per ranked candidate role.

    Score = IDF(owned ∩ required) / IDF(required).
    Written to state as qualification_scores: dict[role_id, float] and also
    embedded onto each ranked_candidate dict for downstream use.
    """
    from src.skill_gap import role_requirements
    from src.transition_effort import get_idf_map

    t0 = time.time()
    ranked_candidates = state.get("ranked_candidates", [])
    owned_skill_ids: set[str] = state.get("owned_skill_ids", set())

    if not ranked_candidates or not owned_skill_ids:
        return {
            "qualification_scores": {},
            "metadata": {**state.get("metadata", {}), "qualification_ms": 0},
        }

    idf_map = get_idf_map(G)
    qualification_scores: dict[str, float] = {}

    for candidate in ranked_candidates:
        role_id = candidate.get("id", "")
        if not role_id:
            continue
        try:
            required, _, _ = role_requirements(role_id, G, settings.onet_importance_threshold)
            if not required:
                qualification_scores[role_id] = 0.0
                candidate["qualification_score"] = 0.0
                continue
            total_idf = sum(idf_map.get(s, 1.0) for s in required)
            if total_idf == 0:
                qualification_scores[role_id] = 0.0
                candidate["qualification_score"] = 0.0
                continue
            owned_idf = sum(idf_map.get(s, 1.0) for s in required if s in owned_skill_ids)
            score = min(1.0, owned_idf / total_idf)
            qualification_scores[role_id] = round(score, 4)
            candidate["qualification_score"] = round(score, 4)
        except Exception as exc:
            print(f"[qualification_node] {role_id}: {exc}")

    return {
        "qualification_scores": qualification_scores,
        "ranked_candidates": ranked_candidates,
        "metadata": {**state.get("metadata", {}), "qualification_ms": int((time.time() - t0) * 1000)},
    }
