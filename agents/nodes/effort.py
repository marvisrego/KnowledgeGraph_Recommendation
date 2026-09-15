"""Effort scoring node — compute TES for each shortlisted role."""

from __future__ import annotations

import time
from typing import Any

from agents.state import CareerAgentState


def effort_node(state: CareerAgentState, *, settings, G, transition_smoother=None, **kwargs) -> dict[str, Any]:
    """Compute Transition Effort Score for each role in the path."""
    from src.transition_effort import transition_effort_score, get_idf_map
    from src.tes_calibration import load_calibration
    from src.inference_pipeline import _build_path_data

    t0 = time.time()
    anchor_ids = state.get("anchor_ids", [])
    triples = state.get("triples", [])
    ranked_candidates = state.get("ranked_candidates", [])
    current_role_id = state.get("current_role_id")
    owned_skill_ids = state.get("owned_skill_ids", set())

    path_data = _build_path_data(anchor_ids, triples, G, ranked_candidates=ranked_candidates)

    roles = path_data.get("roles", [])
    # Use explicit current role; fall back to first recommended role as synthetic source
    # so students (who have no current_role_id) still get effort scores.
    effective_source_id = current_role_id
    effort_source_label = "from_current_role"
    if not effective_source_id and len(roles) >= 2:
        effective_source_id = roles[0]["id"]
        effort_source_label = "relative"

    if effective_source_id and roles:
        idf_map = get_idf_map(G)
        calibration = load_calibration(settings.effort_calibration_path)
        for role_entry in roles:
            target_id = role_entry["id"]
            if target_id == effective_source_id:
                continue
            try:
                effort = transition_effort_score(
                    effective_source_id,
                    target_id,
                    owned_skill_ids,
                    G,
                    idf_map,
                    None,
                    transition_smoother,
                    settings.onet_importance_threshold,
                    calibration,
                )
                role_entry["effort_score"] = round(effort.score, 4)
                role_entry["effort_band"] = effort.band
                role_entry["effort_source"] = effort_source_label
                role_entry["estimated_weeks_min"] = effort.estimated_weeks_min
                role_entry["estimated_weeks_max"] = effort.estimated_weeks_max
                role_entry["estimated_weeks_basis"] = "heuristic"
                role_entry["effort_calibration"] = effort.calibration.to_dict()
            except Exception as exc:
                print(f"[effort_node] Skipping effort for {target_id}: {exc}")

    # Re-sort roles by effort ascending so the frontend shows easiest transitions first.
    # Roles without a score (no source role matched) sort to the end.
    path_data["roles"].sort(
        key=lambda r: (r.get("effort_score") is None, r.get("effort_score", 999))
    )

    return {
        "path_data": path_data,
        "metadata": {**state.get("metadata", {}), "effort_ms": int((time.time() - t0) * 1000)},
    }
