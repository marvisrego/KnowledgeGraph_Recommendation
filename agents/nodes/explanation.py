"""Explanation node — trace evidence chains for each recommendation."""

from __future__ import annotations

import time
from typing import Any

from agents.state import CareerAgentState


def explanation_node(state: CareerAgentState, *, G, **kwargs) -> dict[str, Any]:
    """Generate provenance-traced explanation chains for path roles."""
    from src.explainability import explain_career_path

    t0 = time.time()
    path_data = state.get("path_data", {})
    current_role_id = state.get("current_role_id")
    owned_skill_ids = state.get("owned_skill_ids", set())

    explanations = []
    if current_role_id and path_data.get("roles"):
        try:
            explanations = explain_career_path(
                path_data["roles"], current_role_id, owned_skill_ids, G
            )
        except Exception:
            pass

    return {
        "explanations": explanations,
        "metadata": {**state.get("metadata", {}), "explanation_ms": int((time.time() - t0) * 1000)},
    }
