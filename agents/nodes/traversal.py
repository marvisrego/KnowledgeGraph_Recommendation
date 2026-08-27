"""Graph traversal node — collect REQUIRES, SIMILAR_TO, TRANSITIONS_TO triples."""

from __future__ import annotations

import time
from typing import Any

from agents.state import CareerAgentState


def traversal_node(state: CareerAgentState, *, settings, G, **kwargs) -> dict[str, Any]:
    """Traverse graph neighbourhood for anchor roles."""
    from src.inference_pipeline import traverse_graph

    t0 = time.time()
    anchor_ids = state.get("anchor_ids", [])
    current_role_id = state.get("current_role_id")
    ranked_candidates = state.get("ranked_candidates", [])

    if not anchor_ids:
        return {"triples": []}

    triples = traverse_graph(
        anchor_ids, G, settings.onet_importance_threshold,
        transition_limit=settings.transition_traversal_limit,
    )

    # Add candidate transition triples
    seen_transition_triples = {
        (src, tgt) for src, rel, tgt, _ in triples if rel == "TRANSITIONS_TO"
    }
    for candidate in ranked_candidates:
        transition = candidate.get("transition")
        if not transition or not current_role_id:
            continue
        evidence_type = transition.get("evidence_type", "direct_transition")
        relation = {
            "semantic_transition_backoff": "SEMANTIC_TRANSITION_BACKOFF",
            "predicted_transition": "PREDICTED_TRANSITION",
        }.get(evidence_type, "TRANSITIONS_TO")
        key = (current_role_id, candidate["id"])
        if relation == "TRANSITIONS_TO" and key in seen_transition_triples:
            continue
        triples.append((
            current_role_id, relation, candidate["id"],
            {
                "relation": relation,
                "source": (
                    "embedding_smoothing"
                    if relation == "SEMANTIC_TRANSITION_BACKOFF"
                    else "link_prediction"
                    if relation == "PREDICTED_TRANSITION"
                    else "karrierewege"
                ),
                **transition,
            },
        ))
        if relation == "TRANSITIONS_TO":
            seen_transition_triples.add(key)

    return {
        "triples": triples,
        "metadata": {**state.get("metadata", {}), "traversal_ms": int((time.time() - t0) * 1000)},
    }
