"""Retrieval node — vector search + transition augmentation."""

from __future__ import annotations

import time
from typing import Any

from agents.state import CareerAgentState


def retrieval_node(
    state: CareerAgentState,
    *,
    settings,
    G,
    collection,
    transition_smoother=None,
    link_prediction_runtime=None,
    sequential_runtime=None,
    **kwargs,
) -> dict[str, Any]:
    """Resolve evidence and run the shared multi-source retriever."""
    from src.hybrid_retrieval import hybrid_retrieve
    from src.skill_gap import resolve_current_role, resolve_user_skills
    from src.career_history import resolve_career_history

    t0 = time.time()
    query = state["query"]
    intent = state.get("intent", {})
    user_type = state.get("user_type", "professional")
    messages = state.get("messages", [])

    # Resolve skills and current role
    from src.inference_pipeline import _user_authored_texts
    user_texts = _user_authored_texts(query, messages)
    router_skills = intent.get("skills", [])
    if isinstance(router_skills, str):
        router_skills = [router_skills]
    elif not isinstance(router_skills, list):
        router_skills = []

    try:
        skill_evidence = resolve_user_skills(router_skills, user_texts, G)
        current_role_id = None
        if user_type == "professional":
            current_role_id = resolve_current_role(str(intent.get("current_role", "")), G)
    except Exception:
        skill_evidence = {"skill_ids": set(), "matched": [], "unmatched": list(router_skills), "excluded": []}
        current_role_id = None

    history_role_ids = resolve_career_history(
        intent.get("career_history", []), current_role_id, G
    )

    # Build public evidence
    public_evidence = {
        "current_role": (
            {"id": current_role_id, "title": G.nodes[current_role_id].get("title", current_role_id)}
            if current_role_id and G.has_node(current_role_id) else None
        ),
        "matched_skills": [{"phrase": m["phrase"], "id": m["id"], "title": m["title"]} for m in skill_evidence["matched"]],
        "unmatched_skills": skill_evidence["unmatched"],
        "excluded_non_owned_skills": skill_evidence.get("excluded", []),
        "resolved_career_history": [
            {"id": role_id, "title": G.nodes[role_id].get("title", role_id)}
            for role_id in history_role_ids
            if G.has_node(role_id)
        ],
    }

    candidates = hybrid_retrieve(
        query,
        collection,
        skill_evidence["skill_ids"],
        current_role_id,
        G,
        settings,
        transition_smoother=transition_smoother,
        link_prediction_runtime=link_prediction_runtime,
        sequential_runtime=sequential_runtime,
        history_role_ids=history_role_ids,
    )

    return {
        "candidates": candidates,
        "skill_evidence": skill_evidence,
        "current_role_id": current_role_id,
        "history_role_ids": history_role_ids,
        "owned_skill_ids": skill_evidence["skill_ids"],
        "public_evidence": public_evidence,
        "metadata": {**state.get("metadata", {}), "retrieval_ms": int((time.time() - t0) * 1000)},
    }
