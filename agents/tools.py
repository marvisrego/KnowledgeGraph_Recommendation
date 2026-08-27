"""Tool definitions for the Career Graph Agent.

Each tool wraps a graph query function with a typed interface.
These can be exposed as LangGraph tools or MCP-compatible endpoints.
"""

from __future__ import annotations

from typing import Any

import networkx as nx

from src.kg_enrichment import role_skills
from src.skill_gap import (
    compute_skill_gap,
    resolve_current_role,
    transition_destinations,
    transition_evidence,
)
from src.transition_policy import is_training_transition
from src.transition_effort import (
    EffortWeights,
    get_idf_map,
    transition_effort_score,
)


def search_career_graph(query_skills: list[str], G: nx.MultiDiGraph, top_k: int = 10) -> list[dict]:
    """Find roles matching a set of skills via graph-structural overlap."""
    from src.hybrid_retrieval import graph_retrieve
    skill_index = {}
    for nid, d in G.nodes(data=True):
        if d.get("type") == "skill":
            from src.text_normalization import normalize_label
            norm = normalize_label(str(d.get("title", "")))
            if norm:
                skill_index[norm] = str(nid)

    from src.text_normalization import normalize_label
    skill_ids = set()
    for s in query_skills:
        norm = normalize_label(s)
        if norm in skill_index:
            skill_ids.add(skill_index[norm])

    return graph_retrieve(skill_ids, G, top_k=top_k)


def find_role_skills(role_phrase: str, G: nx.MultiDiGraph) -> dict:
    """Find a role and return its required skills."""
    role_id = resolve_current_role(role_phrase, G)
    if not role_id:
        return {"error": f"Role not found: {role_phrase}", "skills": []}

    skills = role_skills(role_id, G)
    skill_list = []
    for sid in sorted(skills):
        if G.has_node(sid):
            skill_list.append({"id": sid, "title": G.nodes[sid].get("title", sid)})

    return {
        "role_id": role_id,
        "role_title": G.nodes[role_id].get("title", role_id),
        "skill_count": len(skill_list),
        "skills": skill_list[:20],
    }


def find_related_roles(role_phrase: str, G: nx.MultiDiGraph, limit: int = 10) -> dict:
    """Find roles related to a given role via SIMILAR_TO and TRANSITIONS_TO."""
    role_id = resolve_current_role(role_phrase, G)
    if not role_id:
        return {"error": f"Role not found: {role_phrase}", "related": []}

    related = []
    for _, target, data in G.out_edges(role_id, data=True):
        rel = data.get("relation", "")
        if rel == "SIMILAR_TO" or (rel == "TRANSITIONS_TO" and is_training_transition(data)):
            if G.has_node(target):
                related.append({
                    "id": str(target),
                    "title": G.nodes[target].get("title", target),
                    "relation": rel,
                    "weight": float(data.get("similarity", data.get("probability", 0))),
                })

    related.sort(key=lambda x: -x["weight"])
    return {
        "role_id": role_id,
        "role_title": G.nodes[role_id].get("title", role_id),
        "related_count": len(related),
        "related": related[:limit],
    }


def calculate_skill_gap(
    role_phrase: str,
    user_skills: list[str],
    G: nx.MultiDiGraph,
) -> dict:
    """Calculate skill gap between user's skills and a target role."""
    role_id = resolve_current_role(role_phrase, G)
    if not role_id:
        return {"error": f"Role not found: {role_phrase}"}

    from src.skill_gap import resolve_user_skills
    evidence = resolve_user_skills(user_skills, user_skills, G)
    gap = compute_skill_gap(evidence["skill_ids"], role_id, G)
    return {
        "role_id": role_id,
        "role_title": G.nodes[role_id].get("title", role_id),
        **gap,
        "have_ids": list(gap.get("have_ids", set())),
        "need_ids": list(gap.get("need_ids", set())),
    }


def calculate_effort(
    source_role: str,
    target_role: str,
    user_skills: list[str],
    G: nx.MultiDiGraph,
    settings: Any = None,
) -> dict:
    """Calculate transition effort score between two roles."""
    source_id = resolve_current_role(source_role, G)
    target_id = resolve_current_role(target_role, G)

    if not source_id:
        return {"error": f"Source role not found: {source_role}"}
    if not target_id:
        return {"error": f"Target role not found: {target_role}"}

    from src.skill_gap import resolve_user_skills
    evidence = resolve_user_skills(user_skills, user_skills, G)
    idf_map = get_idf_map(G)

    weights = EffortWeights()
    if settings:
        weights = EffortWeights(
            skill_gap=settings.effort_weight_skill_gap,
            domain=settings.effort_weight_domain,
            empirical=settings.effort_weight_empirical,
            transferability=settings.effort_weight_transferability,
        )

    result = transition_effort_score(
        source_id, target_id, evidence["skill_ids"], G, idf_map, weights
    )

    return {
        "source_role": G.nodes[source_id].get("title", source_id),
        "target_role": G.nodes[target_id].get("title", target_id),
        **result.to_dict(),
    }


def get_transition_evidence(
    source_role: str,
    target_role: str,
    G: nx.MultiDiGraph,
) -> dict:
    """Get empirical transition evidence between two roles."""
    source_id = resolve_current_role(source_role, G)
    target_id = resolve_current_role(target_role, G)

    if not source_id or not target_id:
        return {"evidence": None, "error": "One or both roles not found"}

    evidence = transition_evidence(source_id, target_id, G)
    destinations = transition_destinations(source_id, G, limit=5)

    return {
        "source": G.nodes[source_id].get("title", source_id),
        "target": G.nodes[target_id].get("title", target_id),
        "direct_evidence": evidence,
        "top_destinations": [
            {"title": G.nodes[d["id"]].get("title", d["id"]), **d["transition"]}
            for d in destinations
        ],
    }
