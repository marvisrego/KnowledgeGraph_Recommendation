"""Skill Gap Analysis node — ranks missing skills by impact on TES reduction."""

from __future__ import annotations

import time
from typing import Any

from agents.state import CareerAgentState


def skill_gap_node(state: CareerAgentState, *, G, settings, **kwargs) -> dict[str, Any]:
    """For each role in path_data, compute ranked priority skill gaps.

    Returns skill_gap_analysis: list of per-role dicts with priority_skills,
    quick_wins (skills user nearly has), and blockers (high-IDF fully missing skills).
    """
    from src.skill_gap import role_requirements
    from src.transition_effort import get_idf_map

    t0 = time.time()
    path_data = state.get("path_data", {})
    owned_skill_ids: set[str] = state.get("owned_skill_ids", set())

    roles = path_data.get("roles", [])
    if not roles:
        return {
            "skill_gap_analysis": [],
            "metadata": {**state.get("metadata", {}), "skill_gap_ms": 0},
        }

    idf_map = get_idf_map(G)
    skill_gap_analysis: list[dict] = []

    for role_entry in roles:
        role_id = role_entry.get("id", "")
        role_title = role_entry.get("title", role_id)
        if not role_id:
            continue
        try:
            required, _, _ = role_requirements(role_id, G, settings.onet_importance_threshold)
            if not required:
                continue

            # Skill labels from graph node titles
            def skill_label(sid: str) -> str:
                return G.nodes.get(sid, {}).get("title", sid)[:40]

            missing = required - owned_skill_ids
            owned_required = required & owned_skill_ids

            # Sort missing by IDF descending (rarest = most impactful)
            missing_sorted = sorted(
                missing,
                key=lambda s: idf_map.get(s, 1.0),
                reverse=True,
            )

            # Estimate TES reduction per skill: IDF_weight / total_required_IDF
            total_idf = sum(idf_map.get(s, 1.0) for s in required) or 1.0

            priority_skills = [
                {
                    "skill": skill_label(s),
                    "idf_weight": round(idf_map.get(s, 1.0), 3),
                    "tes_reduction": round(idf_map.get(s, 1.0) / total_idf, 4),
                }
                for s in missing_sorted[:8]
            ]

            # Quick wins: owned skills with high IDF (skills that make you stand out)
            quick_wins = sorted(
                [
                    {"skill": skill_label(s), "idf_weight": round(idf_map.get(s, 1.0), 3)}
                    for s in owned_required
                ],
                key=lambda x: x["idf_weight"],
                reverse=True,
            )[:5]

            # Blockers: high-IDF fully missing (top 3)
            blockers = [
                {"skill": skill_label(s), "idf_weight": round(idf_map.get(s, 1.0), 3)}
                for s in missing_sorted[:3]
            ]

            skill_gap_analysis.append({
                "role_id": role_id,
                "role_title": role_title,
                "missing_count": len(missing),
                "priority_skills": priority_skills,
                "quick_wins": quick_wins,
                "blockers": blockers,
            })
        except Exception as exc:
            print(f"[skill_gap_node] {role_id}: {exc}")

    return {
        "skill_gap_analysis": skill_gap_analysis,
        "metadata": {**state.get("metadata", {}), "skill_gap_ms": int((time.time() - t0) * 1000)},
    }
