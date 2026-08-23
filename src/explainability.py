"""Provenance-traced explanation chains for career recommendations.

Traces each recommended role back through the knowledge graph topology,
producing human-readable evidence chains showing the typed edges that
justify each recommendation.

Example output:
    Your role: Data Analyst
      -> TRANSITIONS_TO (0.12, n=847) -> Data Scientist
      -> REQUIRES -> Machine Learning (you need this)
      -> SIMILAR_TO (0.71) -> Data Scientist (ESCO)
      -> REQUIRES -> Python (you already have this)
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from src.kg_enrichment import role_skills


@dataclass
class ExplanationStep:
    relation: str
    source_id: str
    source_title: str
    target_id: str
    target_title: str
    attributes: dict

    def to_dict(self) -> dict:
        return {
            "relation": self.relation,
            "source": self.source_title,
            "target": self.target_title,
            "attributes": self.attributes,
        }


def trace_recommendation_path(
    source_role_id: str,
    target_role_id: str,
    owned_skill_ids: set[str],
    G: nx.MultiDiGraph,
) -> list[ExplanationStep]:
    """Trace typed edges connecting source to target role.

    Builds an explanation chain from:
    1. Direct TRANSITIONS_TO evidence (if exists)
    2. SIMILAR_TO alignment edges (if exists)
    3. Shared REQUIRES skills (transferable / owned)
    4. Missing REQUIRES skills (gap)
    """
    steps: list[ExplanationStep] = []

    if not G.has_node(source_role_id) or not G.has_node(target_role_id):
        return steps

    source_title = str(G.nodes[source_role_id].get("title", source_role_id))
    target_title = str(G.nodes[target_role_id].get("title", target_role_id))

    # 1. Direct transition evidence
    for _, tgt, data in G.out_edges(source_role_id, data=True):
        if (
            str(tgt) == str(target_role_id)
            and data.get("relation") == "TRANSITIONS_TO"
            and data.get("source") == "karrierewege"
        ):
            steps.append(ExplanationStep(
                relation="TRANSITIONS_TO",
                source_id=source_role_id,
                source_title=source_title,
                target_id=target_role_id,
                target_title=target_title,
                attributes={
                    "probability": round(float(data.get("probability", 0)), 4),
                    "count": int(data.get("count", 0)),
                    "evidence": "direct_observation",
                },
            ))
            break

    # 2. SIMILAR_TO edges involving target
    for src, tgt, data in G.edges(data=True):
        if data.get("relation") != "SIMILAR_TO":
            continue
        if str(src) == str(target_role_id) or str(tgt) == str(target_role_id):
            other = str(tgt) if str(src) == str(target_role_id) else str(src)
            other_title = str(G.nodes[other].get("title", other)) if G.has_node(other) else other
            steps.append(ExplanationStep(
                relation="SIMILAR_TO",
                source_id=target_role_id,
                source_title=target_title,
                target_id=other,
                target_title=other_title,
                attributes={
                    "similarity": round(float(data.get("similarity", 0)), 4),
                },
            ))
            break  # Only include the strongest alignment

    # 3. Shared skills (transferable)
    source_skills = role_skills(source_role_id, G)
    target_skills = role_skills(target_role_id, G)
    shared = source_skills & target_skills & owned_skill_ids

    for skill_id in sorted(shared)[:3]:
        if not G.has_node(skill_id):
            continue
        skill_title = str(G.nodes[skill_id].get("title", skill_id))
        steps.append(ExplanationStep(
            relation="REQUIRES",
            source_id=target_role_id,
            source_title=target_title,
            target_id=skill_id,
            target_title=skill_title,
            attributes={"status": "you_have_this", "transferable": True},
        ))

    # 4. Missing skills (gap)
    missing = target_skills - owned_skill_ids
    for skill_id in sorted(missing)[:3]:
        if not G.has_node(skill_id):
            continue
        skill_title = str(G.nodes[skill_id].get("title", skill_id))
        steps.append(ExplanationStep(
            relation="REQUIRES",
            source_id=target_role_id,
            source_title=target_title,
            target_id=skill_id,
            target_title=skill_title,
            attributes={"status": "you_need_this", "transferable": False},
        ))

    return steps


def format_explanation(steps: list[ExplanationStep]) -> str:
    """Format explanation steps as human-readable text."""
    if not steps:
        return "No traceable evidence path available."

    lines: list[str] = []
    for step in steps:
        attrs = step.attributes
        if step.relation == "TRANSITIONS_TO":
            prob = attrs.get("probability", 0)
            count = attrs.get("count", 0)
            lines.append(
                f"  -> TRANSITIONS_TO ({prob:.2%}, n={count}) -> {step.target_title}"
            )
        elif step.relation == "SIMILAR_TO":
            sim = attrs.get("similarity", 0)
            lines.append(
                f"  -> SIMILAR_TO ({sim:.2f}) -> {step.target_title}"
            )
        elif step.relation == "REQUIRES":
            status = attrs.get("status", "")
            marker = "(you already have this)" if "have" in status else "(you need this)"
            lines.append(
                f"  -> REQUIRES -> {step.target_title} {marker}"
            )

    return "\n".join(lines)


def explain_career_path(
    path_roles: list[dict],
    source_role_id: str | None,
    owned_skill_ids: set[str],
    G: nx.MultiDiGraph,
) -> list[dict]:
    """Generate explanation chains for each role in the career path.

    Returns list of {role_id, role_title, explanation_text, steps: [...]}
    """
    explanations: list[dict] = []

    if not source_role_id:
        return explanations

    for role_entry in path_roles:
        target_id = str(role_entry.get("id", ""))
        if target_id == source_role_id:
            continue
        if not G.has_node(target_id):
            continue

        steps = trace_recommendation_path(
            source_role_id, target_id, owned_skill_ids, G
        )

        explanations.append({
            "role_id": target_id,
            "role_title": role_entry.get("title", target_id),
            "explanation_text": format_explanation(steps),
            "steps": [s.to_dict() for s in steps],
        })

    return explanations
