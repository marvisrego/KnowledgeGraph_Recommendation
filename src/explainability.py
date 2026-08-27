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

from src.transition_policy import is_training_transition

from src.skill_gap import role_requirements


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

    # 1. Strongest direct transition evidence
    direct_edges = [
        data
        for _, tgt, data in G.out_edges(source_role_id, data=True)
        if str(tgt) == str(target_role_id) and is_training_transition(data)
    ]
    if direct_edges:
        data = max(
            direct_edges,
            key=lambda edge: (
                float(edge.get("probability", 0.0)),
                int(edge.get("count", 0)),
            ),
        )
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

    # 2. Strongest SIMILAR_TO edge involving target
    alignments: list[tuple[float, str, dict]] = []
    for _, other, data in G.out_edges(target_role_id, data=True):
        if data.get("relation") == "SIMILAR_TO":
            alignments.append((float(data.get("similarity", 0.0)), str(other), data))
    for other, _, data in G.in_edges(target_role_id, data=True):
        if data.get("relation") == "SIMILAR_TO":
            alignments.append((float(data.get("similarity", 0.0)), str(other), data))
    if alignments:
        similarity, other, data = max(alignments, key=lambda item: (item[0], item[1]))
        other_title = str(G.nodes[other].get("title", other)) if G.has_node(other) else other
        steps.append(ExplanationStep(
                relation="SIMILAR_TO",
                source_id=target_role_id,
                source_title=target_title,
                target_id=other,
                target_title=other_title,
                attributes={
                    "similarity": round(similarity, 4),
                },
        ))

    # 3. Shared skills (transferable)
    source_skills, _, _ = role_requirements(source_role_id, G)
    target_skills, _, _ = role_requirements(target_role_id, G)
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
        elif step.relation == "PREDICTED_TRANSITION":
            score = attrs.get("score", 0)
            lines.append(
                f"  -> PREDICTED_TRANSITION (model score {score:.2%}) -> {step.target_title}"
            )
        elif step.relation == "SEMANTIC_TRANSITION_BACKOFF":
            support = attrs.get("neighbour_support", 0)
            lines.append(
                f"  -> SEMANTIC_TRANSITION_BACKOFF ({support} related roles) -> {step.target_title}"
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
        transition = role_entry.get("transition") or {}
        evidence_type = transition.get("evidence_type")
        relation = {
            "predicted_transition": "PREDICTED_TRANSITION",
            "semantic_transition_backoff": "SEMANTIC_TRANSITION_BACKOFF",
        }.get(evidence_type)
        if relation and not any(step.relation == "TRANSITIONS_TO" for step in steps):
            steps.insert(0, ExplanationStep(
                relation=relation,
                source_id=source_role_id,
                source_title=str(G.nodes[source_role_id].get("title", source_role_id)),
                target_id=target_id,
                target_title=str(G.nodes[target_id].get("title", target_id)),
                attributes={
                    key: value
                    for key, value in transition.items()
                    if key in {"score", "neighbour_probability", "neighbour_support", "model"}
                },
            ))

        explanations.append({
            "role_id": target_id,
            "role_title": role_entry.get("title", target_id),
            "explanation_text": format_explanation(steps),
            "steps": [s.to_dict() for s in steps],
        })

    return explanations
