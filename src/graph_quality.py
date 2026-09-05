"""Deterministic quality controls for the career knowledge graph."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import Any

import networkx as nx

from src.text_normalization import normalize_label
from src.transition_policy import is_training_transition


ALLOWED_NODE_TYPES = {
    "role",
    "skill",
    "element",
    "isco_group",
    "skill_group",
    "qualification",
    "job_zone",
}
ALLOWED_RELATIONS = {
    "REQUIRES",
    "BELONGS_TO",
    "BROADER_THAN",
    "NARROWER_THAN",
    "SIMILAR_TO",
    "SAME_ISCO_GROUP",
    "TRANSITIONS_TO",
    "TYPICALLY_REQUIRES_QUALIFICATION",
    "IN_JOB_ZONE",
}


@dataclass
class GraphQualityReport:
    nodes_before: int
    edges_before: int
    nodes_after: int = 0
    edges_after: int = 0
    removed_invalid_nodes: int = 0
    removed_invalid_edges: int = 0
    removed_self_loops: int = 0
    removed_held_out_transitions: int = 0
    removed_duplicate_edges: int = 0
    removed_isolates: int = 0
    added_qualification_nodes: int = 0
    added_job_zone_nodes: int = 0
    added_connectivity_edges: int = 0
    normalized_label_collisions: int = 0
    node_types: dict[str, int] | None = None
    relationship_types: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _slug(value: object) -> str:
    normalized = normalize_label(value)
    return re.sub(r"[^a-z0-9]+", "-", normalized).strip("-") or "unknown"


def _finite_edge_values(data: dict) -> bool:
    for field in ("probability", "similarity", "score", "requirement_level", "level_score"):
        value = data.get(field)
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            return False
    for field in ("count", "source_total", "min_support"):
        value = data.get(field)
        if value is not None:
            try:
                if int(value) < 0:
                    return False
            except (TypeError, ValueError):
                return False
    probability = data.get("probability")
    if probability is not None and not 0.0 <= float(probability) <= 1.0:
        return False
    return True


def _edge_preference(data: dict) -> tuple:
    return (
        int(str(data.get("requirement_level", "")).casefold() == "essential"),
        int(data.get("count", 0) or 0),
        float(data.get("probability", 0.0) or 0.0),
        float(data.get("similarity", 0.0) or 0.0),
        sorted((str(key), str(value)) for key, value in data.items()),
    )


def _add_attribute_entities(graph: nx.MultiDiGraph, report: GraphQualityReport) -> None:
    qualifications: dict[str, str] = {}
    job_zones: dict[str, str] = {}
    role_rows = [
        (str(node_id), data)
        for node_id, data in graph.nodes(data=True)
        if data.get("type") == "role" and data.get("source") == "onet"
    ]
    for role_id, data in role_rows:
        qualification = str(data.get("typical_education_level", "")).strip()
        if qualification:
            normalized = normalize_label(qualification)
            node_id = qualifications.setdefault(
                normalized, f"derived:qualification:{_slug(qualification)}"
            )
            if not graph.has_node(node_id):
                graph.add_node(
                    node_id,
                    type="qualification",
                    source="onet",
                    title=qualification,
                    description="Typical education level from O*NET.",
                    normalized_title=normalized,
                )
                report.added_qualification_nodes += 1
            key = "onet:TYPICALLY_REQUIRES_QUALIFICATION"
            if not graph.has_edge(role_id, node_id, key=key):
                graph.add_edge(
                    role_id,
                    node_id,
                    key=key,
                    relation="TYPICALLY_REQUIRES_QUALIFICATION",
                    source="onet",
                    evidence_type="source_attribute",
                )
                report.added_connectivity_edges += 1

        zone_value = data.get("job_zone")
        zone_title = str(data.get("job_zone_title", "")).strip()
        if zone_value is not None and str(zone_value).strip():
            zone_key = str(zone_value).strip()
            node_id = job_zones.setdefault(zone_key, f"derived:job-zone:{_slug(zone_key)}")
            if not graph.has_node(node_id):
                title = zone_title or f"Job Zone {zone_key}"
                graph.add_node(
                    node_id,
                    type="job_zone",
                    source="onet",
                    title=title,
                    description="O*NET preparation and experience category.",
                    zone=zone_key,
                    normalized_title=normalize_label(title),
                )
                report.added_job_zone_nodes += 1
            key = "onet:IN_JOB_ZONE"
            if not graph.has_edge(role_id, node_id, key=key):
                graph.add_edge(
                    role_id,
                    node_id,
                    key=key,
                    relation="IN_JOB_ZONE",
                    source="onet",
                    evidence_type="source_attribute",
                )
                report.added_connectivity_edges += 1


def improve_graph_quality(graph: nx.MultiDiGraph) -> tuple[nx.MultiDiGraph, GraphQualityReport]:
    """Return a cleaned graph and an auditable deterministic quality report."""
    if not isinstance(graph, nx.MultiDiGraph):
        raise TypeError("Graph quality processing requires a MultiDiGraph")
    report = GraphQualityReport(graph.number_of_nodes(), graph.number_of_edges())
    cleaned = nx.MultiDiGraph()
    cleaned.graph.update(graph.graph)

    for node_id, raw_data in graph.nodes(data=True):
        identifier = str(node_id).strip()
        data = dict(raw_data)
        node_type = str(data.get("type", "")).strip()
        title = str(data.get("title", "")).strip()
        if not identifier or node_type not in ALLOWED_NODE_TYPES or not title:
            report.removed_invalid_nodes += 1
            continue
        data["type"] = node_type
        data["title"] = " ".join(title.split())
        data["normalized_title"] = normalize_label(data["title"])
        cleaned.add_node(identifier, **data)

    selected: dict[tuple[str, str, str, str], tuple[str, dict]] = {}
    for source, target, key, raw_data in graph.edges(keys=True, data=True):
        source_id, target_id = str(source), str(target)
        data = dict(raw_data)
        relation = str(data.get("relation", "")).strip().upper()
        edge_source = str(data.get("source", "derived")).strip() or "derived"
        if source_id not in cleaned or target_id not in cleaned or relation not in ALLOWED_RELATIONS:
            report.removed_invalid_edges += 1
            continue
        if source_id == target_id:
            report.removed_self_loops += 1
            continue
        if relation == "TRANSITIONS_TO" and not is_training_transition(data):
            report.removed_held_out_transitions += 1
            continue
        if not _finite_edge_values(data):
            report.removed_invalid_edges += 1
            continue
        data["relation"] = relation
        data["source"] = edge_source
        signature = (source_id, target_id, relation, edge_source)
        previous = selected.get(signature)
        if previous is not None:
            report.removed_duplicate_edges += 1
            if _edge_preference(data) <= _edge_preference(previous[1]):
                continue
        selected[signature] = (str(key), data)

    for (source, target, relation, edge_source), (key, data) in sorted(selected.items()):
        stable_key = key or f"{edge_source}:{relation}"
        cleaned.add_edge(source, target, key=stable_key, **data)

    isolates = list(nx.isolates(cleaned))
    cleaned.remove_nodes_from(isolates)
    report.removed_isolates = len(isolates)
    _add_attribute_entities(cleaned, report)

    collisions: defaultdict[tuple[str, str, str], list[str]] = defaultdict(list)
    for node_id, data in cleaned.nodes(data=True):
        collisions[(data.get("type", ""), data.get("source", ""), data.get("normalized_title", ""))].append(str(node_id))
    report.normalized_label_collisions = sum(
        1 for (_, _, title), identifiers in collisions.items() if title and len(identifiers) > 1
    )
    report.nodes_after = cleaned.number_of_nodes()
    report.edges_after = cleaned.number_of_edges()
    report.node_types = dict(sorted(Counter(data.get("type", "unknown") for _, data in cleaned.nodes(data=True)).items()))
    report.relationship_types = dict(sorted(Counter(data.get("relation", "UNKNOWN") for _, _, data in cleaned.edges(data=True)).items()))
    return cleaned, report
