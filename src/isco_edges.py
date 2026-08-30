"""ISCO group structural edges for the knowledge graph.

Adds SAME_ISCO_GROUP edges between roles that share a 2-digit ISCO code.
These edges enrich graph traversal and improve domain_distance() for ONET
roles (which otherwise return the 0.5 neutral sentinel due to missing ISCO).

Design constraints:
- Capped at top-20 roles per group by degree to avoid dense cliques.
- Bidirectional (one edge in each direction).
- Excluded from LP training via transition_policy.py (already: only
  TRANSITIONS_TO edges from the training split are used as positives).
- Never written into the graph as observed transitions.
"""

from __future__ import annotations

import networkx as nx

SAME_ISCO_RELATION = "SAME_ISCO_GROUP"
MAX_PER_GROUP = 20


def add_isco_group_edges(G: nx.MultiDiGraph, max_per_group: int = MAX_PER_GROUP) -> int:
    """Add SAME_ISCO_GROUP edges between roles sharing a 2-digit ISCO code.

    Groups are capped at `max_per_group` roles sorted by descending degree so
    only the most-connected (most representative) roles get linked.

    Returns: number of directed edges added (each pair generates 2 edges).
    """
    # Collect roles by ISCO 2-digit code
    groups: dict[str, list[str]] = {}
    for nid, data in G.nodes(data=True):
        if data.get("type") != "role":
            continue
        code = str(data.get("isco_2digit", "")).strip()
        if not code or len(code) < 2:
            continue
        groups.setdefault(code, []).append(str(nid))

    added = 0
    for code, members in groups.items():
        if len(members) < 2:
            continue

        # Cap to top-max_per_group by total degree (highest connectivity first)
        if len(members) > max_per_group:
            members = sorted(members, key=lambda n: G.degree(n), reverse=True)[:max_per_group]

        # Add bidirectional edges for all pairs in the capped group
        for i, src in enumerate(members):
            for dst in members[i + 1:]:
                # Skip if a SAME_ISCO_GROUP edge already exists in this direction
                exists = any(
                    ed.get("relation") == SAME_ISCO_RELATION
                    for _, _, ed in G.out_edges(src, data=True)
                    if _ == src  # noqa: SIM118 — MultiDiGraph out_edges returns (u, v, data)
                )
                # Simpler check: iterate outgoing edges
                already_fwd = any(
                    v == dst and ed.get("relation") == SAME_ISCO_RELATION
                    for _, v, ed in G.out_edges(src, data=True)
                )
                already_bwd = any(
                    v == src and ed.get("relation") == SAME_ISCO_RELATION
                    for _, v, ed in G.out_edges(dst, data=True)
                )
                if not already_fwd:
                    G.add_edge(src, dst, relation=SAME_ISCO_RELATION, isco_2digit=code)
                    added += 1
                if not already_bwd:
                    G.add_edge(dst, src, relation=SAME_ISCO_RELATION, isco_2digit=code)
                    added += 1

    return added


def isco_group_stats(G: nx.MultiDiGraph) -> dict[str, int]:
    """Return {isco_2digit: member_count} for diagnostic purposes."""
    groups: dict[str, int] = {}
    for _, data in G.nodes(data=True):
        if data.get("type") != "role":
            continue
        code = str(data.get("isco_2digit", "")).strip()
        if code:
            groups[code] = groups.get(code, 0) + 1
    return groups
