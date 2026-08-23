"""Hybrid retrieval combining vector search, graph structure, and link prediction.

Fusion strategies:
- Reciprocal Rank Fusion (RRF): combines rankings from multiple sources
- Weighted combination: configurable weights per source

Sources:
1. Vector retrieval (existing ChromaDB cosine similarity)
2. Transition retrieval (direct + smoothed empirical transitions)
3. Link prediction retrieval (LP model predictions for uncovered roles)
4. Graph-structural retrieval (skill overlap with user's owned skills)
"""

from __future__ import annotations

from typing import Sequence

import networkx as nx

from src.kg_enrichment import role_skills
from src.text_normalization import normalize_label


def graph_retrieve(
    owned_skill_ids: set[str],
    G: nx.MultiDiGraph,
    top_k: int = 20,
) -> list[dict]:
    """Retrieve roles by IDF-weighted skill overlap with user's owned skills.

    Returns roles sorted by overlap score (descending).
    """
    if not owned_skill_ids:
        return []

    idf_map = {
        str(nid): float(d.get("idf", 1.0))
        for nid, d in G.nodes(data=True)
        if d.get("type") in ("skill", "element") and "idf" in d
    }

    scored: list[tuple[str, float]] = []
    for nid, data in G.nodes(data=True):
        if data.get("type") != "role":
            continue
        role_skill_set = role_skills(str(nid), G)
        if not role_skill_set:
            continue
        shared = owned_skill_ids & role_skill_set
        if not shared:
            continue
        overlap_score = sum(idf_map.get(s, 1.0) for s in shared) / sum(idf_map.get(s, 1.0) for s in role_skill_set)
        scored.append((str(nid), overlap_score))

    scored.sort(key=lambda x: -x[1])

    results = []
    for role_id, score in scored[:top_k]:
        if not G.has_node(role_id):
            continue
        d = G.nodes[role_id]
        results.append({
            "id": role_id,
            "title": d.get("title", role_id),
            "score": score,
            "source": "graph_structural",
        })

    return results


def reciprocal_rank_fusion(
    *ranked_lists: Sequence[dict],
    k: int = 60,
    id_field: str = "id",
) -> list[dict]:
    """Combine multiple ranked lists using Reciprocal Rank Fusion.

    RRF score = sum(1 / (k + rank_i)) for each list where item appears.
    Higher k smooths the contribution of high ranks.
    """
    rrf_scores: dict[str, float] = {}
    item_data: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            item_id = str(item.get(id_field, ""))
            if not item_id:
                continue
            rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + 1.0 / (k + rank)
            if item_id not in item_data:
                item_data[item_id] = dict(item)

    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: -rrf_scores[x])

    results = []
    for item_id in sorted_ids:
        entry = dict(item_data[item_id])
        entry["rrf_score"] = rrf_scores[item_id]
        results.append(entry)

    return results


def hybrid_retrieve(
    query: str,
    collection,
    owned_skill_ids: set[str],
    current_role_id: str | None,
    G: nx.MultiDiGraph,
    settings,
    transition_smoother=None,
    link_predictor=None,
    vector_top_k: int = 50,
    graph_top_k: int = 20,
    lp_top_k: int = 20,
) -> list[dict]:
    """Hybrid retrieval combining vector, graph-structural, and LP sources.

    Returns fused candidate list via RRF.
    """
    from src.inference_pipeline import retrieve_candidates, filter_candidates_to_graph

    sources: list[list[dict]] = []

    # 1. Vector retrieval (always)
    try:
        vector_candidates = retrieve_candidates(query, collection, settings)
        vector_candidates = filter_candidates_to_graph(vector_candidates, G)
        sources.append(vector_candidates)
    except Exception:
        pass

    # 2. Graph-structural retrieval (if user has skills)
    if owned_skill_ids:
        graph_candidates = graph_retrieve(owned_skill_ids, G, top_k=graph_top_k)
        if graph_candidates:
            sources.append(graph_candidates)

    # 3. Link prediction retrieval (if model available and current role known)
    if link_predictor and current_role_id:
        try:
            from src.link_prediction import predict_transitions, _get_idf
            esco_roles = [
                str(nid) for nid, d in G.nodes(data=True)
                if d.get("type") == "role" and d.get("source") == "esco" and str(nid) != current_role_id
            ]
            idf_map = _get_idf(G)
            lp_results = predict_transitions(
                link_predictor, current_role_id, esco_roles[:500],
                G, idf_map,
            )
            lp_candidates = [
                {"id": tid, "title": G.nodes[tid].get("title", tid), "score": score, "source": "link_prediction"}
                for tid, score in lp_results[:lp_top_k]
            ]
            if lp_candidates:
                sources.append(lp_candidates)
        except Exception:
            pass

    if not sources:
        return []

    # Fuse via RRF
    fused = reciprocal_rank_fusion(*sources)
    return fused


def combine_prediction_maps(
    direct_map: dict[str, list[str]],
    smoothed_map: dict[str, list[str]] | None = None,
    lp_map: dict[str, list[str]] | None = None,
    top_k: int = 50,
) -> dict[str, list[str]]:
    """Combine prediction maps using priority-based fallback.

    Strategy: Use the BEST available signal for each source role:
    1. If direct edges exist → use direct ranking (strongest signal)
    2. Elif smoothed exists → use smoothed ranking
    3. Elif LP exists → use LP ranking (coverage fallback)

    This maximizes both precision (for covered roles) and coverage (for uncovered roles).
    """
    all_sources = set(direct_map.keys())
    if smoothed_map:
        all_sources |= set(smoothed_map.keys())
    if lp_map:
        all_sources |= set(lp_map.keys())

    combined: dict[str, list[str]] = {}
    for source in all_sources:
        if source in direct_map and direct_map[source]:
            combined[source] = direct_map[source][:top_k]
        elif smoothed_map and source in smoothed_map and smoothed_map[source]:
            combined[source] = smoothed_map[source][:top_k]
        elif lp_map and source in lp_map and lp_map[source]:
            combined[source] = lp_map[source][:top_k]

    return combined
