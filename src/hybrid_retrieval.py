"""Hybrid role retrieval for the local GraphRAG runtime.

The runtime fuses five independently useful signals: semantic vector search,
direct training transitions, embedding-smoothed transitions, graph skill
overlap, and model-predicted missing transitions. Predicted links remain
virtual evidence and are never written into the knowledge graph.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import networkx as nx
import numpy as np

from src.skill_gap import role_requirements


_TRANSITION_PRIORITY = {
    "predicted_transition": 1,
    "semantic_transition_backoff": 2,
    "sequential_transition": 2,
    "direct_transition": 3,
}


def _model_feature_count(model: object, available: int) -> int:
    """Return the trained model width without bypassing shape validation."""
    count = None
    num_feature = getattr(model, "num_feature", None)
    if callable(num_feature):
        count = num_feature()
    elif getattr(model, "n_features_in_", None) is not None:
        count = getattr(model, "n_features_in_")
    if count is None:
        return available
    try:
        count = int(count)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Link-prediction model has an invalid feature schema") from exc
    if count < 1 or count > available:
        raise RuntimeError(
            f"Link-prediction model expects {count} features, but runtime provides {available}"
        )
    return count


def _role_candidate(role_id: str, G: nx.MultiDiGraph, **extra: Any) -> dict:
    """Create a reranker-compatible role candidate from a graph node."""
    from src.embeddings_index import build_role_text

    data = G.nodes[role_id]
    return {
        "id": role_id,
        "metadata": {
            "title": data.get("title", role_id),
            "source": data.get("source", "esco"),
            "type": data.get("type", "role"),
        },
        "document": build_role_text(role_id, data),
        **extra,
    }


def graph_retrieve(
    owned_skill_ids: set[str],
    G: nx.MultiDiGraph,
    top_k: int = 20,
    onet_importance_threshold: float = 3.5,
) -> list[dict]:
    """Retrieve roles by IDF-weighted overlap with role-relevant skills."""
    if not owned_skill_ids or top_k <= 0:
        return []

    idf_map = {
        str(nid): float(data.get("idf", 1.0))
        for nid, data in G.nodes(data=True)
        if data.get("type") in ("skill", "element")
    }

    scored: list[tuple[str, float]] = []
    for node_id, data in G.nodes(data=True):
        if data.get("type") != "role":
            continue
        role_id = str(node_id)
        required, _, _ = role_requirements(
            role_id,
            G,
            onet_importance_threshold=onet_importance_threshold,
        )
        if not required:
            continue
        shared = owned_skill_ids & required
        if not shared:
            continue
        denominator = sum(idf_map.get(skill_id, 1.0) for skill_id in required)
        if denominator <= 0.0:
            continue
        score = sum(idf_map.get(skill_id, 1.0) for skill_id in shared) / denominator
        scored.append((role_id, score))

    scored.sort(key=lambda item: (-item[1], item[0]))
    return [
        _role_candidate(
            role_id,
            G,
            score=score,
            source="graph_structural",
            retrieval_sources=["graph_structural"],
        )
        for role_id, score in scored[:top_k]
        if G.has_node(role_id)
    ]


@dataclass
class LinkPredictionRuntime:
    """Precomputed, read-only feature context for request-time predictions."""

    model: object
    graph: nx.MultiDiGraph
    embeddings: dict[str, np.ndarray]
    transition_index: dict[str, dict[str, float]]
    neighbour_index: dict[str, list[str]]
    idf_map: dict[str, float]
    candidate_role_ids: tuple[str, ...]
    skills_cache: dict[str, set[str]]
    skill_degrees: dict[str, int]
    isco_cache: dict[str, str]
    vector_report: dict[str, int]
    prediction_lock: threading.RLock = field(
        default_factory=threading.RLock,
        repr=False,
    )

    def predict(self, source_id: str | None, top_k: int = 20) -> list[tuple[str, float]]:
        """Rank only missing ESCO transition edges for one live source role."""
        from src.link_prediction import FEATURE_NAMES, extract_pair_features_fast

        if (
            not source_id
            or top_k <= 0
            or source_id not in self.embeddings
            or not self.graph.has_node(source_id)
        ):
            return []

        observed = set(self.transition_index.get(source_id, {}))
        candidates = [
            role_id
            for role_id in self.candidate_role_ids
            if role_id != source_id and role_id not in observed
        ]
        if not candidates:
            return []

        source_skills = self.skills_cache.get(source_id, set())
        source_isco = self.isco_cache.get(source_id, "")
        features = np.zeros((len(candidates), len(FEATURE_NAMES)), dtype=np.float64)
        for index, target_id in enumerate(candidates):
            features[index] = extract_pair_features_fast(
                source_id,
                target_id,
                source_skills,
                self.skills_cache.get(target_id, set()),
                self.idf_map,
                self.skill_degrees,
                self.embeddings,
                self.transition_index,
                self.neighbour_index,
                source_isco,
                self.isco_cache.get(target_id, ""),
            )

        trained_feature_count = _model_feature_count(self.model, len(FEATURE_NAMES))
        model_features = features[:, :trained_feature_count]
        with self.prediction_lock:
            scores = np.asarray(self.model.predict(model_features), dtype=np.float64)
        count = min(top_k, len(candidates))
        top_indices = np.argsort(scores, kind="stable")[-count:][::-1]
        return [(candidates[index], float(scores[index])) for index in top_indices]

    def diagnostics(self) -> dict[str, Any]:
        return {
            **self.vector_report,
            "candidate_roles": len(self.candidate_role_ids),
            "transition_sources": len(self.transition_index),
            "neighbour_queries": len(self.neighbour_index),
        }


def build_link_prediction_runtime(
    model_path: Path,
    G: nx.MultiDiGraph,
    collection,
    neighbours: int = 10,
) -> LinkPredictionRuntime:
    """Load the trained model and all feature inputs without mutating storage."""
    from src.link_prediction import (
        _get_idf,
        _precompute_role_skills_cache,
        _precompute_skill_degrees,
        build_transition_index,
        load_model,
    )
    from src.transition_embedding import compute_neighbour_index, load_live_esco_embeddings

    model_path = Path(model_path)
    if not model_path.is_file() and model_path.suffix.lower() in {".pkl", ".pickle"}:
        portable_path = model_path.with_suffix(".json")
        if portable_path.is_file():
            model_path = portable_path
    if not model_path.is_file():
        raise RuntimeError(f"Link-prediction model not found: {model_path}")

    embeddings, vector_report = load_live_esco_embeddings(collection, G)
    if not embeddings:
        raise RuntimeError("No live ESCO embeddings are available for link prediction.")

    transition_index = build_transition_index(G)
    transition_sources = sorted(set(transition_index) & set(embeddings))
    raw_neighbours = compute_neighbour_index(
        embeddings,
        transition_sources,
        max_neighbours=neighbours,
        query_ids=embeddings,
    )
    neighbour_index = {
        source_id: [role_id for role_id, _ in rows]
        for source_id, rows in raw_neighbours.items()
    }
    return LinkPredictionRuntime(
        model=load_model(model_path),
        graph=G,
        embeddings=embeddings,
        transition_index=transition_index,
        neighbour_index=neighbour_index,
        idf_map=_get_idf(G),
        candidate_role_ids=tuple(sorted(embeddings)),
        skills_cache=_precompute_role_skills_cache(G),
        skill_degrees=_precompute_skill_degrees(G),
        isco_cache={
            str(node_id): str(data.get("isco_2digit", ""))
            for node_id, data in G.nodes(data=True)
            if data.get("type") == "role"
        },
        vector_report=vector_report,
    )


def reciprocal_rank_fusion(
    *ranked_lists: Sequence[dict],
    k: int = 60,
    id_field: str = "id",
) -> list[dict]:
    """Fuse rankings while retaining every source and strongest evidence."""
    rrf_scores: dict[str, float] = {}
    item_data: dict[str, dict] = {}
    source_sets: dict[str, set[str]] = {}

    for ranked_list in ranked_lists:
        for rank, raw_item in enumerate(ranked_list, start=1):
            item = dict(raw_item)
            item_id = str(item.get(id_field, ""))
            if not item_id:
                continue
            rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + 1.0 / (k + rank)
            sources = item.get("retrieval_sources") or [item.get("source", "unknown")]
            source_sets.setdefault(item_id, set()).update(str(source) for source in sources if source)

            if item_id not in item_data:
                item_data[item_id] = item
                continue

            existing = item_data[item_id]
            for field in ("metadata", "document", "title"):
                if not existing.get(field) and item.get(field):
                    existing[field] = item[field]
            incoming_transition = item.get("transition") or {}
            existing_transition = existing.get("transition") or {}
            incoming_priority = _TRANSITION_PRIORITY.get(incoming_transition.get("evidence_type", ""), 0)
            existing_priority = _TRANSITION_PRIORITY.get(existing_transition.get("evidence_type", ""), 0)
            if incoming_priority > existing_priority:
                existing["transition"] = incoming_transition

    sorted_ids = sorted(rrf_scores, key=lambda item_id: (-rrf_scores[item_id], item_id))
    results = []
    for item_id in sorted_ids:
        entry = dict(item_data[item_id])
        entry["source_score"] = float(entry.get("score", 0.0) or 0.0)
        entry["rrf_score"] = rrf_scores[item_id]
        entry["score"] = rrf_scores[item_id]
        entry["retrieval_sources"] = sorted(source_sets.get(item_id, set()))
        results.append(entry)
    return results


def fuse_transition_candidates(
    transition_candidates: Sequence[dict],
    sequential_predictions: Sequence[object],
    sequential_weight: float,
    G: nx.MultiDiGraph,
    limit: int,
) -> list[dict]:
    """Fuse smoother and sequential rankings before outer retrieval fusion.

    Scores are rank-based so the two models need not share calibrated numeric
    ranges. Existing direct-transition evidence remains the explanation when a
    role occurs in both lists.
    """
    if limit <= 0:
        return []
    sequential_weight = min(1.0, max(0.0, float(sequential_weight)))
    smoother_weight = 1.0 - sequential_weight
    by_id: dict[str, dict] = {str(item["id"]): dict(item) for item in transition_candidates}
    scores: dict[str, float] = {}
    source_sets: dict[str, set[str]] = {}

    for rank, item in enumerate(transition_candidates, start=1):
        role_id = str(item["id"])
        scores[role_id] = scores.get(role_id, 0.0) + smoother_weight / (60 + rank)
        source_sets.setdefault(role_id, set()).update(item.get("retrieval_sources", []))

    for rank, prediction in enumerate(sequential_predictions, start=1):
        role_id = str(getattr(prediction, "role_id", ""))
        if not role_id or not G.has_node(role_id):
            continue
        scores[role_id] = scores.get(role_id, 0.0) + sequential_weight / (60 + rank)
        source_sets.setdefault(role_id, set()).add("sequential_transition")
        if role_id not in by_id:
            by_id[role_id] = _role_candidate(
                role_id,
                G,
                score=float(getattr(prediction, "score", 0.0)),
                source="sequential_transition",
                retrieval_sources=["sequential_transition"],
                transition={
                    "from_role_id": None,
                    "evidence_type": "sequential_transition",
                    "score": float(getattr(prediction, "score", 0.0)),
                    "model": str(getattr(prediction, "model", "sequential")),
                    "history_length": int(getattr(prediction, "history_length", 0)),
                },
            )

    ranked_ids = sorted(scores, key=lambda role_id: (-scores[role_id], role_id))[:limit]
    result: list[dict] = []
    for role_id in ranked_ids:
        item = dict(by_id[role_id])
        item["score"] = scores[role_id]
        item["source"] = "transition_hybrid"
        item["retrieval_sources"] = sorted(source_sets.get(role_id, set()))
        result.append(item)
    return result


def hybrid_retrieve(
    query: str,
    collection,
    owned_skill_ids: set[str],
    current_role_id: str | None,
    G: nx.MultiDiGraph,
    settings,
    transition_smoother=None,
    link_prediction_runtime: LinkPredictionRuntime | None = None,
    sequential_runtime=None,
    history_role_ids: Sequence[str] | None = None,
    vector_top_k: int | None = None,
    graph_top_k: int = 20,
    lp_top_k: int = 20,
) -> list[dict]:
    """Return a single deterministic candidate list from all live signals."""
    from src.inference_pipeline import (
        augment_candidates_with_transitions,
        filter_candidates_to_graph,
        retrieve_candidates,
    )

    sources: list[list[dict]] = []
    vector_limit = vector_top_k or settings.retrieval_top_k

    try:
        vector_candidates = filter_candidates_to_graph(
            retrieve_candidates(query, collection, settings, limit=vector_limit),
            G,
        )
        vector_candidates = [
            candidate
            for candidate in vector_candidates
            if str(candidate.get("id", "")) != str(current_role_id or "")
        ]
        for candidate in vector_candidates:
            candidate["source"] = "vector"
            candidate["retrieval_sources"] = ["vector"]
        if vector_candidates:
            sources.append(vector_candidates)
    except RuntimeError:
        pass

    transition_candidates: list[dict] = []
    if current_role_id:
        smoothed = None
        if transition_smoother is not None:
            try:
                smoothed = transition_smoother.rank(
                    current_role_id,
                    settings.transition_candidate_limit,
                )
            except Exception:
                smoothed = None
        transition_candidates = augment_candidates_with_transitions(
            [],
            current_role_id,
            G,
            limit=settings.transition_candidate_limit,
            smoothed_destinations=smoothed,
        )
        for candidate in transition_candidates:
            evidence_type = (candidate.get("transition") or {}).get("evidence_type", "direct_transition")
            source = "direct_transition" if evidence_type == "direct_transition" else "transition_smoothing"
            candidate["source"] = source
            candidate["retrieval_sources"] = [source]
            candidate["score"] = float(
                (candidate.get("transition") or {}).get(
                    "probability", (candidate.get("transition") or {}).get("score", 0.0)
                )
            )
        if sequential_runtime is not None and history_role_ids:
            try:
                sequential_predictions = sequential_runtime.rank(
                    list(history_role_ids),
                    max(settings.transition_candidate_limit * 4, 50),
                )
                if sequential_predictions:
                    transition_candidates = fuse_transition_candidates(
                        transition_candidates,
                        sequential_predictions,
                        sequential_runtime.fusion_weight(len(history_role_ids)),
                        G,
                        settings.transition_candidate_limit,
                    )
            except Exception:
                # A promoted artifact is optional; its failure must not remove
                # accepted direct/semantic recommendations.
                pass
        if transition_candidates:
            sources.append(transition_candidates)

    graph_candidates = graph_retrieve(
        owned_skill_ids,
        G,
        top_k=graph_top_k,
        onet_importance_threshold=settings.onet_importance_threshold,
    )
    graph_candidates = [
        candidate
        for candidate in graph_candidates
        if str(candidate.get("id", "")) != str(current_role_id or "")
    ]
    if graph_candidates:
        sources.append(graph_candidates)

    lp_candidates: list[dict] = []
    if link_prediction_runtime is not None and current_role_id:
        predictions = link_prediction_runtime.predict(current_role_id, top_k=lp_top_k)
        lp_candidates = [
            _role_candidate(
                role_id,
                G,
                score=score,
                source="link_prediction",
                retrieval_sources=["link_prediction"],
                transition={
                    "from_role_id": current_role_id,
                    "evidence_type": "predicted_transition",
                    "score": score,
                    "model": "lightgbm",
                },
            )
            for role_id, score in predictions
            if G.has_node(role_id)
        ]
    # LP is a coverage backfill, not an equal-vote rank source: held-out
    # validation shows smoothing is the stronger top-K signal. Existing
    # candidates still retain LP provenance without receiving an RRF boost.
    fused = reciprocal_rank_fusion(*sources) if sources else []
    fused_by_id = {str(candidate["id"]): candidate for candidate in fused}
    for index, candidate in enumerate(lp_candidates, start=1):
        role_id = str(candidate["id"])
        existing = fused_by_id.get(role_id)
        if existing is not None:
            existing["retrieval_sources"] = sorted(
                set(existing.get("retrieval_sources", [])) | {"link_prediction"}
            )
            if not existing.get("transition"):
                existing["transition"] = candidate["transition"]
            continue
        backfill = dict(candidate)
        backfill["source_score"] = float(candidate.get("score", 0.0))
        backfill["rrf_score"] = 0.0 if sources else 1.0 / (60 + index)
        backfill["score"] = backfill["rrf_score"]
        fused.append(backfill)
        fused_by_id[role_id] = backfill
    return fused


def combine_prediction_maps(
    direct_map: dict[str, list[str]],
    smoothed_map: dict[str, list[str]] | None = None,
    lp_map: dict[str, list[str]] | None = None,
    top_k: int = 50,
) -> dict[str, list[str]]:
    """Use validation-accepted smoothing, then direct, then LP fallback."""
    all_sources = set(direct_map)
    if smoothed_map:
        all_sources |= set(smoothed_map)
    if lp_map:
        all_sources |= set(lp_map)

    combined: dict[str, list[str]] = {}
    for source in all_sources:
        if smoothed_map and smoothed_map.get(source):
            # The smoothed rank already includes direct transition evidence.
            # Preserve it in full so evaluation MRR is not lowered by a
            # post-hoc truncation.
            combined[source] = list(smoothed_map[source])
        elif direct_map.get(source):
            combined[source] = list(direct_map[source])
        elif lp_map and lp_map.get(source):
            combined[source] = lp_map[source][:top_k]
    return combined
