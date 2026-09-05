"""Embedding-smoothed ranking for train-derived Karrierewege transitions.

This module consumes existing taxonomy role vectors and aggregate graph edges.
It never embeds raw Karrierewege rows, calls an external API, or reads held-out
transition data.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import networkx as nx
import numpy as np

from src.text_normalization import normalize_label
from src.transition_policy import (
    TRANSITION_RELATION,
    TRANSITION_SOURCE,
    is_training_transition,
)


@dataclass(frozen=True)
class SmoothingConfig:
    neighbours: int
    direct_weight: float
    temperature: float

    def __post_init__(self) -> None:
        if self.neighbours < 1:
            raise ValueError("neighbours must be at least 1")
        if not 0.0 <= self.direct_weight <= 1.0:
            raise ValueError("direct_weight must be between 0 and 1")
        if self.temperature <= 0.0 or not math.isfinite(self.temperature):
            raise ValueError("temperature must be a positive finite number")

    def to_dict(self) -> dict[str, int | float]:
        return {
            "neighbours": self.neighbours,
            "direct_weight": self.direct_weight,
            "temperature": self.temperature,
        }


@dataclass(frozen=True)
class TransitionValue:
    probability: float
    count: int
    source_total: int


@dataclass(frozen=True)
class DestinationScore:
    role_id: str
    score: float
    direct_probability: float
    direct_count: int
    neighbour_probability: float
    neighbour_support: int

    @property
    def evidence_type(self) -> str:
        return "direct_transition" if self.direct_probability > 0.0 else "semantic_transition_backoff"

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "id": self.role_id,
            "score": self.score,
            "direct_probability": self.direct_probability,
            "direct_count": self.direct_count,
            "neighbour_probability": self.neighbour_probability,
            "neighbour_support": self.neighbour_support,
            "evidence_type": self.evidence_type,
        }


def transition_distributions(
    graph: nx.MultiDiGraph,
) -> dict[str, dict[str, TransitionValue]]:
    """Extract retained, training-only transition distributions by node ID."""
    if not graph.is_multigraph():
        raise TypeError("Embedding-smoothed transitions require a MultiDiGraph.")

    distributions: dict[str, dict[str, TransitionValue]] = {}
    for source, target, _, data in graph.edges(keys=True, data=True):
        if data.get("relation") != TRANSITION_RELATION or data.get("source") != TRANSITION_SOURCE:
            continue
        if not is_training_transition(data):
            continue

        source_id = str(source)
        target_id = str(target)
        probability = float(data.get("probability", 0.0))
        count = int(data.get("count", 0))
        source_total = int(data.get("source_total", 0))
        if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
            raise ValueError(f"Invalid transition probability: {source_id} -> {target_id}")
        if count < 0 or source_total < 0:
            raise ValueError(f"Invalid transition support: {source_id} -> {target_id}")
        distributions.setdefault(source_id, {})[target_id] = TransitionValue(
            probability=probability,
            count=count,
            source_total=source_total,
        )
    return distributions


def load_live_esco_embeddings(
    collection,
    graph: nx.MultiDiGraph,
    batch_size: int = 256,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Read and validate live ESCO vectors through the vector-store adapter."""
    live_ids = sorted(
        str(node_id)
        for node_id, data in graph.nodes(data=True)
        if data.get("type") == "role" and data.get("source") == "esco"
    )
    embeddings, base_report = load_normalized_embeddings(
        collection,
        live_ids,
        batch_size=batch_size,
    )
    report = {
        "live_esco_roles": len(live_ids),
        **base_report,
    }
    return embeddings, report


def load_normalized_embeddings(
    collection,
    role_ids: Iterable[str],
    batch_size: int = 256,
    expected_dimension: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Load normalized vectors for explicit IDs without mutating the collection."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    requested_ids = sorted({str(role_id) for role_id in role_ids})
    embeddings: dict[str, np.ndarray] = {}
    rejected = 0
    missing = 0

    for offset in range(0, len(requested_ids), batch_size):
        requested = requested_ids[offset : offset + batch_size]
        result = collection.get(ids=requested, include=["embeddings"])
        returned_ids = [str(item) for item in (result.get("ids") or [])]
        returned_vectors = result.get("embeddings")
        if returned_vectors is None:
            returned_vectors = []
        found = set(returned_ids)
        missing += len(set(requested) - found)

        for role_id, raw_vector in zip(returned_ids, returned_vectors):
            vector = np.asarray(raw_vector, dtype=np.float32)
            if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
                rejected += 1
                continue
            if expected_dimension is None:
                expected_dimension = int(vector.size)
            if vector.size != expected_dimension:
                rejected += 1
                continue
            norm = float(np.linalg.norm(vector))
            if not math.isfinite(norm) or norm <= 1e-12:
                rejected += 1
                continue
            embeddings[role_id] = vector / norm

    report = {
        "loaded_vectors": len(embeddings),
        "missing_vectors": missing,
        "rejected_vectors": rejected,
        "dimension": expected_dimension or 0,
    }
    return embeddings, report


class RuntimeTransitionSmoother:
    """Lazy-process runtime scorer backed by existing stored role vectors."""

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        collection,
        config: SmoothingConfig,
    ) -> None:
        self.graph = graph
        self.collection = collection
        self.config = config
        self.distributions = transition_distributions(graph)
        source_vectors, vector_report = load_normalized_embeddings(
            collection,
            self.distributions,
        )
        if not source_vectors:
            raise RuntimeError("No transition-source role vectors are available.")
        self.vector_report = vector_report
        self._candidate_ids = sorted(source_vectors)
        self._candidate_matrix = np.stack(
            [source_vectors[role_id] for role_id in self._candidate_ids]
        ).astype(np.float32)
        self._dimension = int(self._candidate_matrix.shape[1])
        self._vectors = dict(source_vectors)
        self._ranking_cache: dict[str, tuple[DestinationScore, ...]] = {}
        self._lock = threading.RLock()

    def _source_vector(self, source_id: str) -> np.ndarray | None:
        with self._lock:
            cached = self._vectors.get(source_id)
            if cached is not None:
                return cached
        loaded, _ = load_normalized_embeddings(
            self.collection,
            [source_id],
            expected_dimension=self._dimension,
        )
        vector = loaded.get(source_id)
        if vector is not None:
            with self._lock:
                self._vectors[source_id] = vector
        return vector

    def rank(self, source_id: str | None, limit: int) -> list[DestinationScore]:
        """Return cached hybrid destinations or an empty direct-safe fallback."""
        if not source_id or limit <= 0 or not self.graph.has_node(str(source_id)):
            return []
        source_id = str(source_id)
        with self._lock:
            cached = self._ranking_cache.get(source_id)
        if cached is None:
            source_vector = self._source_vector(source_id)
            if source_vector is None:
                return []
            similarities = self._candidate_matrix @ source_vector
            neighbours = [
                (candidate_id, float(similarities[index]))
                for index, candidate_id in enumerate(self._candidate_ids)
                if candidate_id != source_id
            ]
            neighbours.sort(key=lambda item: (-item[1], item[0]))
            neighbour_index = {source_id: neighbours[: self.config.neighbours]}
            ranked = rank_hybrid_destinations(
                source_id,
                self.distributions,
                neighbour_index,
                self.config,
                graph=self.graph,
            )
            cached = tuple(ranked)
            with self._lock:
                self._ranking_cache[source_id] = cached
        return list(cached[:limit])

    def diagnostics(self) -> dict[str, int | float]:
        return {
            **self.config.to_dict(),
            **self.vector_report,
            "transition_sources": len(self.distributions),
            "cached_roles": len(self._ranking_cache),
        }


def compute_neighbour_index(
    embeddings: Mapping[str, np.ndarray],
    eligible_neighbour_ids: Iterable[str],
    max_neighbours: int,
    query_ids: Iterable[str] | None = None,
    block_size: int = 128,
) -> dict[str, list[tuple[str, float]]]:
    """Return deterministic cosine-nearest transition-source roles."""
    if max_neighbours < 1:
        raise ValueError("max_neighbours must be at least 1")
    if block_size < 1:
        raise ValueError("block_size must be at least 1")

    candidate_ids = sorted({str(item) for item in eligible_neighbour_ids if str(item) in embeddings})
    selected_query_ids = sorted(
        {str(item) for item in (query_ids if query_ids is not None else embeddings) if str(item) in embeddings}
    )
    if not candidate_ids or not selected_query_ids:
        return {}

    dimensions = {int(np.asarray(embeddings[role_id]).size) for role_id in candidate_ids + selected_query_ids}
    if len(dimensions) != 1:
        raise ValueError("All role embeddings must have the same dimension.")

    candidate_matrix = np.stack([embeddings[role_id] for role_id in candidate_ids]).astype(np.float32)
    neighbours: dict[str, list[tuple[str, float]]] = {}

    for offset in range(0, len(selected_query_ids), block_size):
        block_ids = selected_query_ids[offset : offset + block_size]
        query_matrix = np.stack([embeddings[role_id] for role_id in block_ids]).astype(np.float32)
        similarities = query_matrix @ candidate_matrix.T
        for row_index, query_id in enumerate(block_ids):
            rows = [
                (candidate_id, float(similarities[row_index, candidate_index]))
                for candidate_index, candidate_id in enumerate(candidate_ids)
                if candidate_id != query_id
            ]
            rows.sort(key=lambda item: (-item[1], item[0]))
            neighbours[query_id] = rows[:max_neighbours]
    return neighbours


def _softmax_weights(similarities: Sequence[float], temperature: float) -> list[float]:
    if not similarities:
        return []
    scaled = np.asarray(similarities, dtype=np.float64) / temperature
    scaled -= float(scaled.max())
    weights = np.exp(scaled)
    denominator = float(weights.sum())
    if not math.isfinite(denominator) or denominator <= 0.0:
        return [1.0 / len(similarities)] * len(similarities)
    return [float(value / denominator) for value in weights]


def rank_hybrid_destinations(
    source_id: str,
    distributions: Mapping[str, Mapping[str, TransitionValue]],
    neighbour_index: Mapping[str, Sequence[tuple[str, float]]],
    config: SmoothingConfig,
    graph: nx.MultiDiGraph | None = None,
) -> list[DestinationScore]:
    """Rank direct and semantic-neighbour destinations for one source role."""
    source_id = str(source_id)
    direct = distributions.get(source_id, {})
    selected_neighbours = list(neighbour_index.get(source_id, ()))[: config.neighbours]
    selected_neighbours = [item for item in selected_neighbours if distributions.get(item[0])]
    weights = _softmax_weights([similarity for _, similarity in selected_neighbours], config.temperature)

    neighbour_probability: dict[str, float] = {}
    neighbour_support: dict[str, int] = {}
    for (neighbour_id, _), weight in zip(selected_neighbours, weights):
        for destination_id, value in distributions[neighbour_id].items():
            if destination_id == source_id:
                continue
            neighbour_probability[destination_id] = (
                neighbour_probability.get(destination_id, 0.0) + weight * value.probability
            )
            neighbour_support[destination_id] = neighbour_support.get(destination_id, 0) + 1

    has_direct = bool(direct)
    has_neighbour = bool(neighbour_probability)
    if has_direct and has_neighbour:
        direct_weight = config.direct_weight
        neighbour_weight = 1.0 - config.direct_weight
    elif has_direct:
        direct_weight, neighbour_weight = 1.0, 0.0
    elif has_neighbour:
        direct_weight, neighbour_weight = 0.0, 1.0
    else:
        return []

    candidates = (set(direct) | set(neighbour_probability)) - {source_id}
    scored: list[DestinationScore] = []
    for destination_id in candidates:
        direct_value = direct.get(destination_id, TransitionValue(0.0, 0, 0))
        neighbour_value = neighbour_probability.get(destination_id, 0.0)
        score = direct_weight * direct_value.probability + neighbour_weight * neighbour_value
        if score <= 0.0 or not math.isfinite(score):
            continue
        scored.append(
            DestinationScore(
                role_id=destination_id,
                score=score,
                direct_probability=direct_value.probability,
                direct_count=direct_value.count,
                neighbour_probability=neighbour_value,
                neighbour_support=neighbour_support.get(destination_id, 0),
            )
        )

    def destination_key(item: DestinationScore) -> tuple:
        title = ""
        if graph is not None and graph.has_node(item.role_id):
            title = normalize_label(graph.nodes[item.role_id].get("title", ""))
        return (
            -item.score,
            -item.direct_probability,
            -item.direct_count,
            -item.neighbour_support,
            title,
            item.role_id,
        )

    scored.sort(key=destination_key)
    return scored


def hybrid_prediction_map(
    graph: nx.MultiDiGraph,
    distributions: Mapping[str, Mapping[str, TransitionValue]],
    neighbour_index: Mapping[str, Sequence[tuple[str, float]]],
    config: SmoothingConfig,
    source_ids: Iterable[str] | None = None,
) -> dict[str, list[str]]:
    """Build normalized title predictions for held-out metric evaluation."""
    selected_source_ids = (
        {str(node_id) for node_id in source_ids}
        if source_ids is not None
        else None
    )
    selected_sources = sorted(
        str(node_id)
        for node_id, data in graph.nodes(data=True)
        if data.get("type") == "role"
        and data.get("source") == "esco"
        and (selected_source_ids is None or str(node_id) in selected_source_ids)
    )
    predictions: dict[str, list[str]] = {}
    for source_id in selected_sources:
        source_title = normalize_label(graph.nodes[source_id].get("title", ""))
        if not source_title:
            continue
        ranked = rank_hybrid_destinations(
            source_id,
            distributions,
            neighbour_index,
            config,
            graph=graph,
        )
        destinations = [
            normalize_label(graph.nodes[item.role_id].get("title", ""))
            for item in ranked
            if graph.has_node(item.role_id)
        ]
        predictions[source_title] = [destination for destination in destinations if destination]
    return predictions
