"""Leakage-safe, source-grouped learning-to-rank utilities.

The module deliberately separates candidate generation from fold labels.  A
held-out fold may label candidates produced from the other folds, but it may
never add a missing positive to the candidate set.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import networkx as nx
import numpy as np

from src.karrierewege_preprocessing import (
    SplitQualityReport,
    _clean_person_steps,
    _iter_person_runs,
)
from src.transition_embedding import _softmax_weights
from src.transition_walk import transition_walk_components


FEATURE_NAMES = (
    "direct_probability",
    "direct_count",
    "log_direct_count",
    "log_source_total",
    "direct_shrunk_probability",
    "support_ge_2",
    "support_ge_3",
    "support_ge_5",
    "semantic_probability",
    "semantic_max_similarity",
    "semantic_support",
    "walk_2_probability",
    "walk_3_probability",
    "embedding_cosine",
    "embedding_missing",
    "same_isco_2digit",
    "isco_distance",
    "skill_jaccard",
    "idf_skill_coverage",
    "skill_gap_ratio",
    "destination_popularity",
    "log_destination_count",
    "candidate_direct",
    "candidate_semantic",
    "candidate_walk",
    "candidate_embedding",
    "candidate_skill",
    "candidate_popular",
    "jobhop_probability",
    "jobhop_count",
    "log_jobhop_count",
    "jobhop_supported",
    "candidate_jobhop",
)


@dataclass(frozen=True)
class FoldedTransitionCounts:
    """Person-disjoint transition counts for the original training split."""

    fold_pair_counts: tuple[Counter[tuple[str, str]], ...]
    fold_source_totals: tuple[Counter[str], ...]
    report: SplitQualityReport

    @property
    def n_folds(self) -> int:
        return len(self.fold_pair_counts)

    @property
    def pair_counts(self) -> Counter[tuple[str, str]]:
        result: Counter[tuple[str, str]] = Counter()
        for counts in self.fold_pair_counts:
            result.update(counts)
        return result

    @property
    def source_totals(self) -> Counter[str]:
        result: Counter[str] = Counter()
        for counts in self.fold_source_totals:
            result.update(counts)
        return result

    def training_without(self, fold_id: int) -> tuple[Counter, Counter]:
        if not 0 <= fold_id < self.n_folds:
            raise IndexError(f"fold_id out of range: {fold_id}")
        pairs = self.pair_counts
        totals = self.source_totals
        pairs.subtract(self.fold_pair_counts[fold_id])
        totals.subtract(self.fold_source_totals[fold_id])
        return +pairs, +totals


@dataclass(frozen=True)
class TransitionContext:
    pair_counts: Counter[tuple[str, str]]
    source_totals: Counter[str]
    distributions: dict[str, dict[str, float]]
    destination_counts: Counter[str]
    total_transitions: int


@dataclass(frozen=True)
class CandidateSignals:
    direct: frozenset[str]
    semantic: frozenset[str]
    walk: frozenset[str]
    embedding: frozenset[str]
    skill: frozenset[str]
    popular: frozenset[str]
    jobhop: frozenset[str]
    semantic_scores: Mapping[str, tuple[float, float, int]]
    walk_2: Mapping[str, float]
    walk_3: Mapping[str, float]

    @property
    def candidates(self) -> set[str]:
        return set().union(
            self.direct,
            self.semantic,
            self.walk,
            self.embedding,
            self.skill,
            self.popular,
            self.jobhop,
        )


def stable_person_fold(person_id: str, n_folds: int) -> int:
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    digest = hashlib.blake2b(str(person_id).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % n_folds


def aggregate_person_disjoint_folds(
    csv_path: Path,
    title_index: Mapping[str, str],
    *,
    n_folds: int = 5,
    chunk_size: int = 200_000,
) -> FoldedTransitionCounts:
    """Stream Karrierewege train once and aggregate deterministic person folds."""
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    report = SplitQualityReport(split="train_oof", source_file=Path(csv_path).name)
    fold_pairs = tuple(Counter() for _ in range(n_folds))
    fold_totals = tuple(Counter() for _ in range(n_folds))
    observed_titles: set[str] = set()

    for person_id, rows in _iter_person_runs(Path(csv_path), report, chunk_size):
        report.trajectories += 1
        steps = _clean_person_steps(rows, report)
        observed_titles.update(title for _, title in steps)
        fold_id = stable_person_fold(person_id, n_folds)
        for (source_order, source), (target_order, target) in zip(steps, steps[1:]):
            if target_order != source_order + 1:
                report.nonconsecutive_pairs += 1
                continue
            if source == target:
                report.self_transitions += 1
                continue
            if source not in title_index or target not in title_index:
                continue
            source_id = str(title_index[source])
            target_id = str(title_index[target])
            fold_pairs[fold_id][(source_id, target_id)] += 1
            fold_totals[fold_id][source_id] += 1
            report.valid_transitions += 1

    unmapped = sorted(observed_titles.difference(title_index))
    report.unique_titles = len(observed_titles)
    report.mapped_titles = len(observed_titles) - len(unmapped)
    report.unmapped_titles = unmapped
    report.distinct_transition_pairs = len(set().union(*(set(item) for item in fold_pairs)))
    if unmapped:
        raise ValueError(f"Training titles do not map to the graph: {', '.join(unmapped[:10])}")
    return FoldedTransitionCounts(fold_pairs, fold_totals, report)


def build_transition_context(
    pair_counts: Mapping[tuple[str, str], int],
    source_totals: Mapping[str, int] | None = None,
) -> TransitionContext:
    pairs = Counter({(str(s), str(t)): int(c) for (s, t), c in pair_counts.items() if c > 0})
    totals = Counter(source_totals or {})
    if not totals:
        for (source, _), count in pairs.items():
            totals[source] += count
    distributions: dict[str, dict[str, float]] = defaultdict(dict)
    destinations: Counter[str] = Counter()
    for (source, target), count in pairs.items():
        if totals[source] <= 0:
            raise ValueError(f"Non-positive source total for {source}")
        distributions[source][target] = count / totals[source]
        destinations[target] += count
    return TransitionContext(
        pair_counts=pairs,
        source_totals=totals,
        distributions=dict(distributions),
        destination_counts=destinations,
        total_transitions=sum(pairs.values()),
    )


def relevance_from_count(count: int) -> int:
    """Map held-out frequency to stable LambdaMART relevance grades 0..3."""
    if count <= 0:
        return 0
    if count == 1:
        return 1
    if count <= 3:
        return 2
    return 3


class TransitionFeatureBuilder:
    """Build bounded candidates and graph/transition features for a context."""

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        embeddings: Mapping[str, np.ndarray],
        source_neighbours: Mapping[str, Sequence[tuple[str, float]]],
        destination_neighbours: Mapping[str, Sequence[tuple[str, float]]],
        *,
        semantic_neighbours: int = 20,
        semantic_temperature: float = 0.05,
        per_channel_limit: int = 40,
        auxiliary_context: TransitionContext | None = None,
    ) -> None:
        self.graph = graph
        self.embeddings = embeddings
        self.source_neighbours = source_neighbours
        self.destination_neighbours = destination_neighbours
        self.semantic_neighbours = semantic_neighbours
        self.semantic_temperature = semantic_temperature
        self.per_channel_limit = per_channel_limit
        self.auxiliary_context = auxiliary_context
        self.role_skills: dict[str, set[str]] = {}
        self.skill_roles: dict[str, set[str]] = defaultdict(set)
        self.skill_idf: dict[str, float] = {}
        for node_id, data in graph.nodes(data=True):
            if data.get("type") in {"skill", "element"}:
                self.skill_idf[str(node_id)] = float(data.get("idf", 1.0))
        for role_id, data in graph.nodes(data=True):
            if data.get("type") != "role" or data.get("source") != "esco":
                continue
            role = str(role_id)
            skills = {
                str(target)
                for _, target, edge in graph.out_edges(role_id, data=True)
                if edge.get("relation") == "REQUIRES"
                and graph.nodes[target].get("type") in {"skill", "element"}
            }
            self.role_skills[role] = skills
            for skill in skills:
                self.skill_roles[skill].add(role)

    @staticmethod
    def _top(scores: Mapping[str, float], limit: int) -> frozenset[str]:
        return frozenset(
            role_id
            for role_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
        )

    def _semantic_scores(
        self, source_id: str, context: TransitionContext
    ) -> dict[str, tuple[float, float, int]]:
        neighbours = [
            (role_id, similarity)
            for role_id, similarity in self.source_neighbours.get(source_id, ())
            if context.distributions.get(role_id)
        ][: self.semantic_neighbours]
        weights = _softmax_weights(
            [similarity for _, similarity in neighbours], self.semantic_temperature
        )
        probability: defaultdict[str, float] = defaultdict(float)
        max_similarity: defaultdict[str, float] = defaultdict(lambda: -1.0)
        support: Counter[str] = Counter()
        for (neighbour_id, similarity), weight in zip(neighbours, weights):
            for target_id, value in context.distributions[neighbour_id].items():
                if target_id == source_id:
                    continue
                probability[target_id] += weight * value
                max_similarity[target_id] = max(max_similarity[target_id], float(similarity))
                support[target_id] += 1
        return {
            target: (score, max_similarity[target], support[target])
            for target, score in probability.items()
        }

    def _skill_candidates(self, source_id: str) -> frozenset[str]:
        source_skills = self.role_skills.get(source_id, set())
        scores: Counter[str] = Counter()
        for skill in sorted(source_skills):
            weight = self.skill_idf.get(skill, 1.0)
            for role_id in sorted(self.skill_roles.get(skill, ())):
                if role_id != source_id:
                    scores[role_id] += weight
        return self._top(scores, self.per_channel_limit)

    def candidates(self, source_id: str, context: TransitionContext) -> CandidateSignals:
        source_id = str(source_id)
        direct_scores = context.distributions.get(source_id, {})
        direct = self._top(direct_scores, self.per_channel_limit)

        semantic_scores = self._semantic_scores(source_id, context)
        semantic = self._top(
            {target: values[0] for target, values in semantic_scores.items()},
            self.per_channel_limit,
        )

        walks = transition_walk_components(source_id, context.distributions, max_steps=3)
        walk_2 = walks[1]
        walk_3 = walks[2]
        walk = self._top(
            {target: walk_2.get(target, 0.0) + 0.25 * walk_3.get(target, 0.0)
             for target in set(walk_2) | set(walk_3)},
            self.per_channel_limit,
        )
        embedding = frozenset(
            target for target, _ in self.destination_neighbours.get(source_id, ())
            if target != source_id
        )
        skill = self._skill_candidates(source_id)
        popular = frozenset(
            target
            for target, _ in sorted(
                context.destination_counts.items(), key=lambda item: (-item[1], item[0])
            )[: self.per_channel_limit]
            if target != source_id
        )
        auxiliary_scores = (
            self.auxiliary_context.distributions.get(source_id, {})
            if self.auxiliary_context is not None
            else {}
        )
        jobhop = self._top(auxiliary_scores, self.per_channel_limit)
        return CandidateSignals(
            direct,
            semantic,
            walk,
            embedding,
            skill,
            popular,
            jobhop,
            semantic_scores,
            walk_2,
            walk_3,
        )

    def features(
        self,
        source_id: str,
        target_id: str,
        context: TransitionContext,
        signals: CandidateSignals,
    ) -> np.ndarray:
        source_id, target_id = str(source_id), str(target_id)
        count = context.pair_counts.get((source_id, target_id), 0)
        source_total = context.source_totals.get(source_id, 0)
        probability = count / source_total if source_total else 0.0
        semantic_probability, semantic_similarity, semantic_support = signals.semantic_scores.get(
            target_id, (0.0, 0.0, 0)
        )
        source_vector = self.embeddings.get(source_id)
        target_vector = self.embeddings.get(target_id)
        embedding_missing = source_vector is None or target_vector is None
        cosine = (
            float(np.dot(source_vector, target_vector)) if not embedding_missing else 0.0
        )

        source_data = self.graph.nodes.get(source_id, {})
        target_data = self.graph.nodes.get(target_id, {})
        source_isco = str(source_data.get("isco_2digit", ""))
        target_isco = str(target_data.get("isco_2digit", ""))
        same_isco = float(bool(source_isco and source_isco == target_isco))
        try:
            isco_distance = min(abs(int(source_isco) - int(target_isco)) / 90.0, 1.0)
        except ValueError:
            isco_distance = 0.5

        source_skills = self.role_skills.get(source_id, set())
        target_skills = self.role_skills.get(target_id, set())
        shared = source_skills & target_skills
        union = source_skills | target_skills
        skill_jaccard = len(shared) / len(union) if union else 0.0
        target_idf = sum(self.skill_idf.get(skill, 1.0) for skill in sorted(target_skills))
        shared_idf = sum(self.skill_idf.get(skill, 1.0) for skill in sorted(shared))
        idf_coverage = shared_idf / target_idf if target_idf else 0.0
        skill_gap = len(target_skills - source_skills) / len(target_skills) if target_skills else 0.0
        destination_count = context.destination_counts.get(target_id, 0)
        popularity = destination_count / context.total_transitions if context.total_transitions else 0.0
        jobhop_count = (
            self.auxiliary_context.pair_counts.get((source_id, target_id), 0)
            if self.auxiliary_context is not None
            else 0
        )
        jobhop_total = (
            self.auxiliary_context.source_totals.get(source_id, 0)
            if self.auxiliary_context is not None
            else 0
        )
        jobhop_probability = jobhop_count / jobhop_total if jobhop_total else 0.0

        values = (
            probability,
            float(count),
            math.log1p(count),
            math.log1p(source_total),
            count / (source_total + 5.0) if source_total else 0.0,
            float(count >= 2),
            float(count >= 3),
            float(count >= 5),
            semantic_probability,
            semantic_similarity,
            float(semantic_support),
            signals.walk_2.get(target_id, 0.0),
            signals.walk_3.get(target_id, 0.0),
            cosine,
            float(embedding_missing),
            same_isco,
            isco_distance,
            skill_jaccard,
            idf_coverage,
            skill_gap,
            popularity,
            math.log1p(destination_count),
            float(target_id in signals.direct),
            float(target_id in signals.semantic),
            float(target_id in signals.walk),
            float(target_id in signals.embedding),
            float(target_id in signals.skill),
            float(target_id in signals.popular),
            jobhop_probability,
            float(jobhop_count),
            math.log1p(jobhop_count),
            float(jobhop_count > 0),
            float(target_id in signals.jobhop),
        )
        return np.asarray(values, dtype=np.float32)


def build_ranker_rows(
    source_ids: Iterable[str],
    context: TransitionContext,
    labels: Mapping[tuple[str, str], int],
    builder: TransitionFeatureBuilder,
) -> tuple[np.ndarray, np.ndarray, list[int], list[tuple[str, str]], dict]:
    """Build grouped rows without using labels to expand candidate sets."""
    feature_rows: list[np.ndarray] = []
    relevance: list[int] = []
    groups: list[int] = []
    pairs: list[tuple[str, str]] = []
    truth_count = 0
    recalled_count = 0
    positive_rows = 0
    kept_queries = 0
    labels_by_source: defaultdict[str, dict[str, int]] = defaultdict(dict)
    for (source, target), count in labels.items():
        if count > 0:
            labels_by_source[str(source)][str(target)] = int(count)

    for source_id in sorted({str(item) for item in source_ids}):
        signals = builder.candidates(source_id, context)
        candidates = sorted(signals.candidates - {source_id})
        if not candidates:
            continue
        source_truth = labels_by_source.get(source_id, {})
        truth_count += sum(source_truth.values())
        recalled_count += sum(source_truth.get(target, 0) for target in candidates)
        rows_before = len(feature_rows)
        for target_id in candidates:
            feature_rows.append(builder.features(source_id, target_id, context, signals))
            label = relevance_from_count(source_truth.get(target_id, 0))
            relevance.append(label)
            positive_rows += int(label > 0)
            pairs.append((source_id, target_id))
        groups.append(len(feature_rows) - rows_before)
        kept_queries += 1

    matrix = (
        np.stack(feature_rows).astype(np.float32)
        if feature_rows
        else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    )
    report = {
        "queries": kept_queries,
        "rows": len(feature_rows),
        "positive_rows": positive_rows,
        "truth_observations": truth_count,
        "candidate_recalled_observations": recalled_count,
        "candidate_recall": recalled_count / truth_count if truth_count else 0.0,
    }
    return matrix, np.asarray(relevance, dtype=np.int32), groups, pairs, report


def prediction_map_from_scores(
    pairs: Sequence[tuple[str, str]],
    scores: Sequence[float],
    graph: nx.MultiDiGraph,
) -> dict[str, list[str]]:
    ranked: defaultdict[str, list[tuple[str, float]]] = defaultdict(list)
    for (source_id, target_id), score in zip(pairs, scores):
        ranked[source_id].append((target_id, float(score)))
    predictions: dict[str, list[str]] = {}
    for source_id, rows in ranked.items():
        source_title = str(graph.nodes[source_id].get("title", "")).strip()
        if not source_title:
            continue
        rows.sort(key=lambda item: (-item[1], item[0]))
        predictions[source_title] = [
            str(graph.nodes[target].get("title", "")).strip()
            for target, _ in rows
            if graph.has_node(target) and str(graph.nodes[target].get("title", "")).strip()
        ]
    return predictions


def head_preserving_fusion(
    primary: Mapping[str, Sequence[str]],
    secondary: Mapping[str, Sequence[str]],
    *,
    head_size: int,
    secondary_slots: int,
) -> dict[str, list[str]]:
    """Keep the trusted head and use a secondary ranker for the next slots."""
    if head_size < 0 or secondary_slots < 0:
        raise ValueError("head_size and secondary_slots must be non-negative")
    fused: dict[str, list[str]] = {}
    for source in sorted(set(primary) | set(secondary)):
        primary_rows = list(dict.fromkeys(primary.get(source, ())))
        secondary_rows = list(dict.fromkeys(secondary.get(source, ())))
        selected = primary_rows[:head_size]
        seen = set(selected)
        additions = 0
        for target in secondary_rows:
            if target in seen:
                continue
            selected.append(target)
            seen.add(target)
            additions += 1
            if additions >= secondary_slots:
                break
        for rows in (primary_rows[head_size:], secondary_rows):
            for target in rows:
                if target not in seen:
                    selected.append(target)
                    seen.add(target)
        fused[source] = selected
    return fused
