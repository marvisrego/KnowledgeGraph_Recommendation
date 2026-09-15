"""Leakage-safe, prefix-level career-trajectory benchmark utilities."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from src.karrierewege_preprocessing import (
    SplitQualityReport,
    _clean_person_steps,
    _iter_person_runs,
)


@dataclass(frozen=True)
class TrajectoryPrefix:
    """One chronological prediction event, retaining only anonymous person ID."""

    person_id: str
    history_role_ids: tuple[str, ...]
    target_role_id: str


def iter_karrierewege_prefixes(
    csv_path: Path,
    title_index: dict[str, str],
    split: str,
    chunk_size: int = 200_000,
) -> Iterable[TrajectoryPrefix]:
    """Yield next-role examples without ever mixing people across a split.

    Histories restart after a missing order or an unmapped/duplicate role. This
    matches the existing preprocessing rule that only consecutive, non-self
    transitions are valid observations.
    """
    report = SplitQualityReport(split=split, source_file=Path(csv_path).name)
    for person_id, rows in _iter_person_runs(Path(csv_path), report, chunk_size):
        steps = _clean_person_steps(rows, report)
        history: list[str] = []
        previous_order: int | None = None
        for order, title in steps:
            role_id = title_index.get(title)
            if role_id is None:
                history = []
                previous_order = None
                continue
            if previous_order is not None and order != previous_order + 1:
                history = []
            if history and history[-1] != role_id:
                yield TrajectoryPrefix(str(person_id), tuple(history), role_id)
            if not history or history[-1] != role_id:
                history.append(role_id)
            previous_order = order


def evaluate_prefix_rankings(
    examples: Iterable[TrajectoryPrefix],
    ranker: Callable[[Sequence[str]], Sequence[str]],
    ks: tuple[int, ...] = (1, 3, 5, 10),
    bootstrap_samples: int = 500,
    seed: int = 17,
) -> dict:
    """Evaluate a full-candidate ranker at the person level with CIs."""
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("ks must contain positive integers")
    per_person: dict[str, list[dict[str, float]]] = defaultdict(list)
    total = 0
    covered = 0
    for example in examples:
        predicted = [str(role) for role in ranker(example.history_role_ids)]
        rank = next((index + 1 for index, role in enumerate(predicted) if role == example.target_role_id), None)
        total += 1
        if predicted:
            covered += 1
        record = {"mrr": 0.0 if rank is None else 1.0 / rank}
        for k in ks:
            record[f"hits_at_{k}"] = float(rank is not None and rank <= k)
            record[f"ndcg_at_{k}"] = 0.0 if rank is None or rank > k else 1.0 / math.log2(rank + 1)
        per_person[example.person_id].append(record)
    if not total:
        raise ValueError("No trajectory prefixes were available for evaluation")

    def _aggregate(records: Iterable[dict[str, float]]) -> dict[str, float]:
        rows = list(records)
        return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}

    overall = _aggregate(record for records in per_person.values() for record in records)
    people = sorted(per_person)
    rng = np.random.default_rng(seed)
    intervals: dict[str, list[float]] = {key: [] for key in overall}
    # Keep the person as the resampling cluster, but preserve the event-level
    # estimand used by ``overall``.  Averaging person means would silently
    # switch the confidence interval to a macro-person metric.
    person_sums = np.asarray([
        [sum(record[key] for record in per_person[person]) for key in overall]
        for person in people
    ])
    person_counts = np.asarray([len(per_person[person]) for person in people], dtype=np.int64)
    for _ in range(bootstrap_samples):
        draw = rng.integers(0, len(people), size=len(people))
        estimates = person_sums[draw].sum(axis=0) / person_counts[draw].sum()
        for index, key in enumerate(overall):
            intervals[key].append(float(estimates[index]))

    return {
        "examples": total,
        "people": len(people),
        "coverage": covered / total,
        **overall,
        "confidence_intervals_95": {
            key: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
            for key, values in intervals.items()
        },
        "protocol": {
            "unit": "ordered_person_prefix",
            "candidate_set": "all_live_esco_roles",
            "split_isolation": "person_disjoint_input_splits",
            "bootstrap": {"unit": "person", "samples": bootstrap_samples, "seed": seed},
        },
    }
