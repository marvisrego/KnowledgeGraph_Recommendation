"""Evaluate a train-only, reliability-smoothed second-order transition ranker.

This is a research diagnostic, not a production runtime.  It asks whether the
previous occupation adds information beyond the current occupation when
predicting the next role.  All counts are built exclusively from the training
split; validation chooses the fixed grid configuration and test is read once.
Unscored destinations receive zero credit, matching the existing transition
benchmark's treatment of destinations without train-derived evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Settings
from src.karrierewege_preprocessing import (
    SplitQualityReport,
    _clean_person_steps,
    _iter_person_runs,
    discover_split_files,
    write_json_atomic,
)


def iter_prefixes(path: Path, split: str, chunk_size: int):
    """Yield consecutive, de-duplicated label histories and the next label."""
    report = SplitQualityReport(split=split, source_file=path.name)
    for _, rows in _iter_person_runs(path, report, chunk_size):
        history: list[str] = []
        previous_order: int | None = None
        for order, role in _clean_person_steps(rows, report):
            if previous_order is not None and order != previous_order + 1:
                history = []
            if history and history[-1] != role:
                yield tuple(history[-2:]), role
            if not history or history[-1] != role:
                history.append(role)
            previous_order = order


def build_counts(path: Path, chunk_size: int):
    """Build first- and second-order counts from training prefixes only."""
    first = Counter()
    first_total = Counter()
    second = Counter()
    second_total = Counter()
    for history, target in iter_prefixes(path, "train", chunk_size):
        source = history[-1]
        first[source, target] += 1
        first_total[source] += 1
        if len(history) == 2:
            second[history[0], source, target] += 1
            second_total[history] += 1
    first_prob: dict[str, dict[str, float]] = defaultdict(dict)
    for (source, target), count in first.items():
        # Preserve the current graph's support threshold for the one-step part.
        if count >= 5:
            first_prob[source][target] = count / first_total[source]
    second_counts: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for (previous, source, target), count in second.items():
        second_counts[previous, source][target] = count
    return first_prob, second_counts, second_total


def rank_position(
    history: tuple[str, ...],
    target: str,
    first_prob: dict[str, dict[str, float]],
    second_counts: dict[tuple[str, str], dict[str, int]],
    second_total: Counter,
    *,
    weight: float,
    alpha: float,
    minimum_context_support: int,
) -> int:
    """Return the target position without sorting candidates unnecessarily."""
    base = first_prob.get(history[-1], {})
    if len(history) < 2:
        scores = base
    else:
        context = (history[-2], history[-1])
        total = second_total.get(context, 0)
        observed = second_counts.get(context, {})
        if total < minimum_context_support or not observed:
            scores = base
        else:
            # Empirical-Bayes shrinkage keeps a rare two-role context close to P(t|s).
            denominator = total + alpha
            scores = {
                candidate: (1.0 - weight) * probability + weight * alpha * probability / denominator
                for candidate, probability in base.items()
            }
            for candidate, count in observed.items():
                scores[candidate] = scores.get(candidate, 0.0) + weight * count / denominator
    target_score = scores.get(target)
    if target_score is None:
        return 0
    return 1 + sum(
        score > target_score or (score == target_score and candidate < target)
        for candidate, score in scores.items()
        if candidate != target
    )


def evaluate(
    prefixes,
    first_prob,
    second_counts,
    second_total,
    *,
    weight: float,
    alpha: float,
    minimum_context_support: int,
) -> dict[str, float | int]:
    totals = Counter()
    for history, target in prefixes:
        totals["examples"] += 1
        totals["multi_role_examples"] += int(len(history) == 2)
        position = rank_position(
            history,
            target,
            first_prob,
            second_counts,
            second_total,
            weight=weight,
            alpha=alpha,
            minimum_context_support=minimum_context_support,
        )
        if position:
            totals["mrr_sum"] += 1.0 / position
            for cutoff in (1, 3, 5, 10):
                totals[f"hits_{cutoff}"] += int(position <= cutoff)
    count = totals["examples"]
    if not count:
        raise RuntimeError("No prefixes were available")
    return {
        "examples": int(count),
        "multi_role_share": totals["multi_role_examples"] / count,
        "mrr": totals["mrr_sum"] / count,
        **{f"hits_at_{cutoff}": totals[f"hits_{cutoff}"] / count for cutoff in (1, 3, 5, 10)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/karrierewege/variable_order_evaluation.json"))
    args = parser.parse_args()
    settings = Settings.from_env()
    files = discover_split_files(settings.karrierewege_dir)
    print("[variable-order] building train-only counts", flush=True)
    first_prob, second_counts, second_total = build_counts(files["train"], settings.transition_chunk_size)
    print("[variable-order] loading held-out prefixes", flush=True)
    validation_prefixes = list(iter_prefixes(files["validation"], "validation", settings.transition_chunk_size))
    test_prefixes = list(iter_prefixes(files["test"], "test", settings.transition_chunk_size))
    baseline = {"weight": 0.0, "alpha": 1.0, "minimum_context_support": 10**9}
    validation_baseline = evaluate(validation_prefixes, first_prob, second_counts, second_total, **baseline)
    trials = []
    for support in (5, 10):
        for alpha in (10.0, 30.0):
            for weight in (0.1, 0.25):
                params = {"weight": weight, "alpha": alpha, "minimum_context_support": support}
                metrics = evaluate(validation_prefixes, first_prob, second_counts, second_total, **params)
                trials.append({"parameters": params, "metrics": metrics})
    # Validation-only selection: MRR, then Hits@5, then smaller context weight.
    selected = max(
        trials,
        key=lambda trial: (trial["metrics"]["mrr"], trial["metrics"]["hits_at_5"], -trial["parameters"]["weight"]),
    )
    print(f"[variable-order] selected {selected['parameters']}", flush=True)
    test_baseline = evaluate(test_prefixes, first_prob, second_counts, second_total, **baseline)
    test_selected = evaluate(test_prefixes, first_prob, second_counts, second_total, **selected["parameters"])
    payload = {
        "protocol": {
            "training_split": "train",
            "selection_split": "validation",
            "locked_evaluation_split": "test",
            "unit": "ordered career prefix",
            "candidate_policy": "train-derived scored destinations; unscored destinations receive zero credit",
            "method": "empirical-Bayes second-order interpolation",
        },
        "training": {"first_order_sources": len(first_prob), "second_order_contexts": len(second_counts)},
        "validation": {"baseline": validation_baseline, "trials": trials, "selected": selected},
        "test": {
            "baseline": test_baseline,
            "selected": test_selected,
            "delta": {metric: test_selected[metric] - test_baseline[metric] for metric in ("mrr", "hits_at_1", "hits_at_3", "hits_at_5", "hits_at_10")},
        },
    }
    output = args.output if args.output.is_absolute() else Path(__file__).resolve().parents[1] / args.output
    write_json_atomic(payload, output)
    print(json.dumps({"output": str(output), "selected": selected, "test": payload["test"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
