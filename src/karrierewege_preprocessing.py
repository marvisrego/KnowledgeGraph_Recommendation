"""Streaming preprocessing for the Karrierewege career-history dataset.

The raw files are several gigabytes, so only the three sequencing columns are
read.  Occupation labels are matched conservatively to ESCO role titles; the
dataset has complete exact coverage after normalization, so fuzzy matching is
deliberately excluded from the build path.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import networkx as nx
import pandas as pd


REQUIRED_COLUMNS = ("_id", "experience_order", "preferredLabel_en")
TRANSITION_RELATION = "TRANSITIONS_TO"
TRANSITION_SOURCE = "karrierewege"
TRANSITION_EDGE_KEY = f"{TRANSITION_SOURCE}:{TRANSITION_RELATION}"
SUPPORT_THRESHOLDS = (1, 2, 3, 5, 10, 25)


def normalize_label(value: object) -> str:
    """Return a conservative comparison key for an occupation or skill label."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def discover_split_files(data_dir: Path) -> dict[str, Path]:
    """Find exactly one train, validation, and test CSV below ``data_dir``."""
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Karrierewege data directory not found: {root}")

    matches: dict[str, list[Path]] = {"train": [], "validation": [], "test": []}
    for path in root.rglob("*.csv"):
        name = path.name.casefold()
        if "validation" in name:
            matches["validation"].append(path)
        elif "train" in name:
            matches["train"].append(path)
        elif "test" in name:
            matches["test"].append(path)

    errors = [
        f"{split}={len(paths)}"
        for split, paths in matches.items()
        if len(paths) != 1
    ]
    if errors:
        raise RuntimeError(
            "Expected exactly one CSV for each Karrierewege split; "
            + ", ".join(errors)
        )
    return {split: paths[0] for split, paths in matches.items()}


def build_esco_title_index(G: nx.Graph) -> dict[str, str]:
    """Map normalized ESCO role titles to unique graph node identifiers."""
    index: dict[str, str] = {}
    duplicates: dict[str, set[str]] = {}
    for node_id, data in G.nodes(data=True):
        if data.get("type") != "role" or data.get("source") != "esco":
            continue
        key = normalize_label(data.get("title", ""))
        if not key:
            continue
        if key in index and index[key] != str(node_id):
            duplicates.setdefault(key, {index[key]}).add(str(node_id))
        else:
            index[key] = str(node_id)

    if duplicates:
        sample = ", ".join(sorted(duplicates)[:5])
        raise RuntimeError(f"ESCO role titles are not unique after normalization: {sample}")
    if not index:
        raise RuntimeError("The graph contains no ESCO role titles.")
    return index


@dataclass
class SplitQualityReport:
    split: str
    source_file: str
    raw_rows: int = 0
    trajectories: int = 0
    accepted_steps: int = 0
    invalid_rows: int = 0
    missing_person_ids: int = 0
    invalid_orders: int = 0
    missing_titles: int = 0
    exact_duplicate_rows: int = 0
    ambiguous_positions: int = 0
    ambiguous_rows: int = 0
    out_of_order_trajectories: int = 0
    nonconsecutive_pairs: int = 0
    self_transitions: int = 0
    valid_transitions: int = 0
    distinct_transition_pairs: int = 0
    unique_titles: int = 0
    mapped_titles: int = 0
    unmapped_titles: list[str] = field(default_factory=list)
    support_retention: dict[str, dict[str, float | int]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TransitionAggregate:
    split: str
    pair_counts: Counter[tuple[str, str]]
    source_totals: Counter[str]
    report: SplitQualityReport


def _validate_columns(path: Path) -> None:
    columns = set(pd.read_csv(path, nrows=0).columns)
    missing = [column for column in REQUIRED_COLUMNS if column not in columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def _iter_person_runs(
    path: Path,
    report: SplitQualityReport,
    chunk_size: int,
) -> Iterable[tuple[str, pd.DataFrame]]:
    """Yield contiguous person runs while carrying a run across chunk boundaries."""
    _validate_columns(path)
    pending = pd.DataFrame(columns=REQUIRED_COLUMNS)
    completed: set[str] = set()

    reader = pd.read_csv(
        path,
        usecols=list(REQUIRED_COLUMNS),
        dtype={"_id": "string", "experience_order": "string", "preferredLabel_en": "string"},
        chunksize=chunk_size,
        low_memory=False,
    )

    for raw_chunk in reader:
        report.raw_rows += len(raw_chunk)
        report.missing_person_ids += int(raw_chunk["_id"].isna().sum())
        missing_title = (
            raw_chunk["preferredLabel_en"].isna()
            | raw_chunk["preferredLabel_en"].fillna("").str.strip().eq("")
        )
        report.missing_titles += int(missing_title.sum())

        orders = pd.to_numeric(raw_chunk["experience_order"], errors="coerce")
        integral = orders.notna() & orders.mod(1).abs().lt(1e-9)
        report.invalid_orders += int((~integral).sum())
        invalid_row = raw_chunk["_id"].isna() | missing_title | ~integral
        report.invalid_rows += int(invalid_row.sum())

        valid = raw_chunk.loc[
            raw_chunk["_id"].notna()
            & raw_chunk["preferredLabel_en"].notna()
            & raw_chunk["preferredLabel_en"].str.strip().ne("")
            & integral
        ].copy()
        valid["_id"] = valid["_id"].astype(str)
        valid["experience_order"] = orders.loc[valid.index].round().astype("int64")
        valid["preferredLabel_en"] = valid["preferredLabel_en"].astype(str)

        frame = pd.concat([pending, valid], ignore_index=True)
        if frame.empty:
            continue

        last_person = str(frame.iloc[-1]["_id"])
        last_run_start = len(frame) - 1
        while last_run_start > 0 and str(frame.iloc[last_run_start - 1]["_id"]) == last_person:
            last_run_start -= 1

        ready = frame.iloc[:last_run_start]
        pending = frame.iloc[last_run_start:].copy()

        if not ready.empty:
            change = ready["_id"].ne(ready["_id"].shift()).cumsum()
            for _, person_rows in ready.groupby(change, sort=False):
                person = str(person_rows.iloc[0]["_id"])
                if person in completed:
                    raise ValueError(
                        f"Person {person!r} reappeared in a non-contiguous run in {path.name}."
                    )
                completed.add(person)
                yield person, person_rows.reset_index(drop=True)

    if not pending.empty:
        person = str(pending.iloc[0]["_id"])
        if person in completed:
            raise ValueError(
                f"Person {person!r} reappeared in a non-contiguous run in {path.name}."
            )
        yield person, pending.reset_index(drop=True)


def _clean_person_steps(
    rows: pd.DataFrame,
    report: SplitQualityReport,
) -> list[tuple[int, str]]:
    if not rows["experience_order"].is_monotonic_increasing:
        report.out_of_order_trajectories += 1

    by_order: dict[int, list[str]] = {}
    for order, title in rows[["experience_order", "preferredLabel_en"]].itertuples(
        index=False, name=None
    ):
        by_order.setdefault(int(order), []).append(normalize_label(title))

    cleaned: list[tuple[int, str]] = []
    for order in sorted(by_order):
        labels = [label for label in by_order[order] if label]
        unique = sorted(set(labels))
        if len(unique) > 1:
            report.ambiguous_positions += 1
            report.ambiguous_rows += len(labels)
            continue
        if not unique:
            continue
        report.exact_duplicate_rows += max(0, len(labels) - 1)
        cleaned.append((order, unique[0]))

    report.accepted_steps += len(cleaned)
    return cleaned


def aggregate_split_transitions(
    csv_path: Path,
    split: str,
    title_index: dict[str, str],
    chunk_size: int = 200_000,
    strict_mapping: bool = True,
    max_invalid_row_ratio: float = 0.001,
) -> TransitionAggregate:
    """Clean one split and aggregate valid, consecutive non-self transitions."""
    if chunk_size < 2:
        raise ValueError("chunk_size must be at least 2")
    if not 0.0 <= max_invalid_row_ratio <= 1.0:
        raise ValueError("max_invalid_row_ratio must be between 0 and 1")

    path = Path(csv_path)
    report = SplitQualityReport(split=split, source_file=path.name)
    pair_counts: Counter[tuple[str, str]] = Counter()
    source_totals: Counter[str] = Counter()
    observed_titles: set[str] = set()

    for _, rows in _iter_person_runs(path, report, chunk_size):
        report.trajectories += 1
        steps = _clean_person_steps(rows, report)
        observed_titles.update(title for _, title in steps)

        for (source_order, source_title), (target_order, target_title) in zip(steps, steps[1:]):
            if target_order != source_order + 1:
                report.nonconsecutive_pairs += 1
                continue
            if source_title == target_title:
                report.self_transitions += 1
                continue
            if source_title not in title_index or target_title not in title_index:
                continue
            pair_counts[(source_title, target_title)] += 1
            source_totals[source_title] += 1
            report.valid_transitions += 1

    unmapped = sorted(observed_titles.difference(title_index))
    report.unique_titles = len(observed_titles)
    report.mapped_titles = len(observed_titles) - len(unmapped)
    report.unmapped_titles = unmapped
    report.distinct_transition_pairs = len(pair_counts)

    invalid_ratio = report.invalid_rows / report.raw_rows if report.raw_rows else 0.0
    if invalid_ratio > max_invalid_row_ratio:
        raise ValueError(
            f"{path.name} invalid-row ratio {invalid_ratio:.6f} exceeds "
            f"the configured tolerance {max_invalid_row_ratio:.6f}."
        )

    total_observations = sum(pair_counts.values())
    for threshold in SUPPORT_THRESHOLDS:
        retained = [count for count in pair_counts.values() if count >= threshold]
        retained_observations = sum(retained)
        report.support_retention[str(threshold)] = {
            "edges": len(retained),
            "observations": retained_observations,
            "observation_ratio": (
                retained_observations / total_observations if total_observations else 0.0
            ),
        }

    if strict_mapping and unmapped:
        sample = ", ".join(unmapped[:10])
        raise ValueError(
            f"{len(unmapped)} Karrierewege occupation labels do not map exactly to ESCO: {sample}"
        )

    return TransitionAggregate(split, pair_counts, source_totals, report)


def transition_rows(
    aggregate: TransitionAggregate,
    title_index: dict[str, str],
    min_support: int = 5,
) -> list[dict]:
    """Convert aggregate counts into graph-ready transition records."""
    if min_support < 1:
        raise ValueError("min_support must be at least 1")
    if aggregate.split != "train":
        raise ValueError(
            f"Only the training split may be converted to graph edges, got {aggregate.split!r}."
        )

    rows: list[dict] = []
    for (source_title, target_title), count in aggregate.pair_counts.items():
        if count < min_support:
            continue
        source_total = aggregate.source_totals[source_title]
        if source_total <= 0:
            raise ValueError(f"Invalid source total for {source_title!r}: {source_total}")
        probability = count / source_total
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"Invalid transition probability for {source_title!r} -> {target_title!r}: "
                f"{probability}"
            )
        try:
            source_id = title_index[source_title]
            target_id = title_index[target_title]
        except KeyError as exc:
            raise ValueError(f"Transition endpoint is absent from the ESCO graph: {exc}") from exc

        rows.append(
            {
                "src": source_id,
                "dst": target_id,
                "relation": TRANSITION_RELATION,
                "source": TRANSITION_SOURCE,
                "count": int(count),
                "probability": float(probability),
                "source_total": int(source_total),
                "split": "train",
                "min_support": int(min_support),
            }
        )

    rows.sort(key=lambda row: (row["src"], -row["probability"], -row["count"], row["dst"]))
    return rows


def remove_transition_edges(G: nx.MultiDiGraph) -> int:
    """Remove only previously generated Karrierewege transition edges."""
    doomed = [
        (source, target, key)
        for source, target, key, data in G.edges(keys=True, data=True)
        if data.get("relation") == TRANSITION_RELATION
        and data.get("source") == TRANSITION_SOURCE
    ]
    G.remove_edges_from(doomed)
    return len(doomed)


def add_transition_edges(G: nx.MultiDiGraph, rows: Iterable[dict]) -> int:
    """Idempotently replace Karrierewege transition edges in ``G``."""
    if not G.is_multigraph():
        raise TypeError("Karrierewege transitions require a MultiDiGraph.")
    remove_transition_edges(G)

    added = 0
    for row in rows:
        source = str(row["src"])
        target = str(row["dst"])
        if not G.has_node(source) or not G.has_node(target):
            raise ValueError(f"Transition endpoint missing from graph: {source} -> {target}")
        attrs = {key: value for key, value in row.items() if key not in ("src", "dst")}
        G.add_edge(source, target, key=TRANSITION_EDGE_KEY, **attrs)
        added += 1
    return added


def write_json_atomic(payload: dict, path: Path) -> None:
    """Write JSON through a sibling temporary file and atomically replace the target."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.replace(temp_name, destination)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
