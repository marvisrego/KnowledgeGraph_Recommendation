"""Leakage-safe, feature-aware prefix construction for JobHop experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

import pandas as pd


FEATURE_VOCABULARIES = (9, 8, 6)
EDUCATION_TO_INDEX = {"None": 1, "Secondary": 2, "Bachelor": 3, "Master": 4, "PhD": 5}
REQUIRED_COLUMNS = (
    "resume_id", "role_id", "order_quarter", "start_quarter", "end_quarter", "university_level",
)


@dataclass(frozen=True)
class JobHopFeaturePrefix:
    person_id: str
    history_role_ids: tuple[str, ...]
    target_role_id: str
    context_features: tuple[int, int, int]


@dataclass(frozen=True)
class _Record:
    role_id: str
    order: int
    start: int | None
    end: int | None
    education: str


def tenure_bucket(start: int | None, end: int | None) -> int:
    """Bucket observed tenure in quarters; zero is explicitly missing."""
    if start is None or end is None:
        return 0
    duration = end - start
    if duration <= 0:
        return 1
    if duration <= 1:
        return 2
    if duration <= 3:
        return 3
    if duration <= 7:
        return 4
    if duration <= 15:
        return 5
    if duration <= 31:
        return 6
    if duration <= 63:
        return 7
    return 8


def gap_bucket(previous_end: int | None, current_start: int | None) -> int:
    """Bucket the observable gap before the current role; zero is missing."""
    if previous_end is None or current_start is None:
        return 0
    gap = current_start - previous_end
    if gap < 0:
        return 1
    if gap == 0:
        return 2
    if gap <= 1:
        return 3
    if gap <= 3:
        return 4
    if gap <= 7:
        return 5
    if gap <= 15:
        return 6
    return 7


def education_bucket(value: object, *, include_static_resume_education: bool) -> int:
    """Return zero unless the explicitly labelled static-resume ablation is enabled."""
    if not include_static_resume_education:
        return 0
    return EDUCATION_TO_INDEX.get(str(value), 0)


def _quarter(value: object) -> int | None:
    return None if pd.isna(value) else int(value)


def iter_jobhop_feature_prefixes(
    path: Path,
    *,
    maximum_history_length: int,
    include_static_resume_education: bool = False,
) -> Iterator[JobHopFeaturePrefix]:
    """Yield chronological prefixes without exposing target-row information.

    Same-quarter rows are ambiguous and discarded. Consecutive duplicate roles
    add neither a target nor a history event. For target record ``i``, every
    context feature comes from the final retained history record ``i - 1``.
    """
    if maximum_history_length < 1:
        raise ValueError("maximum_history_length must be positive")
    frame = pd.read_parquet(path, columns=list(REQUIRED_COLUMNS))
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"JobHop split is missing columns: {', '.join(sorted(missing))}")
    frame = frame.sort_values(["resume_id", "order_quarter", "role_id"], kind="mergesort")
    for person, rows in frame.groupby("resume_id", sort=False):
        retained: list[_Record] = []
        for row in rows.itertuples(index=False):
            record = _Record(
                role_id=str(row.role_id),
                order=int(row.order_quarter),
                start=_quarter(row.start_quarter),
                end=_quarter(row.end_quarter),
                education=str(row.university_level),
            )
            if retained and record.order == retained[-1].order:
                continue
            if retained and record.role_id == retained[-1].role_id:
                continue
            if retained:
                current = retained[-1]
                previous = retained[-2] if len(retained) > 1 else None
                yield JobHopFeaturePrefix(
                    person_id=str(person),
                    history_role_ids=tuple(item.role_id for item in retained[-maximum_history_length:]),
                    target_role_id=record.role_id,
                    context_features=(
                        tenure_bucket(current.start, current.end),
                        gap_bucket(previous.end if previous else None, current.start),
                        education_bucket(
                            current.education,
                            include_static_resume_education=include_static_resume_education,
                        ),
                    ),
                )
            retained.append(record)


def split_people(path: Path) -> set[str]:
    return set(pd.read_parquet(path, columns=["resume_id"])["resume_id"].astype(str))


def assert_disjoint_splits(paths: Mapping[str, Path]) -> None:
    people = {name: split_people(path) for name, path in paths.items()}
    names = sorted(people)
    overlaps = {
        f"{left}_{right}": len(people[left] & people[right])
        for index, left in enumerate(names)
        for right in names[index + 1:]
    }
    if any(overlaps.values()):
        raise ValueError(f"JobHop split people overlap: {overlaps}")


def feature_availability(prefixes: list[JobHopFeaturePrefix]) -> dict[str, int]:
    return {
        "prefixes": len(prefixes),
        "missing_tenure": sum(prefix.context_features[0] == 0 for prefix in prefixes),
        "missing_gap": sum(prefix.context_features[1] == 0 for prefix in prefixes),
        "static_resume_education": sum(prefix.context_features[2] != 0 for prefix in prefixes),
    }
