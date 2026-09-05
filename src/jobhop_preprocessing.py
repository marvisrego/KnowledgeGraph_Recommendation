"""Validate, clean, and ESCO-map the public JobHop v2 trajectory dataset."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping

import networkx as nx
import pandas as pd


REQUIRED_COLUMNS = (
    "resume_id",
    "matched_code",
    "start_date",
    "end_date",
    "university_level",
)
EXPECTED_SPLITS = ("train", "val", "test")
QUARTER_RE = re.compile(r"^Q([1-4])\s+(\d{4})$")
EDUCATION_NORMALIZATION = {
    "None": "None",
    "Secondary": "Secondary",
    "Secondary school": "Secondary",
    "Bachelor": "Bachelor",
    "Master": "Master",
    "PhD": "PhD",
}


@dataclass
class JobHopQualityReport:
    split: str
    source_file: str
    raw_rows: int = 0
    raw_people: int = 0
    exact_duplicate_rows: int = 0
    unknown_code_rows: int = 0
    unmapped_code_rows: int = 0
    unmapped_codes: list[str] = field(default_factory=list)
    missing_start_dates: int = 0
    missing_end_dates: int = 0
    invalid_start_dates: int = 0
    invalid_end_dates: int = 0
    invalid_date_ranges: int = 0
    unordered_rows: int = 0
    invalid_education_rows: int = 0
    accepted_rows: int = 0
    accepted_people: int = 0
    ambiguous_same_start_pairs: int = 0
    overlapping_pairs: int = 0
    self_transitions: int = 0
    valid_transitions: int = 0
    distinct_transition_pairs: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CleanedJobHopSplit:
    split: str
    frame: pd.DataFrame
    pair_counts: Counter[tuple[str, str]]
    report: JobHopQualityReport


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a lowercase SHA256 digest without loading the whole file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(data_dir: Path) -> dict[str, dict[str, str | int]]:
    """Validate local JobHop files against the checked-in local manifest."""
    root = Path(data_dir)
    manifest_path = root / "MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"JobHop manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for filename, expected in manifest.get("files", {}).items():
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"JobHop file not found: {path}")
        size = path.stat().st_size
        digest = file_sha256(path)
        if size != int(expected["bytes"]):
            raise ValueError(f"JobHop size mismatch for {filename}: {size}")
        if digest.lower() != str(expected["sha256"]).lower():
            raise ValueError(f"JobHop checksum mismatch for {filename}")
    return manifest


def esco_code_index(graph: nx.MultiDiGraph) -> dict[str, str]:
    """Map unique live ESCO codes to role IDs, rejecting ambiguous codes."""
    candidates: dict[str, list[str]] = {}
    for node_id, data in graph.nodes(data=True):
        if data.get("type") != "role" or data.get("source") != "esco":
            continue
        code = str(data.get("esco_code", "")).strip()
        if code:
            candidates.setdefault(code, []).append(str(node_id))
    duplicate_codes = {code: ids for code, ids in candidates.items() if len(ids) != 1}
    if duplicate_codes:
        sample = ", ".join(sorted(duplicate_codes)[:5])
        raise ValueError(f"Ambiguous ESCO codes in graph: {sample}")
    return {code: ids[0] for code, ids in candidates.items()}


def parse_quarter(value: object) -> int | None:
    """Convert ``Q# YYYY`` into a monotonically ordered integer quarter."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    match = QUARTER_RE.fullmatch(text)
    if not match:
        raise ValueError(f"Invalid quarter value: {text!r}")
    quarter, year = int(match.group(1)), int(match.group(2))
    return year * 4 + quarter - 1


def _parse_quarter_column(series: pd.Series) -> tuple[pd.Series, int, int]:
    parsed: list[int | None] = []
    missing = 0
    invalid = 0
    for value in series.tolist():
        if pd.isna(value) or not str(value).strip():
            parsed.append(None)
            missing += 1
            continue
        try:
            parsed.append(parse_quarter(value))
        except ValueError:
            parsed.append(None)
            invalid += 1
    return pd.Series(parsed, index=series.index, dtype="Int64"), missing, invalid


def clean_jobhop_frame(
    frame: pd.DataFrame,
    split: str,
    code_index: Mapping[str, str],
) -> CleanedJobHopSplit:
    """Clean one in-memory JobHop split and aggregate unambiguous transitions."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"JobHop split is missing columns: {', '.join(missing_columns)}")

    report = JobHopQualityReport(split=split, source_file="<memory>")
    work = frame.loc[:, REQUIRED_COLUMNS].copy()
    report.raw_rows = len(work)
    report.raw_people = int(work["resume_id"].nunique(dropna=True))
    report.exact_duplicate_rows = int(work.duplicated().sum())
    work = work.drop_duplicates().copy()

    work["matched_code"] = work["matched_code"].astype("string").fillna("").str.strip()
    unknown = work["matched_code"].str.lower().eq("unknown") | work["matched_code"].eq("")
    report.unknown_code_rows = int(unknown.sum())
    work = work.loc[~unknown].copy()

    mapped = work["matched_code"].map(code_index)
    unmapped = mapped.isna()
    report.unmapped_code_rows = int(unmapped.sum())
    report.unmapped_codes = sorted(work.loc[unmapped, "matched_code"].unique().tolist())
    work = work.loc[~unmapped].copy()
    work["role_id"] = mapped.loc[~unmapped].astype(str)

    start_quarter, missing_start, invalid_start = _parse_quarter_column(work["start_date"])
    end_quarter, missing_end, invalid_end = _parse_quarter_column(work["end_date"])
    report.missing_start_dates = missing_start
    report.missing_end_dates = missing_end
    report.invalid_start_dates = invalid_start
    report.invalid_end_dates = invalid_end
    work["start_quarter"] = start_quarter
    work["end_quarter"] = end_quarter

    invalid_range = (
        work["start_quarter"].notna()
        & work["end_quarter"].notna()
        & (work["end_quarter"] < work["start_quarter"])
    )
    report.invalid_date_ranges = int(invalid_range.sum())
    work = work.loc[~invalid_range].copy()

    work["order_quarter"] = work["start_quarter"].fillna(work["end_quarter"])
    unordered = work["order_quarter"].isna()
    report.unordered_rows = int(unordered.sum())
    work = work.loc[~unordered].copy()

    education = work["university_level"].astype("string").fillna("").str.strip()
    normalized_education = education.map(EDUCATION_NORMALIZATION)
    invalid_education = normalized_education.isna()
    report.invalid_education_rows = int(invalid_education.sum())
    work["university_level"] = normalized_education.fillna("None")

    work = work.sort_values(
        ["resume_id", "order_quarter", "end_quarter", "matched_code", "role_id"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)
    report.accepted_rows = len(work)
    report.accepted_people = int(work["resume_id"].nunique())

    pair_counts: Counter[tuple[str, str]] = Counter()
    for _, rows in work.groupby("resume_id", sort=False):
        records = list(
            rows[["role_id", "order_quarter", "start_quarter", "end_quarter"]].itertuples(
                index=False, name=None
            )
        )
        for source, target in zip(records, records[1:]):
            source_id, source_order, _, source_end = source
            target_id, target_order, target_start, _ = target
            if int(source_order) == int(target_order):
                report.ambiguous_same_start_pairs += 1
                continue
            if pd.notna(source_end) and pd.notna(target_start) and int(target_start) < int(source_end):
                report.overlapping_pairs += 1
            if source_id == target_id:
                report.self_transitions += 1
                continue
            pair_counts[(str(source_id), str(target_id))] += 1
            report.valid_transitions += 1

    report.distinct_transition_pairs = len(pair_counts)
    return CleanedJobHopSplit(split=split, frame=work, pair_counts=pair_counts, report=report)


def _write_json_atomic(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def preprocess_jobhop_dataset(
    graph: nx.MultiDiGraph,
    data_dir: Path,
    output_dir: Path,
) -> dict:
    """Clean all official splits, preserve split isolation, and write local outputs."""
    root = Path(data_dir)
    destination = Path(output_dir)
    manifest = verify_manifest(root)
    code_index = esco_code_index(graph)
    reports: dict[str, dict] = {}
    people: dict[str, set[int]] = {}

    destination.mkdir(parents=True, exist_ok=True)
    for split in EXPECTED_SPLITS:
        source_path = root / f"JobHop_v2_{split}.parquet"
        raw = pd.read_parquet(source_path, columns=list(REQUIRED_COLUMNS))
        people[split] = set(raw["resume_id"].astype(int).tolist())
        cleaned = clean_jobhop_frame(raw, split, code_index)
        cleaned.report.source_file = source_path.name
        cleaned.frame.to_parquet(destination / f"{split}.parquet", index=False)
        transition_rows = [
            {
                "source_role_id": source,
                "target_role_id": target,
                "count": count,
            }
            for (source, target), count in sorted(cleaned.pair_counts.items())
        ]
        pd.DataFrame(
            transition_rows,
            columns=["source_role_id", "target_role_id", "count"],
        ).to_parquet(destination / f"{split}_transition_counts.parquet", index=False)
        reports[split] = cleaned.report.to_dict()

    overlaps = {
        "train_val": len(people["train"] & people["val"]),
        "train_test": len(people["train"] & people["test"]),
        "val_test": len(people["val"] & people["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"JobHop person IDs overlap across official splits: {overlaps}")

    payload = {
        "dataset": manifest.get("dataset"),
        "version": manifest.get("version"),
        "license": manifest.get("license"),
        "graph_esco_codes": len(code_index),
        "split_person_overlap": overlaps,
        "splits": reports,
    }
    _write_json_atomic(payload, destination / "preprocessing_report.json")
    return payload
