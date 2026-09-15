"""Bind validation-selected trajectory-fusion weights to one artifact checksum."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.karrierewege_preprocessing import write_json_atomic


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--validation-fusion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    fusion = json.loads(args.validation_fusion.read_text(encoding="utf-8"))
    artifact_hash = sha256(args.artifact)
    if selection.get("selected_artifact_sha256") != artifact_hash:
        raise ValueError("Selection record does not select this artifact")
    if fusion.get("split") != "validation_only":
        raise ValueError("Fusion report was not produced on validation only")
    weights = fusion.get("selected_weights")
    if not isinstance(weights, dict) or set(weights) != {"single", "multi"}:
        raise ValueError("Fusion report lacks selected single/multi weights")
    payload = {
        "selection_split": "validation",
        "artifact_sha256": artifact_hash,
        "selection_report": str(args.selection),
        "selection_report_sha256": sha256(args.selection),
        "validation_fusion_report": str(args.validation_fusion),
        "validation_fusion_report_sha256": sha256(args.validation_fusion),
        "selected_weights": weights,
        "test_data_read_at_freeze": False,
    }
    write_json_atomic(payload, args.output)
    print(json.dumps({"output": str(args.output), **payload}, indent=2))


if __name__ == "__main__":
    main()
