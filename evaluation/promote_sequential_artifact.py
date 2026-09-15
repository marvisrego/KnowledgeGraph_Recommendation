"""Create a separate deployable artifact only after immutable promotion checks pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_manifest(manifest: dict, source: Path, report_path: Path, report: dict) -> None:
    _require(manifest.get("source_artifact_sha256") == sha256(source), "Manifest source artifact checksum does not match")
    _require(manifest.get("frozen_test_report_sha256") == sha256(report_path), "Manifest test report checksum does not match")
    _require(isinstance(manifest.get("promotion_run"), str) and manifest["promotion_run"].strip(), "Manifest needs promotion_run")
    expected_candidates = int(manifest.get("candidate_roles", 3039))
    _require(int(report.get("candidate_roles", 0)) == expected_candidates, "Test report candidate count does not match manifest")
    _require(float(report.get("coverage", 0.0)) == 1.0, "Test report lacks full candidate coverage")
    protocol = report.get("protocol", {})
    _require(protocol.get("candidate_set") == "all_artifact_live_esco_roles", "Test report is not full live-ESCO ranking")
    _require(protocol.get("current_role") == "excluded", "Test report does not exclude current role")
    gates = manifest.get("gates")
    _require(isinstance(gates, dict), "Manifest needs gates")
    for name in (
        "multi_seed_stable", "paired_bootstrap_improvement", "fusion_policy_frozen",
        "api_integration_passed", "test_blind_for_selected_artifact",
    ):
        _require(gates.get(name) is True, f"Promotion gate failed: {name}")
    for name in ("latency_ms_p95", "latency_limit_ms", "memory_mb", "memory_limit_mb"):
        _require(isinstance(gates.get(name), (int, float)) and gates[name] >= 0, f"Promotion gate is invalid: {name}")
    _require(gates["latency_ms_p95"] <= gates["latency_limit_ms"], "Promotion gate failed: latency")
    _require(gates["memory_mb"] <= gates["memory_limit_mb"], "Promotion gate failed: memory")
    metrics = report.get("metrics", report)
    _require(isinstance(metrics, dict), "Test report metrics are invalid")
    baseline = manifest.get("baseline")
    _require(isinstance(baseline, dict), "Manifest needs baseline metrics")
    _require(float(metrics.get("mrr", 0.0)) > float(baseline.get("mrr", float("inf"))), "Promotion gate failed: MRR")
    _require(float(metrics.get("hits_at_5", 0.0)) > float(baseline.get("hits_at_5", float("inf"))), "Promotion gate failed: Hit@5")
    _require(float(metrics.get("hits_at_10", 0.0)) >= float(baseline.get("hits_at_10", float("inf"))), "Promotion gate failed: Hit@10")


def promote(source: Path, report_path: Path, manifest_path: Path, output: Path) -> None:
    source, report_path, manifest_path, output = map(Path, (source, report_path, manifest_path, output))
    _require(source.is_file() and report_path.is_file() and manifest_path.is_file(), "Source, report, and manifest files are required")
    _require(source.resolve() != output.resolve(), "Promotion output must not overwrite the source artifact")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    validate_manifest(manifest, source, report_path, report)
    with np.load(source, allow_pickle=False) as archive:
        payload = {name: np.asarray(archive[name]) for name in archive.files if name != "metadata"}
        metadata = json.loads(str(archive["metadata"].item()))
    _require(metadata.get("promoted") is False, "Source artifact must be an unpromoted offline artifact")
    metadata.update({
        "promoted": True,
        "promotion_run": manifest["promotion_run"],
        "promotion_manifest_sha256": sha256(manifest_path),
        "promotion_test_report_sha256": sha256(report_path),
    })
    payload["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{output.stem}.", suffix=".npz", dir=output.parent)
    os.close(handle)
    try:
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, output)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--test-report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    promote(args.source, args.test_report, args.manifest, args.output)
    print(json.dumps({"output": str(args.output), "promoted": True}, indent=2))


if __name__ == "__main__":
    main()
