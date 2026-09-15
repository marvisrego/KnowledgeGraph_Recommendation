"""Portable, train-only calibration artifacts for Transition Effort Score.

The runtime deliberately does not train models.  Offline evaluation writes a
small JSON artifact with non-negative component weights and validation-derived
effort-band thresholds; this module validates and loads that artifact safely.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


COMPONENT_NAMES = ("skill_gap", "domain", "empirical", "transferability")


@dataclass(frozen=True)
class TESCalibration:
    """Validated calibration values used by :mod:`src.transition_effort`."""

    weights: dict[str, float]
    low_max: float
    moderate_max: float
    version: str = "trajectory_proxy_v1"
    method: str = "trajectory_choice_proxy"
    limitation: str = (
        "Observed career transitions are a mobility-likelihood proxy, not a causal "
        "measure of an individual's effort or ability."
    )
    training_hash: str | None = None

    def __post_init__(self) -> None:
        missing = set(COMPONENT_NAMES) - set(self.weights)
        if missing:
            raise ValueError(f"TES calibration is missing weights: {sorted(missing)}")
        values = [float(self.weights[name]) for name in COMPONENT_NAMES]
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("TES calibration weights must be finite and non-negative")
        total = sum(values)
        if total <= 0.0 or abs(total - 1.0) > 1e-6:
            raise ValueError(f"TES calibration weights must sum to 1.0, got {total:.8f}")
        if not (0.0 <= self.low_max < self.moderate_max <= 1.0):
            raise ValueError("TES thresholds must satisfy 0 <= low_max < moderate_max <= 1")

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "method": self.method,
            "weights": {name: float(self.weights[name]) for name in COMPONENT_NAMES},
            "thresholds": {"low_max": self.low_max, "moderate_max": self.moderate_max},
            "limitation": self.limitation,
            "training_hash": self.training_hash,
        }


def default_calibration() -> TESCalibration:
    """Return the legacy values as an explicit, transparently labelled fallback."""
    return TESCalibration(
        weights={
            "skill_gap": 0.35,
            "domain": 0.15,
            "empirical": 0.25,
            "transferability": 0.25,
        },
        low_max=0.30,
        moderate_max=0.60,
        version="legacy_heuristic_v1",
        method="heuristic_fallback",
        limitation="No validated calibration artifact was available; legacy heuristic values are in use.",
    )


def load_calibration(path: Path | str | None) -> TESCalibration:
    """Load a calibration artifact, safely falling back when it is absent/invalid."""
    if not path:
        return default_calibration()
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        threshold = payload.get("thresholds", {})
        return TESCalibration(
            weights={name: float(payload["weights"][name]) for name in COMPONENT_NAMES},
            low_max=float(threshold["low_max"]),
            moderate_max=float(threshold["moderate_max"]),
            version=str(payload.get("version", "trajectory_proxy_v1")),
            method=str(payload.get("method", "trajectory_choice_proxy")),
            limitation=str(payload.get("limitation", "")),
            training_hash=(str(payload["training_hash"]) if payload.get("training_hash") else None),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return default_calibration()


def write_calibration(calibration: TESCalibration, path: Path | str) -> None:
    """Atomically persist a validated calibration artifact."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(calibration.to_dict(), stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temp_name, destination)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def fit_nonnegative_pairwise_weights(
    positive_features: Sequence[Sequence[float]],
    negative_features: Sequence[Sequence[float]],
    sample_weights: Sequence[float] | None = None,
    l2: float = 1e-3,
    iterations: int = 500,
    learning_rate: float = 0.1,
) -> dict[str, float]:
    """Fit interpretable non-negative weights from paired transition choices.

    Each positive row is an observed transition and the matching negative row
    is a sampled alternative. Lower effort should favour the observed role.
    Projected gradient descent keeps coefficients non-negative without making
    SciPy a production dependency.
    """
    positives = np.asarray(positive_features, dtype=np.float64)
    negatives = np.asarray(negative_features, dtype=np.float64)
    if positives.ndim != 2 or positives.shape != negatives.shape or positives.shape[1] != 4:
        raise ValueError("positive_features and negative_features must be matching [n, 4] matrices")
    if len(positives) == 0:
        raise ValueError("At least one paired transition example is required")
    if not np.isfinite(positives).all() or not np.isfinite(negatives).all():
        raise ValueError("Calibration features must be finite")
    if l2 < 0.0 or iterations < 1 or learning_rate <= 0.0:
        raise ValueError("Invalid calibration optimizer settings")
    weights = np.ones(len(positives), dtype=np.float64)
    if sample_weights is not None:
        weights = np.asarray(sample_weights, dtype=np.float64)
        if weights.shape != (len(positives),) or not np.isfinite(weights).all() or np.any(weights <= 0.0):
            raise ValueError("sample_weights must be finite positive values for every pair")
    weights /= float(weights.sum())

    # log(1 + exp(beta dot (positive-negative))) penalizes a positive with
    # greater effort than its sampled alternative.
    difference = positives - negatives
    beta = np.full(4, 0.25, dtype=np.float64)
    for _ in range(iterations):
        logits = np.clip(difference @ beta, -40.0, 40.0)
        sigmoid = 1.0 / (1.0 + np.exp(-logits))
        gradient = difference.T @ (weights * sigmoid) + l2 * beta
        beta = np.maximum(0.0, beta - learning_rate * gradient)
        total = float(beta.sum())
        if total <= 1e-12:
            beta[:] = 0.25
        else:
            beta /= total

    return {name: float(beta[index]) for index, name in enumerate(COMPONENT_NAMES)}


def thresholds_from_scores(scores: Iterable[float]) -> tuple[float, float]:
    """Return validation-only 33rd/67th percentile band boundaries."""
    values = np.asarray(list(scores), dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        raise ValueError("At least three finite validation scores are required")
    low, moderate = np.quantile(values, (1.0 / 3.0, 2.0 / 3.0))
    low = float(np.clip(low, 0.0, 1.0))
    moderate = float(np.clip(moderate, 0.0, 1.0))
    if moderate <= low:
        moderate = min(1.0, low + 1e-6)
    return low, moderate
