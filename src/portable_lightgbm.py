"""Dependency-free inference for numeric LightGBM models exported as JSON.

Training and evaluation can continue to use LightGBM. The deployed runtime only
needs NumPy and this evaluator, which avoids Vercel's unavailable libgomp.so.1.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


class PortableLightGBMBooster:
    """Evaluate a binary LightGBM tree ensemble from ``Booster.dump_model``."""

    def __init__(self, model: dict[str, Any]):
        if int(model.get("num_class", 1)) != 1:
            raise ValueError("Only single-output LightGBM models are supported")
        self._model = model
        self._trees = [entry["tree_structure"] for entry in model.get("tree_info", [])]
        self._feature_count = len(model.get("feature_names", []))
        if not self._trees or self._feature_count < 1:
            raise ValueError("Portable LightGBM model is missing trees or feature names")

        objective = str(model.get("objective", ""))
        if not objective.startswith("binary"):
            raise ValueError(f"Unsupported LightGBM objective: {objective or 'missing'}")
        parts = objective.split("sigmoid:", 1)
        self._sigmoid = float(parts[1].split()[0]) if len(parts) == 2 else 1.0

    @classmethod
    def from_file(cls, path: Path) -> "PortableLightGBMBooster":
        with Path(path).open("r", encoding="utf-8") as model_file:
            return cls(json.load(model_file))

    def num_feature(self) -> int:
        return self._feature_count

    @staticmethod
    def _go_left(node: dict[str, Any], value: float) -> bool:
        missing_type = str(node.get("missing_type", "None"))
        is_missing = math.isnan(value) or (missing_type == "Zero" and value == 0.0)
        if is_missing:
            return bool(node.get("default_left", True))

        decision_type = str(node.get("decision_type", "<="))
        threshold = node.get("threshold")
        if decision_type == "<=":
            return value <= float(threshold)
        if decision_type == "==":
            categories = {int(item) for item in str(threshold).split("||") if item}
            return int(value) in categories
        raise ValueError(f"Unsupported LightGBM decision type: {decision_type}")

    @classmethod
    def _tree_value(cls, node: dict[str, Any], row: np.ndarray) -> float:
        while "leaf_value" not in node:
            feature = int(node["split_feature"])
            node = node["left_child"] if cls._go_left(node, float(row[feature])) else node["right_child"]
        return float(node["leaf_value"])

    def predict(self, data: object) -> np.ndarray:
        rows = np.asarray(data, dtype=np.float64)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        if rows.ndim != 2 or rows.shape[1] != self._feature_count:
            raise ValueError(
                f"Expected a 2D feature matrix with {self._feature_count} columns"
            )

        raw_scores = np.zeros(rows.shape[0], dtype=np.float64)
        for tree in self._trees:
            raw_scores += np.fromiter(
                (self._tree_value(tree, row) for row in rows),
                dtype=np.float64,
                count=rows.shape[0],
            )
        scaled = np.clip(self._sigmoid * raw_scores, -709.0, 709.0)
        return 1.0 / (1.0 + np.exp(-scaled))
