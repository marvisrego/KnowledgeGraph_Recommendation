"""Portable NumPy inference for promoted history-aware transition rankers.

Training is deliberately offline-only.  A runtime artifact is accepted only
when its metadata declares a passed held-out promotion gate, so an unfinished
or rejected experiment cannot silently influence recommendations.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def _normalise(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else vector


@dataclass(frozen=True)
class SequentialPrediction:
    role_id: str
    score: float
    model: str
    history_length: int


class SequentialRankerRuntime:
    """Small, dependency-free scorer for promoted sequential-ranker artifacts."""

    def __init__(self, artifact_path: Path | str, *, require_promoted: bool = True) -> None:
        path = Path(artifact_path)
        if not path.is_file():
            raise FileNotFoundError(f"Sequential ranking artifact not found: {path}")
        with np.load(path, allow_pickle=False) as archive:
            if "metadata" not in archive or "candidate_ids" not in archive or "candidate_embeddings" not in archive:
                raise ValueError("Sequential ranking artifact has required fields missing")
            raw_metadata = archive["metadata"]
            metadata_text = str(raw_metadata.item() if raw_metadata.ndim == 0 else raw_metadata[0])
            self.metadata = json.loads(metadata_text)
            self.candidate_ids = tuple(str(item) for item in archive["candidate_ids"].tolist())
            self.candidate_embeddings = np.asarray(archive["candidate_embeddings"], dtype=np.float64)
            self.parameters = {
                key: np.asarray(archive[key], dtype=np.float64)
                for key in archive.files
                if key not in {"metadata", "candidate_ids", "candidate_embeddings"}
            }

        if require_promoted and not self.metadata.get("promoted", False):
            raise ValueError("Sequential ranking artifact is not marked promoted")
        self.model_type = str(self.metadata.get("model_type", ""))
        if self.model_type not in {
            "recency_mlp", "step_gru", "causal_transformer", "causal_transformer_id_residual",
        }:
            raise ValueError(f"Unsupported sequential model type: {self.model_type!r}")
        if self.candidate_embeddings.ndim != 2 or len(self.candidate_ids) != len(self.candidate_embeddings):
            raise ValueError("Sequential candidate matrix does not match candidate IDs")
        if not self.candidate_ids or not np.isfinite(self.candidate_embeddings).all():
            raise ValueError("Sequential candidate embeddings are invalid")
        self.candidate_embeddings = np.stack(
            [_normalise(row) for row in self.candidate_embeddings]
        )
        self._row_by_id = {role_id: index for index, role_id in enumerate(self.candidate_ids)}
        if len(self._row_by_id) != len(self.candidate_ids):
            raise ValueError("Sequential candidate IDs must be unique")
        self._validate_parameters()

    def _require(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
        value = self.parameters.get(name)
        if value is None or value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid sequential parameter {name!r}; expected {shape}")
        return value

    def _validate_parameters(self) -> None:
        dimension = int(self.candidate_embeddings.shape[1])
        if self.model_type in {"causal_transformer", "causal_transformer_id_residual"}:
            self._validate_causal_parameters(dimension)
            return
        if self.model_type == "recency_mlp":
            hidden = int(self.metadata.get("hidden_size", 64))
            self._require("w1", (dimension, hidden))
            self._require("b1", (hidden,))
            self._require("w_out", (hidden, dimension))
            self._require("b_out", (dimension,))
            return
        hidden = int(self.metadata.get("hidden_size", 64))
        for gate in ("z", "r", "h"):
            self._require(f"w_{gate}", (dimension, hidden))
            self._require(f"u_{gate}", (hidden, hidden))
            self._require(f"b_{gate}", (hidden,))
        self._require("attention_w", (hidden,))
        self._require("w_out", (hidden, dimension))
        self._require("b_out", (dimension,))
        tau = float(self.metadata.get("temperature", 0.07))
        if not 1e-3 <= tau <= 1.0:
            raise ValueError("Sequential temperature must be in [1e-3, 1]")

    def _validate_causal_parameters(self, dimension: int) -> None:
        """Validate the exact evaluation graph exported by CausalTransformerRanker."""
        config = self.metadata.get("model_config")
        if not isinstance(config, dict):
            raise ValueError("Causal Transformer artifact lacks model_config")
        hidden = int(config.get("hidden_size", 0))
        heads = int(config.get("attention_heads", 0))
        layers = int(config.get("layers", 0))
        maximum_history = int(config.get("maximum_history_length", 0))
        residual = bool(config.get("id_residual", False))
        if hidden < 1 or heads < 1 or layers < 1 or maximum_history < 1 or hidden % heads:
            raise ValueError("Causal Transformer model_config is invalid")
        if residual != (self.model_type == "causal_transformer_id_residual"):
            raise ValueError("Causal Transformer residual configuration does not match model type")
        if config.get("context_feature_sizes") not in (None, []):
            raise ValueError("Context-feature causal artifacts are not deployable without request features")
        self._causal_hidden = hidden
        self._causal_heads = heads
        self._causal_layers = layers
        self._causal_maximum_history = maximum_history
        self._causal_residual = residual
        self._require("input_projection.weight", (hidden, dimension))
        self._require("input_projection.bias", (hidden,))
        self._require("position_embeddings.weight", (maximum_history, hidden))
        self._require("output_projection.weight", (dimension, hidden))
        self._require("output_projection.bias", (dimension,))
        self._require("log_temperature", ())
        expected = {
            "input_projection.weight", "input_projection.bias", "position_embeddings.weight",
            "output_projection.weight", "output_projection.bias", "log_temperature",
        }
        if residual:
            self._require("input_residual", (len(self.candidate_ids), hidden))
            self._require("output_residual", (len(self.candidate_ids), dimension))
            self._require("output_bias", (len(self.candidate_ids),))
            expected.update({"input_residual", "output_residual", "output_bias"})
        for layer in range(layers):
            prefix = f"encoder.layers.{layer}"
            shapes = {
                "self_attn.in_proj_weight": (hidden * 3, hidden),
                "self_attn.in_proj_bias": (hidden * 3,),
                "self_attn.out_proj.weight": (hidden, hidden),
                "self_attn.out_proj.bias": (hidden,),
                "linear1.weight": (hidden * 2, hidden),
                "linear1.bias": (hidden * 2,),
                "linear2.weight": (hidden, hidden * 2),
                "linear2.bias": (hidden,),
                "norm1.weight": (hidden,), "norm1.bias": (hidden,),
                "norm2.weight": (hidden,), "norm2.bias": (hidden,),
            }
            for suffix, shape in shapes.items():
                name = f"{prefix}.{suffix}"
                self._require(name, shape)
                expected.add(name)
        unexpected = sorted(set(self.parameters).difference(expected))
        if unexpected:
            raise ValueError(f"Causal Transformer artifact has unexpected parameters: {', '.join(unexpected[:5])}")

    def _encode_history(self, history_rows: list[int]) -> np.ndarray:
        vectors = self.candidate_embeddings[history_rows]
        if self.model_type == "recency_mlp":
            weights = np.arange(1, len(vectors) + 1, dtype=np.float64)
            pooled = (vectors * weights[:, None]).sum(axis=0) / float(weights.sum())
            hidden = np.tanh(pooled @ self.parameters["w1"] + self.parameters["b1"])
            return _normalise(hidden @ self.parameters["w_out"] + self.parameters["b_out"])

        hidden_size = int(self.metadata.get("hidden_size", 64))
        hidden = np.zeros(hidden_size, dtype=np.float64)
        states: list[np.ndarray] = []
        for vector in vectors:
            update = _sigmoid(vector @ self.parameters["w_z"] + hidden @ self.parameters["u_z"] + self.parameters["b_z"])
            reset = _sigmoid(vector @ self.parameters["w_r"] + hidden @ self.parameters["u_r"] + self.parameters["b_r"])
            candidate = np.tanh(
                vector @ self.parameters["w_h"]
                + (reset * hidden) @ self.parameters["u_h"]
                + self.parameters["b_h"]
            )
            hidden = (1.0 - update) * hidden + update * candidate
            states.append(hidden.copy())
        state_matrix = np.stack(states)
        attention_logits = state_matrix @ self.parameters["attention_w"]
        attention_logits -= float(attention_logits.max())
        attention = np.exp(attention_logits)
        attention /= float(attention.sum())
        pooled = attention @ state_matrix
        return _normalise(pooled @ self.parameters["w_out"] + self.parameters["b_out"])

    @staticmethod
    def _layer_norm(values: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
        mean = values.mean(axis=-1, keepdims=True)
        variance = ((values - mean) ** 2).mean(axis=-1, keepdims=True)
        return (values - mean) / np.sqrt(variance + 1e-5) * weight + bias

    @staticmethod
    def _gelu(values: np.ndarray) -> np.ndarray:
        # PyTorch TransformerEncoderLayer's default GELU approximation is exact.
        erf = np.fromiter((math.erf(float(value) / math.sqrt(2.0)) for value in values.flat), dtype=np.float64,
                          count=values.size).reshape(values.shape)
        return 0.5 * values * (1.0 + erf)

    @staticmethod
    def _softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
        shifted = values - values.max(axis=axis, keepdims=True)
        exponent = np.exp(shifted)
        return exponent / exponent.sum(axis=axis, keepdims=True)

    def _causal_attention(self, values: np.ndarray, prefix: str) -> np.ndarray:
        hidden, heads = self._causal_hidden, self._causal_heads
        head_dimension = hidden // heads
        qkv = values @ self.parameters[f"{prefix}.self_attn.in_proj_weight"].T
        qkv += self.parameters[f"{prefix}.self_attn.in_proj_bias"]
        query, key, value = np.split(qkv, 3, axis=-1)
        query = query.reshape(len(values), heads, head_dimension).transpose(1, 0, 2)
        key = key.reshape(len(values), heads, head_dimension).transpose(1, 0, 2)
        value = value.reshape(len(values), heads, head_dimension).transpose(1, 0, 2)
        scores = query @ key.transpose(0, 2, 1) / math.sqrt(head_dimension)
        scores[:, np.triu_indices(len(values), k=1)[0], np.triu_indices(len(values), k=1)[1]] = -np.inf
        attended = self._softmax(scores) @ value
        attended = attended.transpose(1, 0, 2).reshape(len(values), hidden)
        return attended @ self.parameters[f"{prefix}.self_attn.out_proj.weight"].T + self.parameters[f"{prefix}.self_attn.out_proj.bias"]

    def _encode_causal_history(self, history_rows: list[int]) -> np.ndarray:
        rows = history_rows[-self._causal_maximum_history:]
        encoded = self.candidate_embeddings[rows] @ self.parameters["input_projection.weight"].T
        encoded += self.parameters["input_projection.bias"]
        encoded += self.parameters["position_embeddings.weight"][:len(rows)]
        if self._causal_residual:
            encoded += self.parameters["input_residual"][rows]
        for layer in range(self._causal_layers):
            prefix = f"encoder.layers.{layer}"
            normalized = self._layer_norm(
                encoded, self.parameters[f"{prefix}.norm1.weight"], self.parameters[f"{prefix}.norm1.bias"],
            )
            encoded = encoded + self._causal_attention(normalized, prefix)
            normalized = self._layer_norm(
                encoded, self.parameters[f"{prefix}.norm2.weight"], self.parameters[f"{prefix}.norm2.bias"],
            )
            feedforward = normalized @ self.parameters[f"{prefix}.linear1.weight"].T
            feedforward += self.parameters[f"{prefix}.linear1.bias"]
            feedforward = self._gelu(feedforward)
            feedforward = feedforward @ self.parameters[f"{prefix}.linear2.weight"].T
            encoded = encoded + feedforward + self.parameters[f"{prefix}.linear2.bias"]
        prediction = encoded[-1] @ self.parameters["output_projection.weight"].T
        prediction += self.parameters["output_projection.bias"]
        return _normalise(prediction)

    def rank(self, history_role_ids: list[str] | tuple[str, ...], limit: int) -> list[SequentialPrediction]:
        """Rank all artifact candidates, omitting the current (last) role."""
        if limit <= 0:
            return []
        rows = [self._row_by_id[role_id] for role_id in history_role_ids if role_id in self._row_by_id]
        if not rows:
            return []
        if self.model_type in {"causal_transformer", "causal_transformer_id_residual"}:
            prediction = self._encode_causal_history(rows)
            temperature = float(np.clip(np.exp(self.parameters["log_temperature"].item()), 1e-3, 1.0))
            candidates = self.candidate_embeddings
            if self._causal_residual:
                candidates = np.stack([_normalise(vector) for vector in candidates + self.parameters["output_residual"]])
            scores = (candidates @ prediction) / temperature
            if self._causal_residual:
                scores = scores + self.parameters["output_bias"]
        else:
            prediction = self._encode_history(rows)
            temperature = float(self.metadata.get("temperature", 0.07))
            scores = (self.candidate_embeddings @ prediction) / temperature
        current_role_id = str(history_role_ids[-1]) if history_role_ids else ""
        ordered = sorted(
            (
                (role_id, float(scores[index]))
                for index, role_id in enumerate(self.candidate_ids)
                if role_id != current_role_id
            ),
            key=lambda item: (-item[1], item[0]),
        )
        return [
            SequentialPrediction(role_id, score, self.model_type, len(rows))
            for role_id, score in ordered[:limit]
        ]

    def fusion_weight(self, history_length: int) -> float:
        key = "fusion_weight_multi" if history_length >= 2 else "fusion_weight_single"
        value = float(self.metadata.get(key, 0.0))
        return min(1.0, max(0.0, value))

    def diagnostics(self) -> dict[str, object]:
        return {
            "model_type": self.model_type,
            "candidate_roles": len(self.candidate_ids),
            "embedding_dimension": int(self.candidate_embeddings.shape[1]),
            "maximum_history_length": getattr(self, "_causal_maximum_history", None),
            "promotion_run": self.metadata.get("promotion_run"),
            "fusion_weight_single": self.fusion_weight(1),
            "fusion_weight_multi": self.fusion_weight(2),
        }


def build_sequential_ranking_runtime(path: Path | str) -> SequentialRankerRuntime:
    """Factory kept symmetrical with the LightGBM runtime loader."""
    return SequentialRankerRuntime(path)
