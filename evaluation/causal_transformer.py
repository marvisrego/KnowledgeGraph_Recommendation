"""Offline-only causal Transformer used for full-candidate career ranking.

The module deliberately accepts already-resolved prefix role IDs.  It has no
access to person IDs, split labels, target roles, or future trajectory steps.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CausalTransformerRanker(nn.Module):
    """A SASRec-style ranker with optional trainable role-ID residuals.

    The frozen ESCO candidate vectors remain the semantic prior.  When
    ``id_residual`` is enabled, trainable input/output residuals and a
    destination bias add occupation-specific transition capacity without
    changing the full-candidate or causal-ranking contract.
    """

    def __init__(
        self,
        candidate_embeddings: torch.Tensor,
        *,
        maximum_history_length: int,
        hidden_size: int = 128,
        attention_heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
        id_residual: bool = False,
        context_feature_sizes: tuple[int, int, int] | None = None,
    ) -> None:
        super().__init__()
        if maximum_history_length < 1:
            raise ValueError("maximum_history_length must be positive")
        if hidden_size % attention_heads:
            raise ValueError("hidden_size must divide evenly across attention_heads")
        self.maximum_history_length = maximum_history_length
        self.hidden_size = hidden_size
        self.attention_heads = attention_heads
        self.layers = layers
        self.id_residual = bool(id_residual)
        self.context_feature_sizes = tuple(context_feature_sizes) if context_feature_sizes else None
        self.register_buffer("candidate", candidate_embeddings)
        dimension = int(candidate_embeddings.shape[1])
        self.input_projection = nn.Linear(dimension, hidden_size)
        self.position_embeddings = nn.Embedding(maximum_history_length, hidden_size)
        if self.id_residual:
            self.input_residual = nn.Parameter(torch.zeros((len(candidate_embeddings), hidden_size)))
            self.output_residual = nn.Parameter(torch.zeros_like(candidate_embeddings))
            self.output_bias = nn.Parameter(torch.zeros(len(candidate_embeddings)))
        if self.context_feature_sizes:
            if len(self.context_feature_sizes) != 3 or any(size < 1 for size in self.context_feature_sizes):
                raise ValueError("context_feature_sizes must contain three positive sizes")
            self.tenure_embeddings = nn.Embedding(self.context_feature_sizes[0], hidden_size)
            self.gap_embeddings = nn.Embedding(self.context_feature_sizes[1], hidden_size)
            self.education_embeddings = nn.Embedding(self.context_feature_sizes[2], hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=attention_heads,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.output_projection = nn.Linear(hidden_size, dimension)
        self.log_temperature = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(0.07)))))

    def _batch(self, histories: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        if not histories or any(not history for history in histories):
            raise ValueError("Each causal Transformer example needs a non-empty history")
        length = min(max(len(history) for history in histories), self.maximum_history_length)
        device = self.candidate.device
        indices = torch.zeros((len(histories), length), dtype=torch.long, device=device)
        valid = torch.zeros((len(histories), length), dtype=torch.bool, device=device)
        for row, history in enumerate(histories):
            recent = history[-length:]
            indices[row, :len(recent)] = torch.tensor(recent, dtype=torch.long, device=device)
            valid[row, :len(recent)] = True
        return indices, valid

    def _context_features(self, features: list[tuple[int, int, int]] | None, batch_size: int) -> torch.Tensor | None:
        if self.context_feature_sizes is None:
            if features is not None:
                raise ValueError("This ranker was not configured with context features")
            return None
        if features is None or len(features) != batch_size:
            raise ValueError("Each history needs one (tenure, gap, education) feature tuple")
        values = torch.tensor(features, dtype=torch.long, device=self.candidate.device)
        if values.shape != (batch_size, 3):
            raise ValueError("Context features must have shape (batch, 3)")
        limits = torch.tensor(self.context_feature_sizes, device=self.candidate.device)
        if torch.any(values < 0) or torch.any(values >= limits):
            raise ValueError("Context feature value is outside its configured vocabulary")
        return (
            self.tenure_embeddings(values[:, 0])
            + self.gap_embeddings(values[:, 1])
            + self.education_embeddings(values[:, 2])
        )

    def forward(
        self,
        histories: list[list[int]],
        context_features: list[tuple[int, int, int]] | None = None,
    ) -> torch.Tensor:
        indices, valid = self._batch(histories)
        length = indices.shape[1]
        positions = torch.arange(length, device=self.candidate.device).unsqueeze(0)
        encoded_input = self.input_projection(self.candidate[indices]) + self.position_embeddings(positions)
        if self.id_residual:
            encoded_input = encoded_input + self.input_residual[indices]
        # True entries are forbidden: a state can attend only to itself and its past.
        causal_mask = torch.triu(
            torch.ones((length, length), dtype=torch.bool, device=self.candidate.device), diagonal=1,
        )
        encoded = self.encoder(
            encoded_input,
            mask=causal_mask,
            src_key_padding_mask=~valid,
        )
        final_indices = valid.sum(dim=1) - 1
        representation = encoded[torch.arange(len(histories), device=self.candidate.device), final_indices]
        context = self._context_features(context_features, len(histories))
        if context is not None:
            representation = representation + context
        predicted = F.normalize(self.output_projection(representation), dim=1)
        temperature = torch.clamp(torch.exp(self.log_temperature), 1e-3, 1.0)
        candidates = self.candidate
        if self.id_residual:
            candidates = F.normalize(candidates + self.output_residual, dim=1)
        logits = predicted @ candidates.T / temperature
        if self.id_residual:
            logits = logits + self.output_bias
        current_rows = indices[torch.arange(len(histories), device=self.candidate.device), final_indices]
        logits[torch.arange(len(histories), device=self.candidate.device), current_rows] = float("-inf")
        return logits

    def export(self) -> dict[str, torch.Tensor]:
        """Exclude the fixed candidate buffer duplicated in the artifact header."""
        return {name: value for name, value in self.state_dict().items() if name != "candidate"}

    def configuration(self) -> dict[str, object]:
        return {
            "hidden_size": self.hidden_size,
            "attention_heads": self.attention_heads,
            "layers": self.layers,
            "maximum_history_length": self.maximum_history_length,
            "id_residual": self.id_residual,
            "context_feature_sizes": self.context_feature_sizes,
        }
