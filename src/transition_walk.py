"""Leakage-safe multi-step probabilities over observed training transitions."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Mapping


ProbabilityIndex = Mapping[str, Mapping[str, float]]


def propagate_distribution(
    distribution: Mapping[str, float],
    transitions: ProbabilityIndex,
    *,
    excluded_role_id: str | None = None,
) -> dict[str, float]:
    """Propagate one probability step through a row-normalized transition map."""
    propagated: defaultdict[str, float] = defaultdict(float)
    excluded = str(excluded_role_id) if excluded_role_id is not None else None
    for intermediate_id, incoming_probability in distribution.items():
        incoming = float(incoming_probability)
        if incoming <= 0.0 or not math.isfinite(incoming):
            continue
        for destination_id, transition_probability in transitions.get(
            str(intermediate_id), {}
        ).items():
            destination = str(destination_id)
            probability = float(transition_probability)
            if destination == excluded:
                continue
            if probability <= 0.0 or not math.isfinite(probability):
                continue
            propagated[destination] += incoming * probability
    return dict(propagated)


def transition_walk_components(
    source_role_id: str,
    transitions: ProbabilityIndex,
    *,
    max_steps: int = 3,
) -> tuple[dict[str, float], ...]:
    """Return one- through ``max_steps``-step destination distributions.

    The caller supplies a transition index already restricted to eligible
    training evidence. The source role is excluded at every depth so cycles do
    not recommend returning to the current role.
    """
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    source = str(source_role_id)
    first = {
        str(destination): float(probability)
        for destination, probability in transitions.get(source, {}).items()
        if str(destination) != source
        and float(probability) > 0.0
        and math.isfinite(float(probability))
    }
    components = [first]
    current = first
    for _ in range(2, max_steps + 1):
        current = propagate_distribution(
            current,
            transitions,
            excluded_role_id=source,
        )
        components.append(current)
    return tuple(components)


def blend_walk_components(
    components: tuple[Mapping[str, float], ...],
    weights: tuple[float, ...],
) -> dict[str, float]:
    """Blend aligned walk components with finite, non-negative weights."""
    if len(components) != len(weights):
        raise ValueError("components and weights must have the same length")
    if not weights or any(weight < 0.0 or not math.isfinite(weight) for weight in weights):
        raise ValueError("weights must be finite and non-negative")
    if sum(weights) <= 0.0:
        raise ValueError("at least one weight must be positive")

    scores: defaultdict[str, float] = defaultdict(float)
    for component, weight in zip(components, weights):
        for destination_id, probability in component.items():
            scores[str(destination_id)] += weight * float(probability)
    return dict(scores)
