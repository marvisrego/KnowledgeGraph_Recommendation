"""Shared eligibility rules for empirical Karrierewege transition evidence."""

from __future__ import annotations

from typing import Mapping, Any


TRANSITION_RELATION = "TRANSITIONS_TO"
TRANSITION_SOURCE = "karrierewege"


def is_training_transition(data: Mapping[str, Any]) -> bool:
    """Return whether an edge is eligible as observed runtime/training evidence.

    Older locally built graphs did not always persist a split marker. Those
    edges are treated as training evidence for backward compatibility. Explicit
    validation and test edges are never eligible.
    """
    return (
        data.get("relation") == TRANSITION_RELATION
        and data.get("source") == TRANSITION_SOURCE
        and data.get("split") in (None, "train")
    )
