"""Conservative resolution of explicitly stated, ordered career history."""

from __future__ import annotations

from typing import Iterable

import networkx as nx

from src.skill_gap import resolve_current_role


def resolve_career_history(
    raw_history: object,
    current_role_id: str | None,
    graph: nx.MultiDiGraph,
    maximum_length: int = 10,
) -> list[str]:
    """Resolve role phrases while preserving order and avoiding invented history.

    Only router-provided strings are considered. Consecutive duplicate roles are
    removed; a known current role is appended when the router omitted it from a
    history, ensuring the final item is the current role used for prediction.
    """
    if maximum_length < 1:
        raise ValueError("maximum_length must be positive")
    phrases: Iterable[object]
    if isinstance(raw_history, str):
        phrases = [raw_history]
    elif isinstance(raw_history, list):
        phrases = raw_history
    else:
        phrases = []

    resolved: list[str] = []
    for phrase in phrases:
        if not isinstance(phrase, str) or not phrase.strip():
            continue
        role_id = resolve_current_role(phrase, graph)
        if role_id and (not resolved or resolved[-1] != role_id):
            resolved.append(str(role_id))

    if current_role_id and (not resolved or resolved[-1] != current_role_id):
        resolved.append(str(current_role_id))
    return resolved[-maximum_length:]
