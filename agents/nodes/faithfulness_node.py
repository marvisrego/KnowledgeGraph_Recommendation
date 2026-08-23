"""Faithfulness verification node — checks LLM output against graph."""

from __future__ import annotations

import time
from typing import Any

from agents.state import CareerAgentState


def faithfulness_node(state: CareerAgentState, *, G, **kwargs) -> dict[str, Any]:
    """Verify that LLM-cited entities exist in graph and are reachable."""
    from src.faithfulness import compute_faithfulness

    t0 = time.time()
    response = state.get("response", "")
    anchor_ids = state.get("anchor_ids", [])

    try:
        result = compute_faithfulness(response, anchor_ids, G)
        faithfulness = result.to_dict()
    except Exception:
        faithfulness = None

    return {
        "faithfulness": faithfulness,
        "metadata": {**state.get("metadata", {}), "faithfulness_ms": int((time.time() - t0) * 1000)},
    }
