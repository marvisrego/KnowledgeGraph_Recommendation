"""Generation node — LLM response grounded in graph context."""

from __future__ import annotations

import time
from typing import Any

from agents.state import CareerAgentState


def generation_node(state: CareerAgentState, *, settings, G, **kwargs) -> dict[str, Any]:
    """Generate a grounded 2-3 sentence response using the LLM."""
    from src.inference_pipeline import generate_response

    t0 = time.time()
    query = state["query"]
    anchor_ids = state.get("anchor_ids", [])
    triples = state.get("triples", [])
    user_type = state.get("user_type", "professional")
    history = state.get("messages")

    if not triples:
        return {"response": "I found some matching roles but could not retrieve supporting details from the knowledge graph."}

    try:
        response = generate_response(query, anchor_ids, triples, G, settings, user_type, history=history)
    except RuntimeError as exc:
        response = f"I encountered an error generating a response: {exc}"

    return {
        "response": response,
        "metadata": {**state.get("metadata", {}), "generation_ms": int((time.time() - t0) * 1000)},
    }
