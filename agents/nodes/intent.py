"""Intent routing node — classifies user type and determines context sufficiency."""

from __future__ import annotations

import time
from typing import Any

from agents.state import CareerAgentState


def intent_node(state: CareerAgentState, *, settings, G, **kwargs) -> dict[str, Any]:
    """Run intent routing via LLM. Extracts user type, skills, role, and goal."""
    from src.inference_pipeline import route_intent, _user_authored_texts

    t0 = time.time()
    query = state["query"]
    history = state.get("messages")

    try:
        intent = route_intent(query, settings, history=history)
    except RuntimeError as exc:
        intent = {
            "has_context": True,
            "user_type": "professional",
            "current_role": "",
            "skills": [],
            "career_goal": "",
        }

    user_type = intent.get("user_type", "professional")
    if user_type not in ("student", "professional"):
        user_type = "professional"

    return {
        "intent": intent,
        "user_type": user_type,
        "has_context": intent.get("has_context", True),
        "metadata": {**state.get("metadata", {}), "intent_ms": int((time.time() - t0) * 1000)},
    }
