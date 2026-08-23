"""Course recommendation node — Coursera search."""

from __future__ import annotations

import time
from typing import Any

from agents.state import CareerAgentState


def courses_node(state: CareerAgentState, *, G, **kwargs) -> dict[str, Any]:
    """Fetch Coursera course recommendations for anchor roles."""
    from src.inference_pipeline import fetch_coursera_courses

    t0 = time.time()
    anchor_ids = state.get("anchor_ids", [])
    user_type = state.get("user_type", "professional")

    courses = fetch_coursera_courses(anchor_ids, G, user_type)

    return {
        "courses": courses,
        "metadata": {**state.get("metadata", {}), "courses_ms": int((time.time() - t0) * 1000)},
    }
