"""LangGraph StateGraph definition for the Career Advisor workflow.

Conditional routing:
- has_context=True  → full pipeline (retrieval → ranking → traversal → effort → generation → faithfulness → explanation → courses)
- has_context=False → partial pipeline (retrieval → explore data only)
"""

from __future__ import annotations

from functools import partial
from typing import Any

from langgraph.graph import END, StateGraph

from agents.state import CareerAgentState
from agents.nodes.intent import intent_node
from agents.nodes.retrieval import retrieval_node
from agents.nodes.ranking import ranking_node
from agents.nodes.qualification_node import qualification_node
from agents.nodes.traversal import traversal_node
from agents.nodes.effort import effort_node
from agents.nodes.skill_gap_node import skill_gap_node
from agents.nodes.learning_plan_node import learning_plan_node
from agents.nodes.generation import generation_node
from agents.nodes.courses import courses_node
from agents.nodes.faithfulness_node import faithfulness_node
from agents.nodes.explanation import explanation_node


def _route_after_intent(state: CareerAgentState) -> str:
    """Conditional edge: route based on has_context."""
    if state.get("has_context", True):
        return "retrieval"
    return "retrieval_partial"


def _route_after_traversal(state: CareerAgentState) -> str:
    """Stop partial-context requests before effort and LLM generation."""
    return "effort" if state.get("has_context", True) else "explore"


def _build_explore(state: CareerAgentState, *, G, **kwargs) -> dict[str, Any]:
    """Build explore data for partial-context responses."""
    from src.inference_pipeline import _build_explore_data

    anchor_ids = state.get("anchor_ids", [])
    triples = state.get("triples", [])
    intent = state.get("intent", {})

    explore_data = _build_explore_data(anchor_ids, triples, G) if anchor_ids else {}
    message = intent.get("followup_questions", "Could you tell me more about your background and goals?")

    return {
        "explore_data": explore_data,
        "response": message,
    }


def build_career_graph(
    settings,
    G,
    collection,
    transition_smoother=None,
    link_prediction_runtime=None,
) -> StateGraph:
    """Construct the LangGraph workflow with injected dependencies.

    Returns a compiled graph ready to invoke.
    """
    # Bind dependencies to node functions via partial
    intent = partial(intent_node, settings=settings, G=G)
    retrieval = partial(
        retrieval_node,
        settings=settings,
        G=G,
        collection=collection,
        transition_smoother=transition_smoother,
        link_prediction_runtime=link_prediction_runtime,
    )
    ranking = partial(ranking_node, settings=settings, G=G)
    qualification = partial(qualification_node, settings=settings, G=G)
    traversal = partial(traversal_node, settings=settings, G=G)
    effort = partial(effort_node, settings=settings, G=G, transition_smoother=transition_smoother)
    skill_gap = partial(skill_gap_node, G=G, settings=settings)
    learning_plan = partial(learning_plan_node, G=G)
    generation = partial(generation_node, settings=settings, G=G)
    courses = partial(courses_node, G=G)
    faithfulness = partial(faithfulness_node, G=G)
    explanation = partial(explanation_node, G=G)
    explore = partial(_build_explore, G=G)

    # Build the state graph
    workflow = StateGraph(CareerAgentState)

    # Add nodes
    workflow.add_node("intent", intent)
    workflow.add_node("retrieval", retrieval)
    workflow.add_node("retrieval_partial", retrieval)
    workflow.add_node("ranking", ranking)
    workflow.add_node("qualification", qualification)
    workflow.add_node("traversal", traversal)
    workflow.add_node("effort", effort)
    workflow.add_node("skill_gap", skill_gap)
    workflow.add_node("learning_plan", learning_plan)
    workflow.add_node("generation", generation)
    workflow.add_node("courses", courses)
    workflow.add_node("faithfulness", faithfulness)
    workflow.add_node("explanation", explanation)
    workflow.add_node("explore", explore)

    # Set entry point
    workflow.set_entry_point("intent")

    # Conditional routing after intent
    workflow.add_conditional_edges("intent", _route_after_intent, {
        "retrieval": "retrieval",
        "retrieval_partial": "retrieval_partial",
    })

    # Full context path
    workflow.add_edge("retrieval", "ranking")
    workflow.add_edge("ranking", "qualification")
    workflow.add_edge("qualification", "traversal")
    workflow.add_conditional_edges("traversal", _route_after_traversal, {
        "effort": "effort",
        "explore": "explore",
    })
    workflow.add_edge("effort", "skill_gap")
    workflow.add_edge("skill_gap", "courses")
    workflow.add_edge("courses", "learning_plan")
    workflow.add_edge("learning_plan", "generation")
    workflow.add_edge("generation", "faithfulness")
    workflow.add_edge("faithfulness", "explanation")
    workflow.add_edge("explanation", END)

    # Partial context path
    workflow.add_edge("retrieval_partial", "ranking")
    workflow.add_edge("explore", END)

    return workflow.compile()


def run_career_workflow(
    query: str,
    messages: list[dict],
    settings,
    G,
    collection,
    transition_smoother=None,
    link_prediction_runtime=None,
) -> dict:
    """Execute the full career advisor workflow and return the response dict.

    This is the LangGraph equivalent of run_query().
    """
    import time

    t0 = time.time()
    app = build_career_graph(
        settings,
        G,
        collection,
        transition_smoother,
        link_prediction_runtime,
    )

    initial_state: CareerAgentState = {
        "query": query,
        "messages": messages,
        "errors": [],
        "metadata": {},
    }

    result = app.invoke(initial_state)
    total_ms = int((time.time() - t0) * 1000)

    # Build output compatible with the existing API response format
    if result.get("explore_data"):
        return {
            "message": result.get("response", ""),
            "courses": [],
            "path": {},
            "explore": result.get("explore_data", {}),
            "evidence": result.get("public_evidence", {}),
            "faithfulness": None,
            "explanations": [],
            "metadata": {**result.get("metadata", {}), "total_ms": total_ms},
        }

    return {
        "message": result.get("response", ""),
        "courses": result.get("courses", []),
        "path": result.get("path_data", {}),
        "explore": {},
        "evidence": result.get("public_evidence", {}),
        "faithfulness": result.get("faithfulness"),
        "explanations": result.get("explanations", []),
        "skill_gap_analysis": result.get("skill_gap_analysis", []),
        "learning_plan": result.get("learning_plan", []),
        "metadata": {**result.get("metadata", {}), "total_ms": total_ms},
    }
