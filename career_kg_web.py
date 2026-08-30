from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path when called via gunicorn / vercel
sys.path.insert(0, str(Path(__file__).parent))

from config import Settings
from src.graph_store import load_graph
from src.embeddings_index import load_chroma_collection
from src.inference_pipeline import run_query
from src.transition_embedding import RuntimeTransitionSmoother, SmoothingConfig
from src.transition_policy import is_training_transition
from src.hybrid_retrieval import build_link_prediction_runtime


def create_app():
    flask = __import__("flask")
    Flask = flask.Flask
    jsonify = flask.jsonify
    render_template = flask.render_template
    request = flask.request

    # Serve React build from public_react/ if it exists, else fall back to legacy public/
    _react_dist = Path(__file__).resolve().parent / "public_react"
    _static_folder = str(_react_dist) if _react_dist.exists() else "public"
    app = Flask(__name__, template_folder="templates", static_folder=_static_folder, static_url_path="")
    app.config["JSON_SORT_KEYS"] = False

    settings = Settings.from_env(Path(__file__).resolve().parent)

    # --- Lazy-load graph and ChromaDB (once per process) ---
    _cache: dict[str, Any] = {
        "graph": None,
        "collection": None,
        "error": None,
        "transition_smoother": None,
        "transition_smoother_attempted": False,
        "transition_smoother_error": None,
        "link_prediction_runtime": None,
        "link_prediction_attempted": False,
        "link_prediction_error": None,
    }
    _resource_lock = threading.Lock()
    _smoother_lock = threading.Lock()
    _link_prediction_lock = threading.Lock()

    def _load_resources() -> tuple:
        if _cache["graph"] is None or _cache["collection"] is None:
            with _resource_lock:
                if _cache["graph"] is None:
                    try:
                        _cache["graph"] = load_graph(settings.graph_path)
                    except RuntimeError as exc:
                        _cache["error"] = str(exc)
                if _cache["collection"] is None:
                    try:
                        _cache["collection"] = load_chroma_collection(settings)
                    except RuntimeError as exc:
                        _cache["error"] = str(exc)
        return _cache["graph"], _cache["collection"]

    def _load_transition_smoother(G, collection):
        if not settings.transition_smoothing_enabled or G is None or collection is None:
            return None
        if _cache["transition_smoother_attempted"]:
            return _cache["transition_smoother"]
        with _smoother_lock:
            if _cache["transition_smoother_attempted"]:
                return _cache["transition_smoother"]
            try:
                config = SmoothingConfig(
                    neighbours=settings.transition_smoothing_neighbours,
                    direct_weight=settings.transition_smoothing_direct_weight,
                    temperature=settings.transition_smoothing_temperature,
                )
                _cache["transition_smoother"] = RuntimeTransitionSmoother(
                    G,
                    collection,
                    config,
                )
            except Exception as exc:
                _cache["transition_smoother_error"] = str(exc)
                print(f"[career_kg_web] Transition smoothing unavailable: {exc}")
            finally:
                _cache["transition_smoother_attempted"] = True
        return _cache["transition_smoother"]

    def _load_link_prediction_runtime(G, collection):
        if not settings.link_prediction_enabled or G is None or collection is None:
            return None
        if _cache["link_prediction_attempted"]:
            return _cache["link_prediction_runtime"]
        with _link_prediction_lock:
            if _cache["link_prediction_attempted"]:
                return _cache["link_prediction_runtime"]
            try:
                _cache["link_prediction_runtime"] = build_link_prediction_runtime(
                    settings.link_prediction_model_path,
                    G,
                    collection,
                )
            except Exception as exc:
                _cache["link_prediction_error"] = str(exc)
                print(f"[career_kg_web] Link prediction unavailable: {exc}")
            finally:
                _cache["link_prediction_attempted"] = True
        return _cache["link_prediction_runtime"]

    @app.get("/")
    def index():
        # Serve React SPA if built, else legacy template
        if _react_dist.exists():
            return flask.send_from_directory(str(_react_dist), "index.html")
        return render_template("index.html")

    @app.get("/graph")
    def graph_view():
        return render_template("graph.html")

    # React SPA catch-all: serve index.html for all non-API, non-static routes
    @app.get("/<path:path>")
    def spa_catch_all(path: str):
        if _react_dist.exists() and not path.startswith("api/"):
            react_file = _react_dist / path
            if react_file.exists() and react_file.is_file():
                return flask.send_from_directory(str(_react_dist), path)
            return flask.send_from_directory(str(_react_dist), "index.html")
        return flask.abort(404)

    @app.get("/api/graph-data")
    def graph_data():
        G, _ = _load_resources()
        if G is None:
            return jsonify({"error": "Graph not loaded"}), 503

        # Sample the top-degree roles from each source
        onet_roles = sorted(
            [n for n, d in G.nodes(data=True) if d.get("type") == "role" and d.get("source") == "onet"],
            key=lambda n: G.degree(n), reverse=True
        )[:60]
        esco_roles = sorted(
            [n for n, d in G.nodes(data=True) if d.get("type") == "role" and d.get("source") == "esco"],
            key=lambda n: G.degree(n), reverse=True
        )[:60]
        sampled_roles = set(onet_roles + esco_roles)

        cy_nodes, cy_edges = [], []
        seen_nodes, seen_edges, skill_nodes = set(), set(), set()

        def _add_node(nid):
            if nid in seen_nodes:
                return
            seen_nodes.add(nid)
            d = G.nodes[nid]
            cy_nodes.append({"data": {
                "id": nid,
                "label": (d.get("title") or nid)[:45],
                "type": d.get("type", "unknown"),
                "source": d.get("source", ""),
                "job_zone": d.get("job_zone_title", ""),
                "description": (d.get("description") or "")[:120],
            }})

        def _add_edge(src, dst, attrs):
            relation = attrs.get("relation", "EDGE")
            eid = f"{src}__{relation}__{dst}"
            if eid in seen_edges:
                return
            seen_edges.add(eid)
            cy_edges.append({"data": {
                "id": eid, "source": src, "target": dst,
                **attrs,
            }})

        for nid in sampled_roles:
            _add_node(nid)

        # SIMILAR_TO edges between sampled roles
        for src in sampled_roles:
            for _, dst, ed in G.out_edges(src, data=True):
                if ed.get("relation") == "SIMILAR_TO" and dst in sampled_roles:
                    _add_edge(src, dst, {"relation": "SIMILAR_TO", "similarity": round(ed.get("similarity", 0), 2)})

        # Strongest empirical transitions between sampled roles, globally capped.
        transition_edges = []
        for src in sampled_roles:
            for _, dst, ed in G.out_edges(src, data=True):
                if is_training_transition(ed) and dst in sampled_roles:
                    transition_edges.append((src, dst, ed))
        transition_edges.sort(
            key=lambda item: (
                -float(item[2].get("probability", 0.0)),
                -int(item[2].get("count", 0)),
                str(item[0]),
                str(item[1]),
            )
        )
        for src, dst, ed in transition_edges[:120]:
            _add_edge(src, dst, {
                "relation": "TRANSITIONS_TO",
                "count": int(ed.get("count", 0)),
                "probability": round(float(ed.get("probability", 0.0)), 4),
            })

        # Top-4 essential REQUIRES edges per role, skill nodes capped at 160
        for nid in sampled_roles:
            essential = [(dst, ed) for _, dst, ed in G.out_edges(nid, data=True)
                         if ed.get("relation") == "REQUIRES"
                         and ed.get("requirement_level") == "essential"
                         and G.has_node(dst)][:4]
            for dst, ed in essential:
                if len(skill_nodes) >= 160 and dst not in skill_nodes:
                    continue
                skill_nodes.add(dst)
                _add_node(dst)
                _add_edge(nid, dst, {"relation": "REQUIRES"})

        return jsonify({
            "nodes": cy_nodes,
            "edges": cy_edges,
            "stats": {
                "total_nodes": G.number_of_nodes(),
                "total_edges": G.number_of_edges(),
                "shown_nodes": len(cy_nodes),
                "shown_edges": len(cy_edges),
            },
        })

    @app.get("/api/status")
    def status():
        G, collection = _load_resources()
        transition_smoother = _load_transition_smoother(G, collection)
        link_prediction_runtime = _load_link_prediction_runtime(G, collection)
        return jsonify(
            {
                "ready": G is not None and collection is not None,
                "graph_loaded": G is not None,
                "chroma_loaded": collection is not None,
                "graph_nodes": G.number_of_nodes() if G else 0,
                "graph_edges": G.number_of_edges() if G else 0,
                "transition_edges": (
                    sum(
                        1
                        for _, _, data in G.edges(data=True)
                        if is_training_transition(data)
                    )
                    if G else 0
                ),
                "transition_smoothing_enabled": settings.transition_smoothing_enabled,
                "transition_smoothing_loaded": transition_smoother is not None,
                "transition_smoothing_error": _cache["transition_smoother_error"],
                "link_prediction_enabled": settings.link_prediction_enabled,
                "link_prediction_loaded": link_prediction_runtime is not None,
                "link_prediction_error": _cache["link_prediction_error"],
                "link_prediction_diagnostics": (
                    link_prediction_runtime.diagnostics()
                    if link_prediction_runtime is not None
                    else None
                ),
                "chat_model": settings.chat_model,
                "embed_model": settings.embed_model,
                "rerank_model": settings.cohere_rerank_model,
                "error": _cache.get("error"),
            }
        )

    @app.post("/api/chat")
    def chat():
        payload = request.get_json(silent=True) or {}
        messages = payload.get("messages") or []

        if not messages:
            return jsonify({"status": "error", "message": "Please enter a question."}), 400

        # Extract the last user message as the active query
        user_turns = [m for m in messages if m.get("role") == "user"]
        if not user_turns:
            return jsonify({"status": "error", "message": "No user message found."}), 400
        query = user_turns[-1].get("content", "").strip()

        G, collection = _load_resources()
        if G is None or collection is None:
            return jsonify(
                {
                    "status": "error",
                    "message": (
                        "The knowledge graph is not ready. "
                        f"Details: {_cache.get('error', 'unknown error')}"
                    ),
                }
            ), 503

        try:
            transition_smoother = _load_transition_smoother(G, collection)
            link_prediction_runtime = _load_link_prediction_runtime(G, collection)

            if settings.use_langgraph:
                from agents.graph import run_career_workflow
                result = run_career_workflow(
                    query,
                    messages,
                    settings,
                    G,
                    collection,
                    transition_smoother,
                    link_prediction_runtime,
                )
            else:
                result = run_query(
                    query,
                    G,
                    collection,
                    settings,
                    history=messages,
                    transition_smoother=transition_smoother,
                    link_prediction_runtime=link_prediction_runtime,
                )
            return jsonify({
                "status": "ok",
                "message": result["message"],
                "courses": result.get("courses", []),
                "path": result.get("path", {}),
                "explore": result.get("explore", {}),
                "evidence": result.get("evidence", {}),
                "faithfulness": result.get("faithfulness"),
                "explanations": result.get("explanations", []),
                "skill_gap_analysis": result.get("skill_gap_analysis", []),
                "learning_plan": result.get("learning_plan", []),
                "metadata": result.get("metadata"),
            })
        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc)}), 500

    return app


def main() -> int:
    app = create_app()
    app.run(host="127.0.0.1", port=8001, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
