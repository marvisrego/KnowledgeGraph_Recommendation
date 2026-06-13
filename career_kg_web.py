from __future__ import annotations

import secrets
from pathlib import Path
from typing import Any

from career_kg_chat import (
    MissingDependencyError,
    QueryState,
    Settings,
    apply_clarification_answers,
    get_clarification_fields,
    prepare_runtime,
    route_query,
    run_query_turn,
)


def create_app():
    flask = __import__("flask")
    Flask = flask.Flask
    jsonify = flask.jsonify
    render_template = flask.render_template
    request = flask.request

    app = Flask(__name__, template_folder="templates", static_folder="public", static_url_path="")
    app.config["JSON_SORT_KEYS"] = False

    settings = Settings.from_env(Path(__file__).resolve().parent)
    pending_queries: dict[str, QueryState] = {}
    runtime_cache: dict[str, Any] = {"bundle": None, "vector_index": None}

    def get_runtime(*, rebuild: bool = False):
        if rebuild or runtime_cache["bundle"] is None or runtime_cache["vector_index"] is None:
            bundle, vector_index = prepare_runtime(settings, rebuild=rebuild)
            runtime_cache["bundle"] = bundle
            runtime_cache["vector_index"] = vector_index
        return runtime_cache["bundle"], runtime_cache["vector_index"]

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/status")
    def status():
        try:
            bundle, _ = get_runtime()
        except Exception as exc:
            return jsonify({"ready": False, "error": str(exc)}), 500

        return jsonify(
            {
                "ready": True,
                "stats": bundle.stats,
                "chat_model": settings.nvidia_chat_model,
                "llm_enabled": bool(settings.resolved_chat_api_key),
            }
        )

    @app.post("/api/chat")
    def chat():
        payload = request.get_json(silent=True) or {}
        message = str(payload.get("message") or "").strip()
        pending_id = str(payload.get("pending_id") or "").strip()
        raw_clarifications = payload.get("clarifications") or {}

        if pending_id:
            query_state = pending_queries.get(pending_id)
            if query_state is None:
                return jsonify({"status": "error", "message": "This clarification session expired. Please send the question again."}), 400
            if not isinstance(raw_clarifications, dict):
                return jsonify({"status": "error", "message": "Clarifications must be sent as an object."}), 400

            query_state = apply_clarification_answers(query_state, {str(key): str(value) for key, value in raw_clarifications.items()})
            if query_state.is_vague:
                pending_queries[pending_id] = query_state
                return jsonify(
                    {
                        "status": "clarification",
                        "message": "I still need a little more context before I query the graph.",
                        "pending_id": pending_id,
                        "fields": get_clarification_fields(query_state),
                    }
                )

            pending_queries.pop(pending_id, None)
            try:
                bundle, vector_index = get_runtime()
                return jsonify(run_query_turn(settings, bundle, vector_index, query_state))
            except Exception as exc:
                return jsonify({"status": "error", "message": str(exc)}), 500

        if not message:
            return jsonify({"status": "error", "message": "Please enter a question."}), 400

        query_state = route_query(settings, message)
        if query_state.is_vague:
            new_pending_id = secrets.token_urlsafe(12)
            pending_queries[new_pending_id] = query_state
            return jsonify(
                {
                    "status": "clarification",
                    "message": "I need a bit more context before I query the graph.",
                    "pending_id": new_pending_id,
                    "fields": get_clarification_fields(query_state),
                }
            )

        try:
            bundle, vector_index = get_runtime()
            return jsonify(run_query_turn(settings, bundle, vector_index, query_state))
        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc)}), 500

    return app


def main() -> int:
    try:
        __import__("flask")
    except ModuleNotFoundError as exc:
        raise MissingDependencyError("Missing dependency 'flask'. Install it with: pip install -r requirements.txt") from exc

    app = create_app()
    app.run(host="127.0.0.1", port=8000, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())