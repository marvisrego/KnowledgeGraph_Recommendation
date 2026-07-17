"""
GraphRAG inference pipeline.

Strict order: intent routing → retrieval → reranking → graph traversal → generation.
All external API calls use urllib.request — no openai SDK dependency.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any

import networkx as nx

if TYPE_CHECKING:
    from config import Settings


# ---------------------------------------------------------------------------
# Low-level HTTP helpers
# ---------------------------------------------------------------------------

def _post_json(url: str, payload: dict, headers: dict, timeout: int = 60) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"[inference_pipeline] HTTP {exc.code} from {url}: {body_text}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"[inference_pipeline] Connection error calling {url}: {exc.reason}"
        ) from exc


def _chat_headers(settings: "Settings") -> dict:
    if not settings.chat_model_api_key:
        raise RuntimeError(
            "[inference_pipeline] CHAT_MODEL_API_KEY is not set. "
            "Add it to your .env file."
        )
    return {
        "Content-Type": "application/json",
        "api-key": settings.chat_model_api_key,
    }


def _call_chat(messages: list[dict], settings: "Settings", max_tokens: int = 1000) -> str:
    """Call the chat model (OpenAI Responses API via APIM proxy)."""
    payload = {
        "model": settings.chat_model,
        "input": messages,
        "max_output_tokens": max_tokens,
    }
    result = _post_json(
        settings.chat_model_endpoint,
        payload,
        _chat_headers(settings),
        timeout=120,
    )
    # Responses API shape: find the first output item of type 'message'
    # (reasoning models also emit a 'reasoning' item which we skip)
    try:
        for item in result.get("output", []):
            if item.get("type") == "message":
                return item["content"][0]["text"].strip()
        raise RuntimeError(
            f"[inference_pipeline] Unexpected chat response shape: {result}"
        )
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"[inference_pipeline] Unexpected chat response shape: {result}"
        ) from exc


# ---------------------------------------------------------------------------
# Step 1: Intent routing
# ---------------------------------------------------------------------------

_INTENT_SYSTEM_PROMPT = """
You are a career advisor intake assistant. Your job is to:
1. Determine whether the user is a STUDENT or a WORKING PROFESSIONAL.
2. Determine whether they have given enough specific context to receive personalised career recommendations.

---
USER TYPE DEFINITIONS:
- STUDENT: currently studying, recently graduated (within ~1 year), or about to graduate. May have a degree/field but no significant work experience.
- PROFESSIONAL: currently employed or has meaningful work experience (internships count if they have a clear current role).
- UNKNOWN: cannot be determined from the message.

---
SUFFICIENT CONTEXT RULES:

For a STUDENT, sufficient context requires ALL of:
  1. Their degree subject or field of study.
  2. At least one skill, tool, or area of interest.
  3. A career goal or direction they want to explore.

For a PROFESSIONAL, sufficient context requires ALL of:
  1. Their current or most recent job role/title.
  2. Their key skills or domain (at least one specific skill or technology).
  3. What they want to do next — switch roles, upskill, change industry, or get promoted.

INSUFFICIENT — always ask follow-up questions for:
- Vague phrases with no background: "help me change career", "what job should I do", "I want a new job", "I want to work in tech"
- Background only with no goal: "I am a teacher", "I studied biology"
- Goal only with no background: "I want to be a data scientist"
- Any message that does not clearly establish BOTH background AND goal.

---
RESPONSE FORMAT — pure JSON only, no markdown, no code fences:

If user type is UNKNOWN (cannot tell if student or professional):
{"has_context": false, "user_type": "unknown", "followup_questions": "Are you currently a student or a working professional?\nWhat is your current role or field of study?\nWhat kind of career guidance are you looking for?"}

If user type is STUDENT but context is insufficient:
{"has_context": false, "user_type": "student", "followup_questions": "What degree or subject are you studying (or did you recently graduate in)?\nWhat skills, tools, or areas interest you most?\nWhat kind of roles or industry are you hoping to enter?"}

If user type is PROFESSIONAL but context is insufficient:
{"has_context": false, "user_type": "professional", "followup_questions": "What is your current job title and how many years of experience do you have?\nWhat are your main skills or technologies?\nAre you looking to switch roles, upskill, change industry, or get promoted?"}

If context is sufficient:
{"has_context": true, "user_type": "<student|professional>"}
""".strip()


def route_intent(query: str, settings: "Settings", history: list[dict] | None = None) -> dict:
    """Classify user intent using full conversation history.

    history: full messages list from the client (role/content dicts).
    Returns dict with 'has_context', 'user_type', and optional 'followup_questions'.
    """
    # Build conversation: system prompt first, then all prior turns, then current query
    messages = [{"role": "system", "content": _INTENT_SYSTEM_PROMPT}]
    if history:
        # Include all turns except the last user message (we add that explicitly)
        for msg in history[:-1]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": query})

    try:
        raw = _call_chat(messages, settings, max_tokens=2000)
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"has_context": True, "user_type": "professional"}
    except RuntimeError:
        raise


# ---------------------------------------------------------------------------
# Step 2: Retrieval (ChromaDB)
# ---------------------------------------------------------------------------

def retrieve_candidates(
    query: str,
    collection,
    settings: "Settings",
) -> list[dict]:
    """Embed the query and retrieve top-K role candidates from ChromaDB."""
    from src.embeddings_index import embed_texts

    try:
        vectors = embed_texts([query], settings)
    except RuntimeError:
        raise

    query_vector = vectors[0]

    try:
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=settings.retrieval_top_k,
            include=["metadatas", "documents", "distances"],
        )
    except Exception as exc:
        raise RuntimeError(
            f"[inference_pipeline] ChromaDB query failed: {exc}"
        ) from exc

    candidates = []
    ids = results.get("ids", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for nid, meta, doc, dist in zip(ids, metadatas, documents, distances):
        candidates.append(
            {
                "id": nid,
                "metadata": meta,
                "document": doc,
                "score": 1.0 - dist,  # ChromaDB cosine distance → similarity
            }
        )
    return candidates


# ---------------------------------------------------------------------------
# Step 3: Reranking (Cohere rerank via Azure APIM)
# ---------------------------------------------------------------------------

def rerank_candidates(
    query: str,
    candidates: list[dict],
    settings: "Settings",
) -> list[dict]:
    """Rerank candidates with Cohere rerank via Azure APIM.
    Returns top settings.rerank_top_n candidates sorted by reranker score."""
    if not settings.cohere_rerank_api_key:
        raise RuntimeError(
            "[inference_pipeline] COHERE_RERANK_API_KEY is not set. "
            "Add it to your .env file."
        )

    documents = [c["document"] for c in candidates]
    payload = {
        "model": settings.cohere_rerank_model,
        "query": query,
        "documents": documents,
        "top_n": settings.rerank_top_n,
    }
    headers = {
        "Content-Type": "application/json",
        "api-key": settings.cohere_rerank_api_key,
    }

    try:
        result = _post_json(settings.cohere_rerank_endpoint, payload, headers, timeout=60)
    except RuntimeError:
        raise

    # Cohere v2 shape: {"results": [{"index": int, "relevance_score": float}, ...]}
    rankings = result.get("results", [])
    ranked = sorted(rankings, key=lambda x: x.get("relevance_score", 0.0), reverse=True)
    top_n = ranked[: settings.rerank_top_n]

    return [
        {**candidates[r["index"]], "rerank_score": r.get("relevance_score", 0.0)}
        for r in top_n
        if r["index"] < len(candidates)
    ]


# ---------------------------------------------------------------------------
# Step 4: Graph traversal
# ---------------------------------------------------------------------------

def traverse_graph(
    anchor_ids: list[str],
    G: nx.DiGraph,
    onet_importance_threshold: float = 3.0,
) -> list[tuple]:
    """Collect graph triples for each anchor node.

    Returns list of (src_id, relation, dst_id, edge_attrs_dict).
    """
    triples: list[tuple] = []
    seen_edges: set[tuple] = set()

    def _add(src: str, rel: str, dst: str, attrs: dict) -> None:
        key = (src, rel, dst)
        if key not in seen_edges:
            seen_edges.add(key)
            triples.append((src, rel, dst, attrs))

    for node_id in anchor_ids:
        if not G.has_node(node_id):
            continue

        node_data = G.nodes[node_id]
        source = node_data.get("source", "")

        for _, dst, edge_data in G.out_edges(node_id, data=True):
            rel = edge_data.get("relation", "")

            if rel == "REQUIRES":
                req_level = edge_data.get("requirement_level")
                if source == "onet":
                    # Only include edges with importance >= threshold
                    try:
                        if float(req_level) < onet_importance_threshold:
                            continue
                    except (TypeError, ValueError):
                        continue
                # ESCO: include both essential and optional
                _add(node_id, rel, dst, dict(edge_data))

            elif rel in ("BELONGS_TO", "BROADER_THAN", "NARROWER_THAN", "SIMILAR_TO"):
                _add(node_id, rel, dst, dict(edge_data))

        # Include inbound SIMILAR_TO edges (cross-framework alignment)
        for src, _, edge_data in G.in_edges(node_id, data=True):
            if edge_data.get("relation") == "SIMILAR_TO":
                _add(src, "SIMILAR_TO", node_id, dict(edge_data))

    return triples


# ---------------------------------------------------------------------------
# Step 5: Generation
# ---------------------------------------------------------------------------

def _build_context_block(
    anchor_ids: list[str],
    triples: list[tuple],
    G: nx.DiGraph,
) -> str:
    lines: list[str] = []

    # Anchor node summaries
    lines.append("=== Anchor Roles ===")
    for nid in anchor_ids:
        if not G.has_node(nid):
            continue
        d = G.nodes[nid]
        lines.append(
            f"[{d.get('source','?').upper()}] {d.get('title', nid)} (id: {nid})"
        )
        if d.get("description"):
            lines.append(f"  Description: {d['description'][:200]}")
        if d.get("job_zone_title"):
            lines.append(f"  Preparation: {d['job_zone_title']}")
        if d.get("typical_education_level"):
            lines.append(f"  Education: {d['typical_education_level']}")
        if d.get("green_share") not in (None, ""):
            lines.append(f"  Green share: {d['green_share']}")
        if d.get("isco_group"):
            lines.append(f"  ISCO group: {d['isco_group']}")
        for flag in ("is_digital", "is_green_skill", "is_research_occupation"):
            if d.get(flag):
                lines.append(f"  Tag: {flag}")

    # Graph triples
    lines.append("\n=== Graph Triples ===")
    for src, rel, dst, attrs in triples:
        src_title = G.nodes[src].get("title", src) if G.has_node(src) else src
        dst_title = G.nodes[dst].get("title", dst) if G.has_node(dst) else dst
        extra = []
        if "requirement_level" in attrs:
            extra.append(f"level={attrs['requirement_level']}")
        if "domain" in attrs:
            extra.append(f"domain={attrs['domain']}")
        if "skill_type" in attrs:
            extra.append(f"skill_type={attrs['skill_type']}")
        if "similarity" in attrs:
            extra.append(f"similarity={attrs['similarity']:.2f}")
        extra_str = f" [{', '.join(extra)}]" if extra else ""
        lines.append(f"  ({src_title}) --[{rel}]--> ({dst_title}){extra_str}")

    return "\n".join(lines)


_GENERATION_SYSTEM_PROMPT_STUDENT = """
You are a warm, knowledgeable university careers advisor speaking directly to a student or recent graduate.
Your recommendations draw EXCLUSIVELY from the ONET and ESCO knowledge graph data provided — never invent roles, skills, or pathways that are not in the graph.

HOW TO WRITE:
- Speak naturally in flowing paragraphs, as if sitting across from the student in a one-on-one advising session.
- Address them directly ("Based on your background in…", "You're well-positioned for…").
- Recommend 2–3 roles maximum. For each, weave in why it fits their profile by referencing graph data naturally (e.g. "this role lists Python as an essential requirement, which lines up with your experience").
- When mentioning skill gaps, integrate them into your advice conversationally ("To strengthen your candidacy, you'll want to develop…").
- Use **bold** only for role titles and skill names — never for section headings or structural formatting.
- Do NOT use markdown headings (##), bullet lists, numbered lists, or horizontal rules.
- Do NOT use phrases like "According to the graph" or "The ONET data shows" — just state the advice naturally.
- Keep it concise: 3–4 paragraphs total. Quality over quantity.
- End with a brief encouraging note about next steps or what to focus on first.

CONSTRAINTS:
- Every role and skill you mention MUST appear in the provided graph triples or anchor node data.
- Do NOT invent or hallucinate anything not in the graph.
""".strip()

_GENERATION_SYSTEM_PROMPT_PROFESSIONAL = """
You are an experienced career coach speaking directly to a working professional about their next move.
Your recommendations draw EXCLUSIVELY from the ONET and ESCO knowledge graph data provided — never invent roles, skills, or pathways that are not in the graph.

HOW TO WRITE:
- Speak naturally in flowing paragraphs, like a trusted senior colleague giving career advice over coffee.
- Address them directly ("Given your experience in…", "Your background in X puts you in a strong position for…").
- Recommend 2–3 target roles maximum. For each, explain the fit by weaving in graph data naturally (e.g. "this role requires the same core skills you already have, plus…").
- For skill gaps, be specific but encouraging ("The main gap I see is… — picking that up would open the door to…").
- Mention transition difficulty honestly — if a role is a stretch, say so and suggest a stepping-stone path using related roles from the graph.
- Use **bold** only for role titles and skill names — never for section headings or structural formatting.
- Do NOT use markdown headings (##), bullet lists, numbered lists, or horizontal rules.
- Do NOT use phrases like "According to the graph" or "The data shows" — just state the advice naturally.
- Keep it concise: 3–4 paragraphs total. Quality over quantity.
- End with a clear "what to do first" recommendation.

CONSTRAINTS:
- Every role and skill you mention MUST appear in the provided graph triples or anchor node data.
- Do NOT invent or hallucinate anything not in the graph.
""".strip()


def generate_response(
    query: str,
    anchor_ids: list[str],
    triples: list[tuple],
    G: nx.DiGraph,
    settings: "Settings",
    user_type: str = "professional",
    history: list[dict] | None = None,
) -> str:
    """Generate a grounded career recommendation from graph triples."""
    system_prompt = (
        _GENERATION_SYSTEM_PROMPT_STUDENT
        if user_type == "student"
        else _GENERATION_SYSTEM_PROMPT_PROFESSIONAL
    )
    context = _build_context_block(anchor_ids, triples, G)

    # Build messages: system + prior conversation turns + final user message with graph context
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        for msg in history[:-1]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": f"User query: {query}\n\n{context}"})

    try:
        return _call_chat(messages, settings, max_tokens=4000)
    except RuntimeError:
        raise


def fetch_coursera_courses(anchor_ids: list[str], G: nx.DiGraph, user_type: str) -> list[dict]:
    """Fetch Coursera course recommendations as structured data."""
    try:
        from coursera_client import build_search_queries, search_course_recommendations_multi
    except ImportError:
        return []

    role_titles = []
    essential_skills = []
    for nid in anchor_ids:
        if not G.has_node(nid):
            continue
        d = G.nodes[nid]
        title = d.get("title", "")
        if title:
            role_titles.append(title)
        for _, dst, edge in G.out_edges(nid, data=True):
            if edge.get("relation") == "REQUIRES" and edge.get("requirement_level") in ("essential", "essential/optional"):
                skill_title = G.nodes[dst].get("title", "") if G.has_node(dst) else ""
                if skill_title:
                    essential_skills.append(skill_title)

    if not role_titles:
        return []

    fallback = f"{'entry level ' if user_type == 'student' else ''}{role_titles[0]}"
    queries = build_search_queries(role_titles[0], essential_skills[:5], fallback)

    try:
        courses = search_course_recommendations_multi(queries, limit=3, timeout=15)
    except Exception:
        return []

    return courses if courses else []


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run_query(
    query: str,
    G: nx.DiGraph,
    collection,
    settings: "Settings",
    history: list[dict] | None = None,
) -> dict:
    """Orchestrate the full GraphRAG inference loop.

    Returns a dict: {"message": str, "courses": list[dict]}
    """
    # 1. Intent routing
    try:
        intent = route_intent(query, settings, history=history)
    except RuntimeError as exc:
        print(f"[inference_pipeline] Intent routing failed (continuing): {exc}")
        intent = {"has_context": True, "user_type": "professional"}

    if not intent.get("has_context", True):
        msg = intent.get(
            "followup_questions",
            "Could you tell me more about your current background and career goals?",
        )
        return {"message": msg, "courses": []}

    user_type = intent.get("user_type", "professional")
    if user_type not in ("student", "professional"):
        user_type = "professional"

    # 2. Retrieval
    try:
        candidates = retrieve_candidates(query, collection, settings)
    except RuntimeError as exc:
        return {"message": f"I was unable to search for matching roles: {exc}", "courses": []}

    if not candidates:
        return {"message": "No matching roles were found in the knowledge base for your query.", "courses": []}

    # 3. Reranking
    try:
        top_candidates = rerank_candidates(query, candidates, settings)
    except RuntimeError as exc:
        print(f"[inference_pipeline] Reranking failed (using top retrieval results): {exc}")
        top_candidates = candidates[: settings.rerank_top_n]

    anchor_ids = [c["id"] for c in top_candidates]

    # 4. Graph traversal
    try:
        triples = traverse_graph(anchor_ids, G, settings.onet_importance_threshold)
    except Exception as exc:
        return {"message": f"Graph traversal failed: {exc}", "courses": []}

    if not triples:
        return {
            "message": "I found some matching roles but could not retrieve supporting details from the knowledge graph.",
            "courses": [],
        }

    # 5. Generation
    try:
        response = generate_response(query, anchor_ids, triples, G, settings, user_type, history=history)
    except RuntimeError as exc:
        return {"message": f"I encountered an error generating a response: {exc}", "courses": []}

    # 6. Coursera recommendations (structured data, separate from message)
    courses = fetch_coursera_courses(anchor_ids, G, user_type)

    return {"message": response, "courses": courses}
