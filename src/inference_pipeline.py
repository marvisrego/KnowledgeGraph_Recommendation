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

from src.transition_policy import is_training_transition

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
EXTRACTION RULES:
- Copy the current role and skills only from information the USER explicitly provided.
- Never infer that the user has a skill merely because it belongs to their desired role.
- Exclude skills the user says they do not have or still need to learn.
- Use an empty string or empty list when a field is unavailable.
- career_history is an ordered list of role titles the USER explicitly stated,
  oldest to newest. Do not infer, embellish, or add desired roles. It may be [].

RESPONSE FORMAT — pure JSON only, no markdown, no code fences. Always include
current_role, career_history, skills, and career_goal:

If user type is UNKNOWN (cannot tell if student or professional):
{"has_context": false, "user_type": "unknown", "current_role": "", "career_history": [], "skills": [], "career_goal": "", "followup_questions": "Are you currently a student or a working professional?\nWhat is your current role or field of study?\nWhat kind of career guidance are you looking for?"}

If user type is STUDENT but context is insufficient:
{"has_context": false, "user_type": "student", "current_role": "", "career_history": [], "skills": ["<explicitly stated skill>"], "career_goal": "", "followup_questions": "What degree or subject are you studying (or did you recently graduate in)?\nWhat skills, tools, or areas interest you most?\nWhat kind of roles or industry are you hoping to enter?"}

If user type is PROFESSIONAL but context is insufficient:
{"has_context": false, "user_type": "professional", "current_role": "<explicit role or empty>", "career_history": ["<explicit past or current role title>"], "skills": ["<explicitly stated skill>"], "career_goal": "", "followup_questions": "What is your current job title and how many years of experience do you have?\nWhat are your main skills or technologies?\nAre you looking to switch roles, upskill, change industry, or get promoted?"}

If context is sufficient:
{"has_context": true, "user_type": "<student|professional>", "current_role": "<explicit current role or empty for student>", "career_history": ["<explicit roles oldest to newest>"], "skills": ["<explicitly owned skill>"], "career_goal": "<explicit goal>"}
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
        return {
            "has_context": True,
            "user_type": "professional",
            "current_role": "",
            "career_history": [],
            "skills": [],
            "career_goal": "",
        }
    except RuntimeError:
        raise


# ---------------------------------------------------------------------------
# Step 2: Retrieval (vector database)
# ---------------------------------------------------------------------------

def retrieve_candidates(
    query: str,
    collection,
    settings: "Settings",
    limit: int | None = None,
) -> list[dict]:
    """Embed the query and retrieve top-K role candidates from the vector store."""
    from src.embeddings_index import embed_texts

    try:
        vectors = embed_texts([query], settings)
    except RuntimeError:
        raise

    query_vector = vectors[0]

    try:
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=limit or settings.retrieval_top_k,
            include=["metadatas", "documents", "distances"],
        )
    except Exception as exc:
        raise RuntimeError(
            f"[inference_pipeline] Vector database query failed: {exc}"
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
                "score": 1.0 - dist,  # cosine distance to similarity
            }
        )
    return candidates


def _user_authored_texts(query: str, history: list[dict] | None) -> list[str]:
    texts = [
        str(message.get("content", ""))
        for message in (history or [])
        if message.get("role") == "user" and str(message.get("content", "")).strip()
    ]
    if not texts or texts[-1].strip() != query.strip():
        texts.append(query)
    return texts


def filter_candidates_to_graph(candidates: list[dict], G: nx.Graph) -> list[dict]:
    """Drop stale vector-index hits whose node identifiers are absent from the KG."""
    return [candidate for candidate in candidates if G.has_node(str(candidate.get("id", "")))]


def augment_candidates_with_transitions(
    candidates: list[dict],
    current_role_id: str | None,
    G: nx.MultiDiGraph,
    limit: int,
    smoothed_destinations: list | None = None,
) -> list[dict]:
    """Union semantic candidates with direct and accepted smoothed transitions."""
    from src.embeddings_index import build_role_text
    from src.skill_gap import transition_destinations, transition_evidence

    combined: dict[str, dict] = {
        str(candidate["id"]): dict(candidate)
        for candidate in filter_candidates_to_graph(candidates, G)
        if str(candidate.get("id", "")) != str(current_role_id or "")
    }
    order = list(combined)

    def _merge(role_id: str, transition: dict) -> None:
        if role_id in combined:
            combined[role_id]["transition"] = transition
            return
        data = G.nodes[role_id]
        combined[role_id] = {
            "id": role_id,
            "metadata": {
                "title": data.get("title", role_id),
                "source": data.get("source", "esco"),
                "type": data.get("type", "role"),
            },
            "document": build_role_text(role_id, data),
            "score": 0.0,
            "transition": transition,
        }
        order.append(role_id)

    if smoothed_destinations:
        for raw_item in smoothed_destinations[: max(0, limit)]:
            item = raw_item.to_dict() if hasattr(raw_item, "to_dict") else dict(raw_item)
            role_id = str(item.get("id", ""))
            if not role_id or role_id == current_role_id or not G.has_node(role_id):
                continue
            if item.get("evidence_type") == "direct_transition":
                transition = transition_evidence(current_role_id, role_id, G)
                if transition:
                    _merge(role_id, transition)
                continue
            if item.get("evidence_type") != "semantic_transition_backoff":
                continue
            _merge(
                role_id,
                {
                    "from_role_id": current_role_id,
                    "evidence_type": "semantic_transition_backoff",
                    "score": float(item.get("score", 0.0)),
                    "neighbour_probability": float(item.get("neighbour_probability", 0.0)),
                    "neighbour_support": int(item.get("neighbour_support", 0)),
                },
            )
    else:
        for empirical in transition_destinations(current_role_id, G, limit=limit):
            role_id = empirical["id"]
            if role_id == current_role_id or not G.has_node(role_id):
                continue
            _merge(role_id, empirical["transition"])

    return [combined[role_id] for role_id in order]


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
    G: nx.MultiDiGraph,
    onet_importance_threshold: float = 3.5,
    transition_limit: int = 8,
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

        transition_edges: list[tuple[str, dict]] = []
        for _, dst, edge_data in G.out_edges(node_id, data=True):
            rel = edge_data.get("relation", "")

            if rel == "REQUIRES":
                req_level = edge_data.get("requirement_level")
                if source == "onet":
                    try:
                        if float(req_level) < onet_importance_threshold:
                            continue
                    except (TypeError, ValueError):
                        continue
                else:
                    # ESCO: only essential skills — optional adds noise
                    if req_level not in ("essential",):
                        continue
                _add(node_id, rel, dst, dict(edge_data))

            elif rel in ("BELONGS_TO", "BROADER_THAN", "NARROWER_THAN", "SIMILAR_TO"):
                _add(node_id, rel, dst, dict(edge_data))

            elif is_training_transition(edge_data):
                transition_edges.append((str(dst), dict(edge_data)))

        transition_edges.sort(
            key=lambda item: (
                -float(item[1].get("probability", 0.0)),
                -int(item[1].get("count", 0)),
                str(G.nodes[item[0]].get("title", "")) if G.has_node(item[0]) else item[0],
                item[0],
            )
        )
        for dst, edge_data in transition_edges[: max(0, transition_limit)]:
            _add(node_id, "TRANSITIONS_TO", dst, edge_data)

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
    G: nx.MultiDiGraph,
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
        if "count" in attrs and rel == "TRANSITIONS_TO":
            extra.append(f"observed_count={int(attrs['count'])}")
        if "probability" in attrs and rel == "TRANSITIONS_TO":
            extra.append(f"observed_probability={float(attrs['probability']):.4f}")
        if rel == "SEMANTIC_TRANSITION_BACKOFF":
            extra.append(f"inferred_score={float(attrs.get('score', 0.0)):.4f}")
            extra.append(f"related_source_roles={int(attrs.get('neighbour_support', 0))}")
        if rel == "PREDICTED_TRANSITION":
            extra.append(f"predicted_score={float(attrs.get('score', 0.0)):.4f}")
            extra.append(f"model={attrs.get('model', 'unknown')}")
        extra_str = f" [{', '.join(extra)}]" if extra else ""
        lines.append(f"  ({src_title}) --[{rel}]--> ({dst_title}){extra_str}")

    return "\n".join(lines)


_GENERATION_SYSTEM_PROMPT_STUDENT = """
You are a university careers advisor. Respond in exactly 1 short paragraph — 2 to 3 sentences only.
Address the student directly, name their best-fit role in **bold**, and briefly mention one key skill they already have and one to develop. Bold every role and skill you name.
Call something a strength or existing skill only when the user explicitly states they have it; a role requirement alone is not evidence of ownership.
Prioritize occupation-specific skill gaps. Mention general language or comprehension abilities as the main gap only when they are central to the destination role or no stronger graph-supported gap exists.
No headings, no bullet lists. Every role and skill you name MUST appear in the provided graph data.
Use exact graph labels; do not introduce abbreviations, combined labels, or umbrella terms that are absent from the graph.
Empirical transitions are population-level observations, not guarantees about an individual.
Predicted or semantic transition evidence is inferred and must never be described as a directly observed move.
""".strip()

_GENERATION_SYSTEM_PROMPT_PROFESSIONAL = """
You are a senior career coach. Respond in exactly 1 short paragraph — 2 to 3 sentences only.
Address the professional directly, name their best-fit next role in **bold**, and briefly mention one strength they have and one skill gap to bridge. Bold every role and skill you name.
Call something a strength or existing skill only when the user explicitly states they have it; a role requirement alone is not evidence of ownership.
Prioritize occupation-specific skill gaps. Mention general language or comprehension abilities as the main gap only when they are central to the destination role or no stronger graph-supported gap exists.
No headings, no bullet lists. Every role and skill you name MUST appear in the provided graph data.
Use exact graph labels; do not introduce abbreviations, combined labels, or umbrella terms that are absent from the graph.
Treat empirical transitions as population-level evidence, never as a guaranteed outcome.
SEMANTIC_TRANSITION_BACKOFF is inferred from training transitions of semantically related source roles; never describe it as a directly observed move.
PREDICTED_TRANSITION is model-inferred missing-edge evidence; never describe it as observed population evidence.
""".strip()


def generate_response(
    query: str,
    anchor_ids: list[str],
    triples: list[tuple],
    G: nx.MultiDiGraph,
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
        return _call_chat(messages, settings, max_tokens=1000)
    except RuntimeError:
        raise


def fetch_coursera_courses(anchor_ids: list[str], G: nx.MultiDiGraph, user_type: str) -> list[dict]:
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
            if edge.get("relation") == "REQUIRES" and edge.get("requirement_level") == "essential":
                skill_title = G.nodes[dst].get("title", "") if G.has_node(dst) else ""
                if skill_title:
                    essential_skills.append(skill_title)

    if not role_titles:
        return []

    fallback = f"{'entry level ' if user_type == 'student' else ''}{role_titles[0]}"
    queries = build_search_queries(role_titles[0], essential_skills[:5], fallback)

    try:
        raw_courses = search_course_recommendations_multi(queries, limit=5, timeout=15)
    except Exception:
        return []

    # Normalize to frontend schema
    courses = []
    for c in (raw_courses or []):
        partners = c.get("partner_names") or []
        courses.append({
            "title": c.get("title", ""),
            "description": c.get("description") or c.get("tagline", ""),
            "url": c.get("canonical_url", ""),
            "provider": partners[0] if partners else "",
            "estimated_workload": c.get("estimated_workload", ""),
            "content_type": c.get("content_type", "course"),
            "skills": [],
        })
    return courses


# ---------------------------------------------------------------------------
# Explore data builder (partial context — separate skills + roles visuals)
# ---------------------------------------------------------------------------

def _build_explore_data(anchor_ids: list[str], triples: list[tuple], G: nx.MultiDiGraph) -> dict:
    """Build separate roles list and skills list for the explore panels shown
    when the user only provided partial context (e.g. degree only).

    Returns:
        {
            "roles":  [{"id", "title", "source", "description", "prep"}, ...],
            "skills": [{"id", "title", "type"}, ...]   # deduplicated across all anchors
        }
    """
    roles = []
    seen_skill_ids: dict[str, dict] = {}
    seen_role_ids = set()

    for nid in anchor_ids:
        if not G.has_node(nid):
            continue
        d = G.nodes[nid]
        roles.append({
            "id": nid,
            "title": d.get("title", nid),
            "source": d.get("source", ""),
            "description": (d.get("description") or "")[:120],
            "prep": d.get("job_zone_title") or d.get("typical_education_level") or "",
        })
        seen_role_ids.add(nid)

    for src, rel, dst, attrs in triples:
        if rel != "REQUIRES" or src not in seen_role_ids:
            continue
        if not G.has_node(dst):
            continue
        req = attrs.get("requirement_level")
        # ESCO: essential only; ONET: numeric (already filtered by threshold in traverse_graph)
        if req not in ("essential",) and not isinstance(req, (int, float)):
            continue
        if dst not in seen_skill_ids:
            d = G.nodes[dst]
            raw_title = d.get("title", dst)
            seen_skill_ids[dst] = {
                "id": dst,
                "title": raw_title[:38] + "…" if len(raw_title) > 38 else raw_title,
                "type": d.get("skill_type") or d.get("type") or "skill",
                "count": 0,
            }
        seen_skill_ids[dst]["count"] += 1

    # Sort skills by how many roles require them (most common first), cap at 20
    skills = sorted(seen_skill_ids.values(), key=lambda s: s["count"], reverse=True)[:20]
    for sk in skills:
        sk.pop("count", None)

    return {"roles": roles, "skills": skills}


# ---------------------------------------------------------------------------
# Path data builder (for frontend linear career path visual)
# ---------------------------------------------------------------------------

def _build_path_data(
    anchor_ids: list[str],
    triples: list[tuple],
    G: nx.MultiDiGraph,
    ranked_candidates: list[dict] | None = None,
) -> dict:
    """Build a flat structure for the frontend linear path component.

    Returns:
        {
            "roles": [{"id", "title", "source", "description", "prep"}, ...],
            "skills": [{"id", "title", "type", "roles": [role_id, ...]}, ...],
            "similar_pairs": [[role_id_a, role_id_b], ...]
        }
    """
    candidate_by_id = {
        str(candidate["id"]): candidate for candidate in (ranked_candidates or [])
    }
    roles = []
    seen_role_ids = []
    for nid in anchor_ids:
        if not G.has_node(nid):
            continue
        d = G.nodes[nid]
        candidate = candidate_by_id.get(nid, {})
        skill_gap = candidate.get("skill_gap") or {}

        def _display_items(items: list[dict]) -> list[dict]:
            displayed = []
            for item in items:
                title = str(item.get("title", item.get("id", "")))
                displayed.append(
                    {
                        "id": item.get("id", ""),
                        "title": title[:38] + "…" if len(title) > 38 else title,
                    }
                )
            return displayed

        roles.append({
            "id": nid,
            "title": d.get("title", nid),
            "source": d.get("source", ""),
            "description": (d.get("description") or "")[:150],
            "prep": d.get("job_zone_title") or d.get("typical_education_level") or "",
            "accessibility": skill_gap.get("accessibility"),
            "gap": skill_gap.get("gap"),
            "gap_evidence": skill_gap.get("gap_evidence", "not_calculated"),
            "required_skill_count": int(skill_gap.get("required_skill_count", 0)),
            "have_count": int(skill_gap.get("have_count", 0)),
            "need_count": int(skill_gap.get("need_count", 0)),
            "have": _display_items(skill_gap.get("have", [])),
            "need": _display_items(skill_gap.get("need", [])),
            "transition": candidate.get("transition"),
            "semantic_score": candidate.get("semantic_score"),
            "retrieval_sources": candidate.get("retrieval_sources", []),
        })
        seen_role_ids.append(nid)

    # Collect top-5 skills per role (essential / high importance)
    skill_map: dict[str, dict] = {}
    for src, rel, dst, attrs in triples:
        if rel != "REQUIRES" or src not in seen_role_ids:
            continue
        if not G.has_node(dst):
            continue
        req = attrs.get("requirement_level")
        # ESCO: essential only; ONET: numeric (already threshold-filtered in traverse_graph)
        if req not in ("essential",) and not isinstance(req, (int, float)):
            continue
        if dst not in skill_map:
            d = G.nodes[dst]
            raw_title = d.get("title", dst)
            skill_map[dst] = {
                "id": dst,
                "title": raw_title[:38] + "…" if len(raw_title) > 38 else raw_title,
                "type": d.get("skill_type") or d.get("type") or "skill",
                "roles": [],
            }
        if src not in skill_map[dst]["roles"]:
            skill_map[dst]["roles"].append(src)

    # Cap at 5 skills per role, picking those linked to most roles first
    skills_sorted = sorted(skill_map.values(), key=lambda s: len(s["roles"]), reverse=True)
    role_skill_count: dict[str, int] = {rid: 0 for rid in seen_role_ids}
    selected_skills = []
    for sk in skills_sorted:
        include = False
        for rid in sk["roles"]:
            if role_skill_count.get(rid, 0) < 5:
                role_skill_count[rid] = role_skill_count.get(rid, 0) + 1
                include = True
        if include:
            selected_skills.append(sk)

    # SIMILAR_TO pairs between anchor roles
    similar_pairs = []
    seen_pairs: set[frozenset] = set()
    for src, rel, dst, _ in triples:
        if rel == "SIMILAR_TO" and src in seen_role_ids and dst in seen_role_ids:
            pair = frozenset([src, dst])
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                similar_pairs.append([src, dst])

    return {
        "roles": roles,
        "skills": selected_skills,
        "similar_pairs": similar_pairs,
        "ordering": "accessibility_then_transition_then_semantic",
    }


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run_query(
    query: str,
    G: nx.MultiDiGraph,
    collection,
    settings: "Settings",
    history: list[dict] | None = None,
    transition_smoother=None,
    link_prediction_runtime=None,
    sequential_runtime=None,
) -> dict:
    """Orchestrate the full GraphRAG inference loop.

    Returns a dict: {"message": str, "courses": list[dict]}
    """
    # 1. Intent routing
    try:
        intent = route_intent(query, settings, history=history)
    except RuntimeError as exc:
        print(f"[inference_pipeline] Intent routing failed (continuing): {exc}")
        intent = {
            "has_context": True,
            "user_type": "professional",
            "current_role": "",
            "career_history": [],
            "skills": [],
            "career_goal": "",
        }

    user_type = intent.get("user_type", "professional")
    if user_type not in ("student", "professional"):
        user_type = "professional"

    from src.skill_gap import rank_roles_by_gap, resolve_current_role, resolve_user_skills

    user_texts = _user_authored_texts(query, history)
    router_skills = intent.get("skills", [])
    if isinstance(router_skills, str):
        router_skills = [router_skills]
    elif not isinstance(router_skills, list):
        router_skills = []
    try:
        skill_evidence = resolve_user_skills(router_skills, user_texts, G)
        current_role_id = None
        if user_type == "professional":
            current_role_id = resolve_current_role(str(intent.get("current_role", "")), G)
    except Exception as exc:
        print(f"[inference_pipeline] Skill/role evidence extraction failed (continuing): {exc}")
        skill_evidence = {
            "skill_ids": set(),
            "matched": [],
            "unmatched": list(router_skills),
            "excluded": [],
        }
        current_role_id = None

    from src.career_history import resolve_career_history
    history_role_ids = resolve_career_history(
        intent.get("career_history", []), current_role_id, G
    )

    public_evidence = {
        "current_role": (
            {
                "id": current_role_id,
                "title": G.nodes[current_role_id].get("title", current_role_id),
            }
            if current_role_id and G.has_node(current_role_id)
            else None
        ),
        "matched_skills": [
            {
                "phrase": item["phrase"],
                "id": item["id"],
                "title": item["title"],
            }
            for item in skill_evidence["matched"]
        ],
        "unmatched_skills": skill_evidence["unmatched"],
        "excluded_non_owned_skills": skill_evidence.get("excluded", []),
        "resolved_career_history": [
            {"id": role_id, "title": G.nodes[role_id].get("title", role_id)}
            for role_id in history_role_ids
            if G.has_node(role_id)
        ],
    }

    # 2. Hybrid retrieval (always run — even for partial context)
    from src.hybrid_retrieval import hybrid_retrieve

    candidates = hybrid_retrieve(
        query,
        collection,
        skill_evidence["skill_ids"],
        current_role_id,
        G,
        settings,
        transition_smoother=transition_smoother,
        link_prediction_runtime=link_prediction_runtime,
        sequential_runtime=sequential_runtime,
        history_role_ids=history_role_ids,
    )

    if not candidates:
        return {"message": "No matching roles were found in the knowledge base for your query.", "courses": [], "path": {}, "explore": {}, "evidence": public_evidence}

    # 3. Reranking
    try:
        top_candidates = rerank_candidates(query, candidates, settings)
    except RuntimeError as exc:
        print(f"[inference_pipeline] Reranking failed (using top retrieval results): {exc}")
        top_candidates = candidates[: settings.rerank_top_n]

    top_candidates = rank_roles_by_gap(
        skill_evidence["skill_ids"],
        top_candidates,
        G,
        current_role_id=current_role_id,
        onet_importance_threshold=settings.onet_importance_threshold,
    )

    anchor_ids = [c["id"] for c in top_candidates]

    # 4. Graph traversal
    try:
        triples = traverse_graph(
            anchor_ids,
            G,
            settings.onet_importance_threshold,
            transition_limit=settings.transition_traversal_limit,
        )
        seen_transition_triples = {
            (source, target)
            for source, relation, target, _ in triples
            if relation == "TRANSITIONS_TO"
        }
        for candidate in top_candidates:
            transition = candidate.get("transition")
            if not transition or not current_role_id:
                continue
            evidence_type = transition.get("evidence_type", "direct_transition")
            relation = {
                "semantic_transition_backoff": "SEMANTIC_TRANSITION_BACKOFF",
                "predicted_transition": "PREDICTED_TRANSITION",
            }.get(evidence_type, "TRANSITIONS_TO")
            key = (current_role_id, candidate["id"])
            if relation == "TRANSITIONS_TO" and key in seen_transition_triples:
                continue
            triples.append(
                (
                    current_role_id,
                    relation,
                    candidate["id"],
                    {
                        "relation": relation,
                        "source": (
                            "embedding_smoothing"
                            if relation == "SEMANTIC_TRANSITION_BACKOFF"
                            else "link_prediction"
                            if relation == "PREDICTED_TRANSITION"
                            else "karrierewege"
                        ),
                        **transition,
                    },
                )
            )
            if relation == "TRANSITIONS_TO":
                seen_transition_triples.add(key)
    except Exception as exc:
        return {"message": f"Graph traversal failed: {exc}", "courses": [], "path": {}, "explore": {}, "evidence": public_evidence}

    # --- Partial context: show explore visuals + ask follow-up, skip full generation ---
    if not intent.get("has_context", True):
        msg = intent.get("followup_questions", "Could you tell me more about your background and goals?")
        explore_data = _build_explore_data(anchor_ids, triples, G)
        return {"message": msg, "courses": [], "path": {}, "explore": explore_data, "evidence": public_evidence}

    if not triples:
        return {
            "message": "I found some matching roles but could not retrieve supporting details from the knowledge graph.",
            "courses": [], "path": {}, "explore": {}, "evidence": public_evidence,
        }

    # 5. Generation
    try:
        response = generate_response(query, anchor_ids, triples, G, settings, user_type, history=history)
    except RuntimeError as exc:
        return {"message": f"I encountered an error generating a response: {exc}", "courses": [], "path": {}, "explore": {}, "evidence": public_evidence}

    # 6. Coursera recommendations (structured data, separate from message)
    courses = fetch_coursera_courses(anchor_ids, G, user_type)

    # 7. Build linear path data for frontend visual
    path_data = _build_path_data(anchor_ids, triples, G, ranked_candidates=top_candidates)

    # 7b. Compute Transition Effort Scores for each role in the path
    try:
        from src.transition_effort import transition_effort_score, get_idf_map
        from src.tes_calibration import load_calibration

        roles = path_data.get("roles", [])
        # Fall back to first recommended role as synthetic source when current role is unknown
        # (covers student users and professionals whose role title could not be resolved).
        effective_source_id = current_role_id
        effort_source_label = "from_current_role"
        if not effective_source_id and len(roles) >= 2:
            effective_source_id = roles[0]["id"]
            effort_source_label = "relative"

        if effective_source_id and roles:
            idf_map = get_idf_map(G)
            calibration = load_calibration(settings.effort_calibration_path)
            owned = skill_evidence.get("skill_ids", set())
            for role_entry in roles:
                target_id = role_entry["id"]
                if target_id == effective_source_id:
                    continue
                try:
                    effort = transition_effort_score(
                        effective_source_id, target_id,
                        owned, G, idf_map, None,
                        transition_smoother,
                        settings.onet_importance_threshold,
                        calibration,
                    )
                    role_entry["effort_score"] = round(effort.score, 4)
                    role_entry["effort_band"] = effort.band
                    role_entry["effort_source"] = effort_source_label
                    role_entry["estimated_weeks_min"] = effort.estimated_weeks_min
                    role_entry["estimated_weeks_max"] = effort.estimated_weeks_max
                    role_entry["estimated_weeks_basis"] = "heuristic"
                    role_entry["effort_calibration"] = effort.calibration.to_dict()
                except Exception as exc:
                    print(f"[inference_pipeline] Effort for {target_id}: {exc}")
    except Exception as exc:
        print(f"[inference_pipeline] Effort scoring failed (non-critical): {exc}")

    # Re-sort roles by effort ascending (easiest first); unscoreds sort last.
    path_data["roles"].sort(
        key=lambda r: (r.get("effort_score") is None, r.get("effort_score", 999))
    )

    # 7c. Faithfulness verification
    faithfulness_result = None
    try:
        from src.faithfulness import compute_faithfulness
        faithfulness_result = compute_faithfulness(response, anchor_ids, G)
    except Exception as exc:
        print(f"[inference_pipeline] Faithfulness check failed (non-critical): {exc}")

    # 7d. Explanation chains
    explanations = []
    try:
        from src.explainability import explain_career_path
        if current_role_id and path_data.get("roles"):
            explanations = explain_career_path(
                path_data["roles"], current_role_id, skill_evidence["skill_ids"], G
            )
    except Exception as exc:
        print(f"[inference_pipeline] Explanation generation failed (non-critical): {exc}")

    return {
        "message": response,
        "courses": courses,
        "path": path_data,
        "explore": {},
        "evidence": public_evidence,
        "faithfulness": faithfulness_result.to_dict() if faithfulness_result else None,
        "explanations": explanations,
    }
