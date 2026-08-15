"""Deterministic skill evidence and graph-structural career accessibility scores."""

from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Iterable

import networkx as nx

from src.karrierewege_preprocessing import normalize_label


_POSSESSION_MARKERS = re.compile(
    r"\b(?:i\s+(?:know|use|have|work(?:ed)?\s+with|am\s+(?:skilled|proficient|experienced)\s+in)|"
    r"my\s+skills\s+(?:are|include)|experience\s+(?:with|in)|experienced\s+in|"
    r"proficient\s+in|skilled\s+in|using|know)\b",
    re.IGNORECASE,
)
_NEGATION_MARKERS = re.compile(
    r"\b(?:do\s+not|don't|dont|not|never|without|lack(?:ing)?|need\s+to\s+learn|want\s+to\s+learn)\b",
    re.IGNORECASE,
)
_GOAL_MARKERS = re.compile(
    r"\b(?:want|wants|hope|hoping|aim|aiming|goal|interested|move|moving|transition|"
    r"switch|switching|learn|learning|develop|developing|need|needing|aspire|aspiring)\b",
    re.IGNORECASE,
)
_PROFICIENCY_PREFIX = re.compile(
    r"^(?:basic|beginner|intermediate|advanced|expert|some|strong|good|working)\s+",
    re.IGNORECASE,
)
_SHORT_ALIAS_ALLOWLIST = {"ai", "c", "ml", "r", "sql"}
_CURATED_ALIASES = {
    "ml": "machine learning",
}


def _skill_aliases(title: str) -> set[str]:
    preferred = normalize_label(title)
    aliases = {preferred} if preferred else set()
    without_parenthetical = normalize_label(re.sub(r"\s*\([^)]*\)\s*", " ", title))
    if without_parenthetical:
        aliases.add(without_parenthetical)
    return aliases


def build_skill_index(G: nx.Graph) -> dict[str, tuple[str, ...]]:
    """Build normalized ESCO skill aliases without fuzzy expansion."""
    index: dict[str, set[str]] = defaultdict(set)
    preferred_to_ids: dict[str, set[str]] = defaultdict(set)
    for node_id, data in G.nodes(data=True):
        if data.get("type") != "skill" or data.get("source") != "esco":
            continue
        title = str(data.get("title", ""))
        preferred = normalize_label(title)
        if preferred:
            preferred_to_ids[preferred].add(str(node_id))
        for alias in _skill_aliases(title):
            if len(alias) >= 3 or alias in _SHORT_ALIAS_ALLOWLIST:
                index[alias].add(str(node_id))

    for alias, preferred in _CURATED_ALIASES.items():
        for node_id in preferred_to_ids.get(preferred, set()):
            index[alias].add(node_id)
    return {alias: tuple(sorted(node_ids)) for alias, node_ids in index.items()}


def _phrase_candidates(phrase: str) -> list[str]:
    normalized = normalize_label(phrase)
    candidates = [normalized] if normalized else []
    stripped = normalize_label(_PROFICIENCY_PREFIX.sub("", normalized))
    if stripped and stripped not in candidates:
        candidates.append(stripped)
    return candidates


def _is_explicitly_owned(alias: str, user_texts: Iterable[str]) -> bool:
    """Require possession evidence closer to a mention than goal/negation evidence."""
    pattern = re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)", re.IGNORECASE)
    for text in user_texts:
        normalized = normalize_label(text)
        for match in pattern.finditer(normalized):
            prefix = normalized[: match.start()]
            possession_positions = []
            for item in _POSSESSION_MARKERS.finditer(prefix):
                marker_prefix = prefix[max(0, item.start() - 16) : item.start()]
                if _NEGATION_MARKERS.search(marker_prefix):
                    continue
                possession_positions.append(item.end())
            blocker_positions = [item.end() for item in _NEGATION_MARKERS.finditer(prefix)]
            blocker_positions.extend(item.end() for item in _GOAL_MARKERS.finditer(prefix))
            last_possession = max(possession_positions, default=-1)
            last_blocker = max(blocker_positions, default=-1)
            if last_possession > last_blocker:
                return True
            # A terse answer such as "Python, SQL" is a skill list unless it
            # contains explicit goal or negation language.
            if (
                last_possession < 0
                and last_blocker < 0
                and len(normalized.split()) <= 10
            ):
                return True
    return False


def match_skill_phrases(
    phrases: Iterable[str],
    G: nx.Graph,
    user_texts: Iterable[str] = (),
) -> dict:
    """Resolve router-provided skill phrases through exact safe aliases."""
    index = build_skill_index(G)
    texts = list(user_texts)
    matched: list[dict] = []
    unmatched: list[str] = []
    excluded: list[str] = []
    skill_ids: set[str] = set()

    for raw_phrase in phrases:
        phrase = str(raw_phrase).strip()
        if not phrase:
            continue
        selected_alias = ""
        selected_ids: tuple[str, ...] = ()
        for alias in _phrase_candidates(phrase):
            if alias in index:
                selected_alias = alias
                selected_ids = index[alias]
                break
        if not selected_ids:
            unmatched.append(phrase)
            continue
        if texts and not _is_explicitly_owned(selected_alias, texts):
            excluded.append(phrase)
            continue
        skill_ids.update(selected_ids)
        representative = selected_ids[0]
        matched.append(
            {
                "phrase": phrase,
                "alias": selected_alias,
                "id": representative,
                "title": G.nodes[representative].get("title", phrase),
                "all_ids": list(selected_ids),
            }
        )

    return {
        "skill_ids": skill_ids,
        "matched": matched,
        "unmatched": unmatched,
        "excluded": excluded,
    }


def extract_possessed_skill_phrases(user_texts: Iterable[str], G: nx.Graph) -> list[str]:
    """Fallback extraction from user clauses that explicitly signal possession."""
    aliases = sorted(build_skill_index(G), key=lambda value: (-len(value), value))
    found: set[str] = set()
    for raw_text in user_texts:
        for clause in re.split(r"[.;!?\n]+", str(raw_text)):
            normalized = normalize_label(clause)
            if not normalized or not _POSSESSION_MARKERS.search(normalized):
                continue
            for alias in aliases:
                if len(alias) < 3 and alias not in _SHORT_ALIAS_ALLOWLIST:
                    continue
                match = re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", normalized)
                if not match:
                    continue
                if not _is_explicitly_owned(alias, [raw_text]):
                    continue
                found.add(alias)
    return sorted(found)


def resolve_user_skills(
    router_skills: Iterable[str],
    user_texts: Iterable[str],
    G: nx.Graph,
) -> dict:
    """Resolve explicitly extracted skills plus conservative user-text fallback."""
    texts = [str(text) for text in user_texts if str(text).strip()]
    phrases = [str(phrase) for phrase in router_skills if str(phrase).strip()]
    phrases.extend(extract_possessed_skill_phrases(texts, G))
    deduplicated_by_key: dict[str, str] = {}
    for phrase in phrases:
        deduplicated_by_key.setdefault(normalize_label(phrase), phrase)
    deduplicated = [phrase for key, phrase in deduplicated_by_key.items() if key]
    return match_skill_phrases(deduplicated, G, texts)


def _esco_requirements(role_id: str, G: nx.MultiDiGraph) -> set[str]:
    requirements: set[str] = set()
    for _, target, data in G.out_edges(role_id, data=True):
        if data.get("relation") != "REQUIRES":
            continue
        if data.get("source") != "esco" or data.get("requirement_level") != "essential":
            continue
        if G.has_node(target) and G.nodes[target].get("type") == "skill":
            requirements.add(str(target))
    return requirements


def aligned_esco_role(role_id: str, G: nx.MultiDiGraph) -> str | None:
    """Return the strongest ESCO role aligned with an ONET role."""
    candidates: list[tuple[float, str]] = []
    for source, target, data in G.out_edges(role_id, data=True):
        if data.get("relation") == "SIMILAR_TO" and G.nodes[target].get("source") == "esco":
            candidates.append((float(data.get("similarity", 0.0)), str(target)))
    for source, _, data in G.in_edges(role_id, data=True):
        if data.get("relation") == "SIMILAR_TO" and G.nodes[source].get("source") == "esco":
            candidates.append((float(data.get("similarity", 0.0)), str(source)))
    return max(candidates, default=(0.0, None), key=lambda item: (item[0], item[1] or ""))[1]


def role_requirements(role_id: str, G: nx.MultiDiGraph) -> tuple[set[str], str, str | None]:
    """Return comparable ESCO requirements, evidence status, and evidence role."""
    if not G.has_node(role_id):
        return set(), "role_missing", None
    data = G.nodes[role_id]
    if data.get("source") == "esco":
        required = _esco_requirements(role_id, G)
        return required, "direct_esco" if required else "no_requirements", role_id
    if data.get("source") == "onet":
        aligned = aligned_esco_role(role_id, G)
        if aligned:
            required = _esco_requirements(aligned, G)
            return required, "aligned_esco" if required else "no_requirements", aligned
        return set(), "no_esco_alignment", None
    return set(), "unsupported_role_source", None


def _skill_items(skill_ids: Iterable[str], G: nx.Graph) -> list[dict]:
    items = [
        {"id": skill_id, "title": str(G.nodes[skill_id].get("title", skill_id))}
        for skill_id in skill_ids
        if G.has_node(skill_id)
    ]
    items.sort(key=lambda item: (normalize_label(item["title"]), item["id"]))
    return items


def compute_skill_gap(
    user_skill_ids: Iterable[str],
    role_id: str,
    G: nx.MultiDiGraph,
    display_limit: int = 5,
) -> dict:
    """Compute graph-structural overlap without treating missing evidence as zero."""
    required, evidence, evidence_role_id = role_requirements(role_id, G)
    owned = set(user_skill_ids)
    if not required:
        return {
            "accessibility": None,
            "gap": None,
            "gap_evidence": evidence,
            "evidence_role_id": evidence_role_id,
            "required_skill_count": 0,
            "have_count": 0,
            "need_count": 0,
            "have": [],
            "need": [],
            "have_ids": set(),
            "need_ids": set(),
        }

    have = required.intersection(owned)
    need = required.difference(owned)
    has_user_evidence = bool(owned)
    accessibility = len(have) / len(required) if has_user_evidence else None
    gap = 1.0 - accessibility if accessibility is not None else None
    status = evidence if has_user_evidence else "no_user_skills"
    return {
        "accessibility": accessibility,
        "gap": gap,
        "gap_evidence": status,
        "evidence_role_id": evidence_role_id,
        "required_skill_count": len(required),
        "have_count": len(have),
        "need_count": len(need) if has_user_evidence else 0,
        "have": _skill_items(have, G)[:display_limit],
        "need": _skill_items(need, G)[:display_limit] if has_user_evidence else [],
        "have_ids": have,
        "need_ids": need if has_user_evidence else set(),
    }


def resolve_current_role(
    role_phrase: str,
    G: nx.Graph,
    fuzzy_threshold: float = 0.90,
    ambiguity_margin: float = 0.03,
) -> str | None:
    """Resolve a current-role phrase conservatively to one ESCO role."""
    query = normalize_label(role_phrase)
    if not query:
        return None

    index: dict[str, list[str]] = defaultdict(list)
    for node_id, data in G.nodes(data=True):
        if data.get("type") == "role" and data.get("source") == "esco":
            title = str(data.get("title", ""))
            for alias in _skill_aliases(title):
                index[alias].append(str(node_id))

    exact = sorted(set(index.get(query, [])))
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        return None

    scored = sorted(
        (
            (SequenceMatcher(None, query, alias).ratio(), alias, tuple(sorted(set(node_ids))))
            for alias, node_ids in index.items()
        ),
        reverse=True,
    )
    if not scored or scored[0][0] < fuzzy_threshold or len(scored[0][2]) != 1:
        return None
    if len(scored) > 1 and scored[0][0] - scored[1][0] < ambiguity_margin:
        return None
    return scored[0][2][0]


def transition_evidence(
    source_role_id: str | None,
    target_role_id: str,
    G: nx.MultiDiGraph,
) -> dict | None:
    if not source_role_id or not G.has_node(source_role_id):
        return None
    evidence = [
        data
        for _, target, data in G.out_edges(source_role_id, data=True)
        if str(target) == str(target_role_id)
        and data.get("relation") == "TRANSITIONS_TO"
        and data.get("source") == "karrierewege"
    ]
    if not evidence:
        return None
    best = max(
        evidence,
        key=lambda data: (float(data.get("probability", 0.0)), int(data.get("count", 0))),
    )
    return {
        "from_role_id": source_role_id,
        "count": int(best.get("count", 0)),
        "probability": float(best.get("probability", 0.0)),
        "source_total": int(best.get("source_total", 0)),
    }


def transition_destinations(
    source_role_id: str | None,
    G: nx.MultiDiGraph,
    limit: int = 12,
) -> list[dict]:
    if not source_role_id or not G.has_node(source_role_id) or limit <= 0:
        return []
    rows: list[dict] = []
    for _, target, data in G.out_edges(source_role_id, data=True):
        if data.get("relation") != "TRANSITIONS_TO" or data.get("source") != "karrierewege":
            continue
        rows.append(
            {
                "id": str(target),
                "transition": {
                    "from_role_id": source_role_id,
                    "count": int(data.get("count", 0)),
                    "probability": float(data.get("probability", 0.0)),
                    "source_total": int(data.get("source_total", 0)),
                },
            }
        )
    rows.sort(
        key=lambda row: (
            -row["transition"]["probability"],
            -row["transition"]["count"],
            normalize_label(G.nodes[row["id"]].get("title", "")),
            row["id"],
        )
    )
    return rows[:limit]


def rank_roles_by_gap(
    user_skill_ids: Iterable[str],
    candidates: list[dict],
    G: nx.MultiDiGraph,
    current_role_id: str | None = None,
    display_limit: int = 5,
) -> list[dict]:
    """Annotate and stably order a semantically shortlisted candidate list."""
    ranked: list[dict] = []
    for candidate in candidates:
        role_id = str(candidate["id"])
        gap = compute_skill_gap(user_skill_ids, role_id, G, display_limit=display_limit)
        transition = candidate.get("transition") or transition_evidence(
            current_role_id, role_id, G
        )
        semantic_score = float(candidate.get("rerank_score", candidate.get("score", 0.0)) or 0.0)
        enriched = dict(candidate)
        enriched.update(
            {
                "skill_gap": gap,
                "transition": transition,
                "semantic_score": semantic_score,
            }
        )
        ranked.append(enriched)

    any_accessibility = any(
        candidate["skill_gap"]["accessibility"] is not None for candidate in ranked
    )

    def _key(candidate: dict) -> tuple:
        accessibility = candidate["skill_gap"]["accessibility"]
        transition = candidate.get("transition") or {}
        title = G.nodes[candidate["id"]].get("title", "") if G.has_node(candidate["id"]) else ""
        if any_accessibility:
            evidence_bucket = 0 if accessibility is not None else 1
            accessibility_key = -(accessibility if accessibility is not None else -1.0)
        else:
            evidence_bucket = 0
            accessibility_key = 0.0
        return (
            evidence_bucket,
            accessibility_key,
            -float(transition.get("probability", 0.0)),
            -int(transition.get("count", 0)),
            -candidate["semantic_score"],
            normalize_label(title),
            candidate["id"],
        )

    ranked.sort(key=_key)
    return ranked
