"""Graph-provenance faithfulness verification for LLM-generated recommendations.

Checks that every entity (role/skill) the LLM cites in its response actually
exists in the knowledge graph and is reachable from the user's anchor nodes.

Faithfulness score = |matched & reachable entities| / |total entities mentioned|
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from typing import Iterable

import networkx as nx

from src.text_normalization import normalize_label


@dataclass
class EntityMatch:
    text: str
    node_id: str | None
    node_type: str | None
    reachable: bool


@dataclass
class FaithfulnessResult:
    score: float
    total_entities: int
    matched_count: int
    reachable_count: int
    unmatched: list[str]
    unreachable: list[str]
    matches: list[EntityMatch]

    def to_dict(self) -> dict:
        return {
            "faithfulness_score": round(self.score, 4),
            "total_entities": self.total_entities,
            "matched_count": self.matched_count,
            "reachable_count": self.reachable_count,
            "unmatched_entities": self.unmatched,
            "unreachable_entities": self.unreachable,
        }


def extract_entities(text: str) -> list[str]:
    """Extract bolded and candidate entity phrases from LLM output.

    Prioritizes **bold** text (the LLM is instructed to bold role names).
    Falls back to capitalized multi-word phrases as potential entity candidates.
    """
    entities: list[str] = []
    seen: set[str] = set()

    def _add(phrase: str) -> None:
        phrase = phrase.strip(" \t\r\n.,;:!?()[]{}\"'“”‘’")
        phrase = re.sub(
            r"^(?:your|my|their|our|the|a|an)\s+",
            "",
            phrase,
            flags=re.IGNORECASE,
        )
        slash_parts = [part.strip() for part in phrase.split("/")]
        if (
            len(slash_parts) > 1
            and all(re.fullmatch(r"[A-Za-z][A-Za-z0-9+#.-]*", part) for part in slash_parts)
            and not all(part.isupper() for part in slash_parts)
        ):
            for part in slash_parts:
                _add(part)
            return
        normalized = normalize_label(phrase)
        if normalized in {"a", "an", "the", "your", "my", "their", "our"}:
            return
        if normalized and normalized not in seen and len(normalized) > 2:
            entities.append(phrase)
            seen.add(normalized)

    bold_pattern = re.compile(r"\*\*([^*]+)\*\*")
    for match in bold_pattern.finditer(text):
        _add(match.group(1))

    # Quoted role/skill candidates.
    for match in re.finditer(r"[\"“]([^\"”]{3,80})[\"”]", text):
        _add(match.group(1))

    # Capitalized multi-word entities and in-sentence technology names such as
    # Python. Sentence-opening prose words are intentionally ignored.
    capitalized = re.compile(
        r"\b[A-Z][A-Za-z0-9+#./-]*(?:\s+[A-Z][A-Za-z0-9+#./-]*){0,4}\b"
    )
    for match in capitalized.finditer(text):
        phrase = match.group(0)
        is_multiword = " " in phrase
        is_technical_token = phrase.isupper() or bool(re.search(r"[0-9+#./-]", phrase))
        before = text[: match.start()].rstrip()
        is_sentence_start = not before or before[-1:] in ".!?\n"
        if is_multiword or is_technical_token or not is_sentence_start:
            _add(phrase)

    # Lower-case candidates explicitly introduced as skills or development
    # targets. Stop at punctuation or a coordinating conjunction.
    cue_pattern = re.compile(
        r"\b(?:skill(?:s)?\s+(?:such\s+as|like|including)|"
        r"learn|develop|using|use|know|knowledge\s+of|experience\s+with|proficient\s+in)"
        r"\s+([A-Za-z][A-Za-z0-9+#./-]*(?:\s+[A-Za-z][A-Za-z0-9+#./-]*){0,3})",
        re.IGNORECASE,
    )
    for match in cue_pattern.finditer(text):
        phrase = re.split(r"\s+(?:and|but|because|while|to)\s+", match.group(1), maxsplit=1)[0]
        phrase = re.sub(r"^(?:your|my|their|the|a|an)\s+", "", phrase, flags=re.IGNORECASE)
        phrase = re.sub(r"\s+(?:skills?|knowledge|experience)$", "", phrase, flags=re.IGNORECASE)
        _add(phrase)

    return entities


def build_entity_index(G: nx.MultiDiGraph) -> dict[str, str]:
    """Build normalized title → node_id index for roles and skills."""
    index: dict[str, str] = {}
    for nid, data in G.nodes(data=True):
        node_type = data.get("type", "")
        if node_type not in ("role", "skill", "element"):
            continue
        title = str(data.get("title", ""))
        normalized = normalize_label(title)
        if normalized:
            index[normalized] = str(nid)
        short_title = re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()
        short_normalized = normalize_label(short_title)
        if short_normalized:
            index.setdefault(short_normalized, str(nid))
    return index


def match_entities_to_graph(
    entities: list[str],
    G: nx.MultiDiGraph,
    entity_index: dict[str, str] | None = None,
) -> list[EntityMatch]:
    """Match extracted entity phrases to graph nodes via normalized lookup."""
    if entity_index is None:
        entity_index = build_entity_index(G)

    matches: list[EntityMatch] = []
    for phrase in entities:
        normalized = normalize_label(phrase)
        node_id = entity_index.get(normalized)
        if node_id and G.has_node(node_id):
            node_type = G.nodes[node_id].get("type", "unknown")
            matches.append(EntityMatch(
                text=phrase,
                node_id=node_id,
                node_type=node_type,
                reachable=False,
            ))
        else:
            matches.append(EntityMatch(
                text=phrase,
                node_id=None,
                node_type=None,
                reachable=False,
            ))

    return matches


def check_reachability(
    node_id: str,
    anchor_ids: Iterable[str],
    G: nx.MultiDiGraph,
    max_hops: int = 3,
) -> bool:
    """BFS: is node_id reachable from any anchor within max_hops?

    Traverses both outgoing and incoming edges (undirected reachability).
    """
    anchors = set(str(a) for a in anchor_ids)
    if str(node_id) in anchors:
        return True

    for anchor in anchors:
        if not G.has_node(anchor):
            continue
        visited: set[str] = {anchor}
        queue: deque[tuple[str, int]] = deque([(anchor, 0)])
        while queue:
            current, depth = queue.popleft()
            if depth >= max_hops:
                continue
            for neighbor in set(G.successors(current)) | set(G.predecessors(current)):
                nbr = str(neighbor)
                if nbr == str(node_id):
                    return True
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append((nbr, depth + 1))

    return False


def compute_faithfulness(
    llm_output: str,
    anchor_role_ids: list[str],
    G: nx.MultiDiGraph,
    max_hops: int = 3,
    entity_index: dict[str, str] | None = None,
) -> FaithfulnessResult:
    """Compute faithfulness score for LLM output against the knowledge graph.

    Faithfulness = |entities that are both matched in graph AND reachable from anchors|
                   / |total entities mentioned|
    """
    entities = extract_entities(llm_output)

    if not entities:
        return FaithfulnessResult(
            score=1.0,
            total_entities=0,
            matched_count=0,
            reachable_count=0,
            unmatched=[],
            unreachable=[],
            matches=[],
        )

    matches = match_entities_to_graph(entities, G, entity_index)

    unmatched: list[str] = []
    unreachable: list[str] = []
    reachable_count = 0
    matched_count = 0

    for match in matches:
        if match.node_id is None:
            unmatched.append(match.text)
            continue
        matched_count += 1
        is_reachable = check_reachability(match.node_id, anchor_role_ids, G, max_hops)
        match.reachable = is_reachable
        if is_reachable:
            reachable_count += 1
        else:
            unreachable.append(match.text)

    total = len(entities)
    score = reachable_count / total if total > 0 else 1.0

    return FaithfulnessResult(
        score=score,
        total_entities=total,
        matched_count=matched_count,
        reachable_count=reachable_count,
        unmatched=unmatched,
        unreachable=unreachable,
        matches=matches,
    )
