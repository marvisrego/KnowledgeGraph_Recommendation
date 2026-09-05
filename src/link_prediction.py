"""Link prediction for career transition edges using graph-structural features.

Predicts missing TRANSITIONS_TO edges to increase coverage and improve
Hits@K/MRR metrics. Uses a LightGBM classifier trained on graph topology
features from the training split only.

Features per (source, target) role pair:
1. Common Neighbours (shared skills count)
2. Jaccard Coefficient (normalized skill overlap)
3. Adamic-Adar Index (shared skills weighted by inverse log-degree)
4. Resource Allocation Index (shared skills weighted by inverse degree)
5. Preferential Attachment (product of skill counts)
6. Embedding Cosine Similarity (from stored role vectors)
7. Same ISCO Group (binary)
8. IDF-weighted Skill Overlap (rare skills matter more)
9. Neighbour Transition Evidence (do semantically similar roles transition here?)
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

from src.kg_enrichment import role_skills, skill_degree
from src.transition_policy import is_training_transition

if TYPE_CHECKING:
    pass

FEATURE_NAMES = [
    "common_neighbours",
    "jaccard_coefficient",
    "adamic_adar",
    "resource_allocation",
    "preferential_attachment",
    "embedding_cosine_sim",
    "isco_group_distance",     # replaces binary same_isco_group (0=same, 0.5=adjacent, 1=different)
    "idf_skill_overlap",
    "neighbour_evidence",
    "title_tfidf_sim",         # TF-IDF cosine between role title strings
    "meta_path_count",         # roles sharing ≥3 skills with both source and target
]


def _isco_distance(src_isco: str, tgt_isco: str) -> float:
    """ISCO structural distance: 0.0=same group, 0.5=adjacent, 1.0=different.

    Uses numeric 2-digit ISCO codes. Returns 0.5 neutral when either is missing.
    """
    if not src_isco or not tgt_isco:
        return 0.5
    if src_isco == tgt_isco:
        return 0.0
    try:
        diff = abs(int(src_isco) - int(tgt_isco))
        return 0.5 if diff <= 5 else 1.0
    except ValueError:
        return 1.0


def _get_idf(G: nx.MultiDiGraph) -> dict[str, float]:
    """Retrieve IDF values from graph node attributes."""
    return {
        str(nid): float(d.get("idf", 1.0))
        for nid, d in G.nodes(data=True)
        if d.get("type") in ("skill", "element") and "idf" in d
    }


def extract_pair_features(
    source_id: str,
    target_id: str,
    G: nx.MultiDiGraph,
    idf_map: dict[str, float] | None = None,
    embeddings: dict[str, np.ndarray] | None = None,
    transition_index: dict[str, dict[str, float]] | None = None,
    neighbour_roles: dict[str, list[str]] | None = None,
) -> np.ndarray:
    """Extract 9-dimensional feature vector for a (source, target) role pair."""
    if idf_map is None:
        idf_map = _get_idf(G)

    source_skills = role_skills(source_id, G)
    target_skills = role_skills(target_id, G)

    shared = source_skills & target_skills
    union = source_skills | target_skills

    # 1. Common Neighbours
    cn = len(shared)

    # 2. Jaccard Coefficient
    jc = len(shared) / len(union) if union else 0.0

    # 3. Adamic-Adar Index
    aa = 0.0
    for skill_id in shared:
        deg = skill_degree(skill_id, G)
        if deg > 1:
            aa += 1.0 / math.log(deg)

    # 4. Resource Allocation Index
    ra = 0.0
    for skill_id in shared:
        deg = skill_degree(skill_id, G)
        if deg > 0:
            ra += 1.0 / deg

    # 5. Preferential Attachment
    pa = len(source_skills) * len(target_skills)

    # 6. Embedding Cosine Similarity
    cos_sim = 0.0
    if embeddings is not None:
        src_emb = embeddings.get(source_id)
        tgt_emb = embeddings.get(target_id)
        if src_emb is not None and tgt_emb is not None:
            dot = float(np.dot(src_emb, tgt_emb))
            norm_s = float(np.linalg.norm(src_emb))
            norm_t = float(np.linalg.norm(tgt_emb))
            if norm_s > 0 and norm_t > 0:
                cos_sim = dot / (norm_s * norm_t)

    # 7. ISCO Group Distance (replaces binary same_isco_group)
    source_isco = G.nodes.get(source_id, {}).get("isco_2digit", "")
    target_isco = G.nodes.get(target_id, {}).get("isco_2digit", "")
    isco_dist = _isco_distance(source_isco, target_isco)

    # 8. IDF-weighted Skill Overlap
    if target_skills:
        shared_idf = sum(idf_map.get(s, 1.0) for s in shared)
        target_idf = sum(idf_map.get(s, 1.0) for s in target_skills)
        idf_overlap = shared_idf / target_idf if target_idf > 0 else 0.0
    else:
        idf_overlap = 0.0

    # 9. Neighbour Transition Evidence
    ne = 0.0
    if neighbour_roles and transition_index:
        neighbours = neighbour_roles.get(source_id, [])
        for nbr in neighbours[:10]:
            nbr_transitions = transition_index.get(nbr, {})
            prob = nbr_transitions.get(target_id, 0.0)
            if prob > ne:
                ne = prob

    # 10. Title TF-IDF Similarity (0.0 at inference without precomputed matrix)
    title_sim = 0.0

    # 11. Meta-path Count (0.0 at inference without precomputed index)
    meta_path = 0.0

    return np.array([cn, jc, aa, ra, pa, cos_sim, isco_dist, idf_overlap, ne, title_sim, meta_path], dtype=np.float64)


def build_transition_index(G: nx.MultiDiGraph) -> dict[str, dict[str, float]]:
    """Build {source_id: {target_id: probability}} from train TRANSITIONS_TO edges."""
    index: dict[str, dict[str, float]] = {}
    for src, tgt, data in G.edges(data=True):
        if not is_training_transition(data):
            continue
        src_str = str(src)
        tgt_str = str(tgt)
        if src_str not in index:
            index[src_str] = {}
        prob = float(data.get("probability", 0.0))
        if prob > index[src_str].get(tgt_str, 0.0):
            index[src_str][tgt_str] = prob
    return index


def build_neighbour_index(
    G: nx.MultiDiGraph,
    embeddings: dict[str, np.ndarray],
    role_ids: list[str],
    k: int = 10,
) -> dict[str, list[str]]:
    """Build top-K nearest neighbour index for source roles using embeddings."""
    valid_ids = [rid for rid in role_ids if rid in embeddings]
    if not valid_ids:
        return {}

    id_to_idx = {rid: i for i, rid in enumerate(valid_ids)}
    matrix = np.array([embeddings[rid] for rid in valid_ids])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = matrix / norms

    neighbours: dict[str, list[str]] = {}
    batch_size = 200
    for start in range(0, len(valid_ids), batch_size):
        end = min(start + batch_size, len(valid_ids))
        batch = normed[start:end]
        sims = batch @ normed.T
        for local_idx in range(end - start):
            global_idx = start + local_idx
            role_id = valid_ids[global_idx]
            row = sims[local_idx].copy()
            row[global_idx] = -1.0
            top_indices = np.argsort(row)[-k:][::-1]
            neighbours[role_id] = [valid_ids[i] for i in top_indices if row[i] > 0]

    return neighbours


def _precompute_role_skills_cache(G: nx.MultiDiGraph) -> dict[str, set[str]]:
    """Pre-compute skill sets for all roles (avoids repeated graph traversal)."""
    cache: dict[str, set[str]] = {}
    for nid, data in G.nodes(data=True):
        if data.get("type") != "role":
            continue
        skills: set[str] = set()
        for _, target, edata in G.out_edges(nid, data=True):
            if edata.get("relation") == "REQUIRES":
                node_data = G.nodes.get(target)
                if node_data and node_data.get("type") in ("skill", "element"):
                    skills.add(str(target))
        cache[str(nid)] = skills
    return cache


def _precompute_skill_degrees(G: nx.MultiDiGraph) -> dict[str, int]:
    """Pre-compute in-degree (REQUIRES count) for all skills."""
    degrees: dict[str, int] = {}
    for nid, data in G.nodes(data=True):
        if data.get("type") not in ("skill", "element"):
            continue
        count = sum(1 for _, _, ed in G.in_edges(nid, data=True) if ed.get("relation") == "REQUIRES")
        degrees[str(nid)] = count
    return degrees


def extract_pair_features_fast(
    source_id: str,
    target_id: str,
    source_skills: set[str],
    target_skills: set[str],
    idf_map: dict[str, float],
    skill_degrees: dict[str, int],
    embeddings: dict[str, np.ndarray] | None,
    transition_index: dict[str, dict[str, float]] | None,
    neighbour_index: dict[str, list[str]] | None,
    source_isco: str,
    target_isco: str,
    title_vectors: dict[str, np.ndarray] | None = None,
    role_neighbours_3: dict[str, set[str]] | None = None,
) -> np.ndarray:
    """Fast feature extraction using pre-computed caches (11 features)."""
    shared = source_skills & target_skills
    union = source_skills | target_skills

    cn = len(shared)
    jc = len(shared) / len(union) if union else 0.0

    aa = 0.0
    ra = 0.0
    for skill_id in shared:
        deg = skill_degrees.get(skill_id, 1)
        if deg > 1:
            aa += 1.0 / math.log(deg)
        if deg > 0:
            ra += 1.0 / deg

    pa = len(source_skills) * len(target_skills)

    cos_sim = 0.0
    if embeddings is not None:
        src_emb = embeddings.get(source_id)
        tgt_emb = embeddings.get(target_id)
        if src_emb is not None and tgt_emb is not None:
            dot = float(np.dot(src_emb, tgt_emb))
            norm_s = float(np.linalg.norm(src_emb))
            norm_t = float(np.linalg.norm(tgt_emb))
            if norm_s > 0 and norm_t > 0:
                cos_sim = dot / (norm_s * norm_t)

    # 7. ISCO Group Distance
    isco_dist = _isco_distance(source_isco, target_isco)

    if target_skills:
        shared_idf = sum(idf_map.get(s, 1.0) for s in shared)
        target_idf = sum(idf_map.get(s, 1.0) for s in target_skills)
        idf_overlap = shared_idf / target_idf if target_idf > 0 else 0.0
    else:
        idf_overlap = 0.0

    ne = 0.0
    if neighbour_index and transition_index:
        neighbours = neighbour_index.get(source_id, [])
        for nbr in neighbours[:10]:
            nbr_transitions = transition_index.get(nbr, {})
            prob = nbr_transitions.get(target_id, 0.0)
            if prob > ne:
                ne = prob

    # 10. Title TF-IDF Similarity
    title_sim = 0.0
    if title_vectors is not None:
        sv = title_vectors.get(source_id)
        tv = title_vectors.get(target_id)
        if sv is not None and tv is not None:
            title_sim = float(np.dot(sv, tv))  # pre-normalised

    # 11. Meta-path Count (log-normalised roles sharing ≥3 skills with both)
    meta_path = 0.0
    if role_neighbours_3 is not None:
        src_nbrs = role_neighbours_3.get(source_id, set())
        tgt_nbrs = role_neighbours_3.get(target_id, set())
        common = src_nbrs & tgt_nbrs
        meta_path = math.log2(1 + len(common))

    return np.array([cn, jc, aa, ra, pa, cos_sim, isco_dist, idf_overlap, ne, title_sim, meta_path], dtype=np.float64)


def precompute_title_vectors(G: nx.MultiDiGraph) -> dict[str, np.ndarray]:
    """Build L2-normalised TF-IDF title vectors for all role nodes.

    Uses unigram + bigram features over role titles. Returns {role_id: unit_vector}.
    Falls back to empty dict if sklearn is unavailable.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        return {}

    role_ids = []
    titles = []
    for nid, data in G.nodes(data=True):
        if data.get("type") == "role":
            role_ids.append(str(nid))
            titles.append(str(data.get("title", "")))

    if not role_ids:
        return {}

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=8000, sublinear_tf=True)
    matrix = vectorizer.fit_transform(titles)

    # L2-normalise rows in-place (sparse → dense per row)
    vectors: dict[str, np.ndarray] = {}
    for i, rid in enumerate(role_ids):
        vec = matrix[i].toarray().flatten().astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        vectors[rid] = vec

    return vectors


def precompute_role_neighbours_3(
    skills_cache: dict[str, set[str]],
    role_ids: list[str],
    min_shared: int = 3,
) -> dict[str, set[str]]:
    """For each role, find all roles sharing ≥ min_shared essential skills.

    Used for meta-path count feature. O(roles²) — run once at training time.
    """
    result: dict[str, set[str]] = {rid: set() for rid in role_ids}
    ids = [r for r in role_ids if r in skills_cache]
    for i, src in enumerate(ids):
        src_skills = skills_cache[src]
        for tgt in ids[i + 1:]:
            if len(src_skills & skills_cache[tgt]) >= min_shared:
                result[src].add(tgt)
                result[tgt].add(src)
    return result


def build_training_data(
    G: nx.MultiDiGraph,
    idf_map: dict[str, float],
    embeddings: dict[str, np.ndarray] | None = None,
    neg_ratio: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
    """Build feature matrix and labels from train-split transition edges.

    Positives: existing TRANSITIONS_TO edges.
    Negatives: sampled non-edges (neg_ratio per positive).

    Returns (X, y, pairs) where pairs is [(source, target), ...].
    """
    rng = np.random.default_rng(seed)

    # Pre-compute caches for fast feature extraction
    skills_cache = _precompute_role_skills_cache(G)
    skill_degrees = _precompute_skill_degrees(G)
    isco_cache = {str(nid): str(d.get("isco_2digit", "")) for nid, d in G.nodes(data=True)}

    transition_index = build_transition_index(G)
    esco_roles = [
        str(nid) for nid, d in G.nodes(data=True)
        if d.get("type") == "role" and d.get("source") == "esco"
    ]
    esco_set = set(esco_roles)

    source_roles = sorted(transition_index.keys())
    neighbour_index = build_neighbour_index(G, embeddings or {}, source_roles, k=40)

    # New: TF-IDF title vectors and meta-path precomputation
    print("[link_prediction] Precomputing title TF-IDF vectors …")
    title_vectors = precompute_title_vectors(G)
    all_role_ids = [str(nid) for nid, d in G.nodes(data=True) if d.get("type") == "role"]
    print("[link_prediction] Precomputing meta-path neighbour sets (>=3 shared skills) ...")
    role_neighbours_3 = precompute_role_neighbours_3(skills_cache, all_role_ids)

    positives: list[tuple[str, str]] = []
    for src, targets in transition_index.items():
        for tgt in targets:
            if tgt in esco_set:
                positives.append((src, tgt))

    positive_set = set(positives)
    target_count = len(positives) * neg_ratio

    # Hard negatives: semantic neighbours with no training transition (50% of negatives)
    hard_target = target_count // 2
    hard_negatives: list[tuple[str, str]] = []
    hard_seen: set[tuple[str, str]] = set()
    for src in source_roles:
        neighbours = neighbour_index.get(src, [])
        for nbr in neighbours:
            pair = (src, nbr)
            if nbr in esco_set and pair not in positive_set and pair not in hard_seen:
                hard_negatives.append(pair)
                hard_seen.add(pair)
                if len(hard_negatives) >= hard_target:
                    break
        if len(hard_negatives) >= hard_target:
            break

    # Random negatives for the remaining 50%
    random_target = target_count - len(hard_negatives)
    random_negatives: list[tuple[str, str]] = []
    seen_all = positive_set | hard_seen
    attempts = 0
    max_attempts = random_target * 20
    while len(random_negatives) < random_target and attempts < max_attempts:
        src = rng.choice(source_roles)
        tgt = rng.choice(esco_roles)
        pair = (src, tgt)
        if src != tgt and pair not in seen_all:
            random_negatives.append(pair)
            seen_all.add(pair)
        attempts += 1

    negatives = hard_negatives + random_negatives
    all_pairs = positives + negatives
    labels = np.array([1] * len(positives) + [0] * len(negatives), dtype=np.int32)

    print(f"[link_prediction] Training data: {len(positives)} positives, "
          f"{len(hard_negatives)} hard negatives, {len(random_negatives)} random negatives")

    features = np.zeros((len(all_pairs), len(FEATURE_NAMES)), dtype=np.float64)
    for i, (src, tgt) in enumerate(all_pairs):
        features[i] = extract_pair_features_fast(
            src, tgt,
            skills_cache.get(src, set()),
            skills_cache.get(tgt, set()),
            idf_map, skill_degrees, embeddings,
            transition_index, neighbour_index,
            isco_cache.get(src, ""),
            isco_cache.get(tgt, ""),
            title_vectors=title_vectors,
            role_neighbours_3=role_neighbours_3,
        )

    return features, labels, all_pairs


def train_link_predictor(
    X: np.ndarray,
    y: np.ndarray,
    params: dict | None = None,
) -> object:
    """Train a LightGBM binary classifier for link prediction.

    Returns trained booster object.
    """
    import lightgbm as lgb

    if params is None:
        params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": 8,
            "min_child_samples": 20,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "seed": 42,
        }

    dataset = lgb.Dataset(X, label=y, feature_name=FEATURE_NAMES)
    booster = lgb.train(
        params,
        dataset,
        num_boost_round=300,
        valid_sets=[dataset],
        callbacks=[lgb.log_evaluation(period=0)],
    )
    return booster


def predict_transitions(
    model: object,
    source_id: str,
    candidate_targets: list[str],
    G: nx.MultiDiGraph,
    idf_map: dict[str, float],
    embeddings: dict[str, np.ndarray] | None = None,
    transition_index: dict[str, dict[str, float]] | None = None,
    neighbour_index: dict[str, list[str]] | None = None,
) -> list[tuple[str, float]]:
    """Predict transition scores for a source role to multiple candidates.

    Returns sorted list of (target_id, score) descending by score.
    """
    if not candidate_targets:
        return []

    features = np.zeros((len(candidate_targets), len(FEATURE_NAMES)), dtype=np.float64)
    for i, tgt in enumerate(candidate_targets):
        features[i] = extract_pair_features(
            source_id, tgt, G, idf_map, embeddings,
            transition_index, neighbour_index,
        )

    scores = model.predict(features)
    results = [(tgt, float(scores[i])) for i, tgt in enumerate(candidate_targets)]
    results.sort(key=lambda x: -x[1])
    return results


def link_prediction_map(
    model: object,
    G: nx.MultiDiGraph,
    idf_map: dict[str, float],
    embeddings: dict[str, np.ndarray] | None = None,
    top_k: int = 50,
    source_role_ids: list[str] | None = None,
) -> dict[str, list[str]]:
    """Generate ranked transition predictions for ESCO roles.

    If source_role_ids is provided, only predict for those roles (faster).
    Otherwise predicts for all ESCO roles with transition data.

    Returns {normalized_source_title: [normalized_target_title, ...]}.
    Compatible with evaluate_prediction_map().
    """
    from src.text_normalization import normalize_label

    esco_roles = [
        (str(nid), str(d.get("title", "")))
        for nid, d in G.nodes(data=True)
        if d.get("type") == "role" and d.get("source") == "esco"
    ]
    esco_ids = [rid for rid, _ in esco_roles]
    id_to_title = {rid: title for rid, title in esco_roles}

    transition_index = build_transition_index(G)
    tr_source_roles = sorted(transition_index.keys())
    neighbour_index = build_neighbour_index(G, embeddings or {}, tr_source_roles, k=40)

    # Pre-compute caches for fast prediction
    skills_cache = _precompute_role_skills_cache(G)
    skill_degrees = _precompute_skill_degrees(G)
    isco_cache = {str(nid): str(d.get("isco_2digit", "")) for nid, d in G.nodes(data=True)}

    # New features: title TF-IDF and meta-path
    title_vectors = precompute_title_vectors(G)
    all_role_ids = [str(nid) for nid, d in G.nodes(data=True) if d.get("type") == "role"]
    role_neighbours_3 = precompute_role_neighbours_3(skills_cache, all_role_ids)

    # Determine which source roles to predict for
    if source_role_ids is not None:
        predict_roles = [(rid, id_to_title.get(rid, "")) for rid in source_role_ids if rid in id_to_title]
    else:
        predict_roles = [(rid, id_to_title[rid]) for rid in tr_source_roles if rid in id_to_title]

    predictions: dict[str, list[str]] = {}
    for idx, (source_id, source_title) in enumerate(predict_roles):
        if idx % 100 == 0 and idx > 0:
            print(f"  Predicting: {idx}/{len(predict_roles)} roles...")

        candidates = [rid for rid in esco_ids if rid != source_id]
        if not candidates:
            continue

        # Fast batch feature extraction
        n_cand = len(candidates)
        features = np.zeros((n_cand, len(FEATURE_NAMES)), dtype=np.float64)
        source_skills = skills_cache.get(source_id, set())
        source_isco = isco_cache.get(source_id, "")

        for i, tgt in enumerate(candidates):
            features[i] = extract_pair_features_fast(
                source_id, tgt,
                source_skills, skills_cache.get(tgt, set()),
                idf_map, skill_degrees, embeddings,
                transition_index, neighbour_index,
                source_isco, isco_cache.get(tgt, ""),
                title_vectors=title_vectors,
                role_neighbours_3=role_neighbours_3,
            )

        scores = model.predict(features)
        top_indices = np.argsort(scores)[-top_k:][::-1]

        source_label = normalize_label(source_title)
        if source_label:
            predictions[source_label] = [
                normalize_label(id_to_title[candidates[i]])
                for i in top_indices
                if normalize_label(id_to_title.get(candidates[i], ""))
            ]

    return predictions


def save_model(model: object, path: Path) -> None:
    """Save trained model to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(path: Path) -> object:
    """Load trained model from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)
