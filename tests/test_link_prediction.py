"""Unit tests for src/link_prediction.py — feature extraction and training data."""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx
import numpy as np
from src.kg_enrichment import compute_skill_idf, compute_isco_2digit
from src.link_prediction import (
    FEATURE_NAMES,
    build_training_data,
    build_transition_index,
    extract_pair_features,
)


def _build_lp_graph() -> nx.MultiDiGraph:
    """Graph for link prediction testing."""
    G = nx.MultiDiGraph()

    # 4 roles
    G.add_node("r1", type="role", source="esco", title="Analyst", isco_group="2511")
    G.add_node("r2", type="role", source="esco", title="Scientist", isco_group="2120")
    G.add_node("r3", type="role", source="esco", title="Engineer", isco_group="2511")
    G.add_node("r4", type="role", source="esco", title="Manager", isco_group="1120")

    # 5 skills
    G.add_node("sk1", type="skill", source="esco", title="Python")
    G.add_node("sk2", type="skill", source="esco", title="SQL")
    G.add_node("sk3", type="skill", source="esco", title="ML")
    G.add_node("sk4", type="skill", source="esco", title="Leadership")
    G.add_node("sk5", type="skill", source="esco", title="Statistics")

    # r1 requires: Python, SQL, Statistics
    for sk in ("sk1", "sk2", "sk5"):
        G.add_edge("r1", sk, key="esco:REQUIRES", relation="REQUIRES", source="esco", requirement_level="essential")
    # r2 requires: Python, SQL, ML, Statistics
    for sk in ("sk1", "sk2", "sk3", "sk5"):
        G.add_edge("r2", sk, key="esco:REQUIRES", relation="REQUIRES", source="esco", requirement_level="essential")
    # r3 requires: Python, ML
    for sk in ("sk1", "sk3"):
        G.add_edge("r3", sk, key="esco:REQUIRES", relation="REQUIRES", source="esco", requirement_level="essential")
    # r4 requires: Leadership
    G.add_edge("r4", "sk4", key="esco:REQUIRES", relation="REQUIRES", source="esco", requirement_level="essential")

    # Transition: r1 → r2
    G.add_edge("r1", "r2", key="karrierewege:TRANSITIONS_TO",
               relation="TRANSITIONS_TO", source="karrierewege",
               probability=0.15, count=100, source_total=667, split="train")

    # Enrich
    compute_skill_idf(G)
    compute_isco_2digit(G)
    return G


class TestFeatureExtraction(unittest.TestCase):

    def test_feature_dimension(self):
        G = _build_lp_graph()
        features = extract_pair_features("r1", "r2", G)
        self.assertEqual(len(features), len(FEATURE_NAMES))

    def test_common_neighbours(self):
        G = _build_lp_graph()
        features = extract_pair_features("r1", "r2", G)
        # r1 and r2 share: Python, SQL, Statistics = 3
        self.assertEqual(features[0], 3.0)

    def test_jaccard(self):
        G = _build_lp_graph()
        features = extract_pair_features("r1", "r2", G)
        # r1 has 3 skills, r2 has 4 skills, share 3 → Jaccard = 3/4
        self.assertAlmostEqual(features[1], 3.0 / 4.0)

    def test_isco_group_distance(self):
        G = _build_lp_graph()
        # r1 (isco_2digit=25) and r3 (isco_2digit=25) → distance = 0.0 (same group)
        features_same = extract_pair_features("r1", "r3", G)
        self.assertEqual(features_same[6], 0.0)
        # r1 (25) and r2 (21) → diff=4 ≤ 5 → distance = 0.5 (adjacent)
        features_adj = extract_pair_features("r1", "r2", G)
        self.assertEqual(features_adj[6], 0.5)
        # r1 (25) and r4 (11) → diff=14 > 5 → distance = 1.0 (different)
        features_diff = extract_pair_features("r1", "r4", G)
        self.assertEqual(features_diff[6], 1.0)

    def test_no_overlap(self):
        G = _build_lp_graph()
        # r1 (Python, SQL, Stats) vs r4 (Leadership) → no shared skills
        features = extract_pair_features("r1", "r4", G)
        self.assertEqual(features[0], 0.0)  # CN
        self.assertEqual(features[1], 0.0)  # Jaccard
        self.assertEqual(features[2], 0.0)  # AA
        self.assertEqual(features[3], 0.0)  # RA

    def test_preferential_attachment(self):
        G = _build_lp_graph()
        features = extract_pair_features("r1", "r2", G)
        # r1 has 3 skills, r2 has 4 skills → PA = 12
        self.assertEqual(features[4], 12.0)

    def test_with_embeddings(self):
        G = _build_lp_graph()
        emb = {
            "r1": np.array([1.0, 0.0, 0.0]),
            "r2": np.array([0.9, 0.1, 0.0]),
        }
        features = extract_pair_features("r1", "r2", G, embeddings=emb)
        # Cosine similarity should be > 0.9
        self.assertGreater(features[5], 0.9)


class TestTransitionIndex(unittest.TestCase):

    def test_index_structure(self):
        G = _build_lp_graph()
        idx = build_transition_index(G)
        self.assertIn("r1", idx)
        self.assertIn("r2", idx["r1"])
        self.assertAlmostEqual(idx["r1"]["r2"], 0.15)

    def test_no_reverse(self):
        G = _build_lp_graph()
        idx = build_transition_index(G)
        # No edge r2 → r1
        self.assertNotIn("r2", idx)

    def test_held_out_edges_are_excluded(self):
        G = _build_lp_graph()
        G.add_edge(
            "r1", "r3", key="karrierewege:test",
            relation="TRANSITIONS_TO", source="karrierewege", split="test",
            probability=0.9, count=90, source_total=100,
        )
        idx = build_transition_index(G)
        self.assertNotIn("r3", idx["r1"])


class TestBuildTrainingData(unittest.TestCase):

    def test_positive_labels(self):
        G = _build_lp_graph()
        idf = {nid: d.get("idf", 1.0) for nid, d in G.nodes(data=True) if "idf" in d}
        X, y, pairs = build_training_data(G, idf, neg_ratio=2)
        # Should have 1 positive (r1→r2), rest negative
        self.assertEqual(y.sum(), 1)
        self.assertEqual(len(y), 1 + 2)  # 1 positive + 2 negatives

    def test_feature_shape(self):
        G = _build_lp_graph()
        idf = {nid: d.get("idf", 1.0) for nid, d in G.nodes(data=True) if "idf" in d}
        X, y, pairs = build_training_data(G, idf, neg_ratio=3)
        self.assertEqual(X.shape[1], len(FEATURE_NAMES))
        self.assertEqual(X.shape[0], len(y))
        self.assertEqual(X.shape[0], len(pairs))

    def test_no_leakage(self):
        G = _build_lp_graph()
        idf = {nid: d.get("idf", 1.0) for nid, d in G.nodes(data=True) if "idf" in d}
        X, y, pairs = build_training_data(G, idf, neg_ratio=5)
        # Positive pair should be in the list
        self.assertIn(("r1", "r2"), pairs)
        # No self-transitions
        for src, tgt in pairs:
            self.assertNotEqual(src, tgt)


if __name__ == "__main__":
    unittest.main()
