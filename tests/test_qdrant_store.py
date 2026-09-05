from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np

from src.qdrant_store import (
    PreparedVector,
    QdrantCollectionAdapter,
    load_prepared_vectors,
    qdrant_point_id,
    role_payload,
    save_prepared_vectors,
)


class _Client:
    def __init__(self) -> None:
        self.records = {
            qdrant_point_id("g", "a"): SimpleNamespace(
                id=qdrant_point_id("g", "a"),
                payload={"graph_id": "g", "record_id": "a", "text": "Alpha", "title": "Alpha"},
                vector=[1.0, 0.0],
            ),
            qdrant_point_id("g", "b"): SimpleNamespace(
                id=qdrant_point_id("g", "b"),
                payload={"graph_id": "g", "record_id": "b", "text": "Beta", "title": "Beta"},
                vector=[0.0, 1.0],
            ),
        }

    def retrieve(self, collection_name, ids, with_payload, with_vectors):
        return [self.records[item] for item in ids if item in self.records]

    def query_points(self, **kwargs):
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    id=self.records[qdrant_point_id("g", "a")].id,
                    payload=self.records[qdrant_point_id("g", "a")].payload,
                    score=0.8,
                )
            ]
        )


class QdrantStoreTests(unittest.TestCase):
    def test_point_ids_are_stable_and_namespaced(self) -> None:
        self.assertEqual(qdrant_point_id("g", "a"), qdrant_point_id("g", "a"))
        self.assertNotEqual(qdrant_point_id("g", "a"), qdrant_point_id("other", "a"))

    def test_payload_contains_required_metadata(self) -> None:
        payload = role_payload(
            "g", "role", {"type": "role", "source": "esco", "title": "Role", "esco_code": "1"}, "Role text"
        )
        for key in ("graph_id", "record_id", "source", "entity_type", "text", "chunk_id"):
            self.assertIn(key, payload)

    def test_adapter_get_preserves_requested_order_and_query_shape(self) -> None:
        adapter = QdrantCollectionAdapter(_Client(), "roles", "g")
        fetched = adapter.get(ids=["b", "missing", "a"], include=["embeddings"])
        self.assertEqual(fetched["ids"], ["b", "a"])
        self.assertEqual(fetched["embeddings"], [[0.0, 1.0], [1.0, 0.0]])

        result = adapter.query(query_embeddings=[[1.0, 0.0]], n_results=3)
        self.assertEqual(result["ids"], [["a"]])
        self.assertAlmostEqual(result["distances"][0][0], 0.2)

    def test_prepared_vector_cache_requires_matching_graph_and_model(self) -> None:
        graph = nx.MultiDiGraph()
        graph.add_node("a", type="role", source="esco", title="Alpha")
        point = PreparedVector(
            "a",
            qdrant_point_id("g", "a"),
            np.array([1.0, 0.0], dtype=np.float32),
            {"record_id": "a"},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.pkl"
            save_prepared_vectors([point], path, graph_id="g", embedding_model="m")
            loaded = load_prepared_vectors(
                path, graph, graph_id="g", embedding_model="m"
            )
            rejected = load_prepared_vectors(
                path, graph, graph_id="g", embedding_model="different"
            )

        self.assertEqual([item.record_id for item in loaded], ["a"])
        self.assertEqual(rejected, [])


if __name__ == "__main__":
    unittest.main()
