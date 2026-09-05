"""Qdrant Cloud persistence with a Chroma-compatible runtime adapter."""

from __future__ import annotations

import math
import os
import pickle
import tempfile
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np

from src.embeddings_index import build_role_text, embed_texts


@dataclass(frozen=True)
class PreparedVector:
    record_id: str
    point_id: str
    vector: np.ndarray
    payload: dict[str, Any]


def save_prepared_vectors(
    points: Sequence[PreparedVector],
    path,
    *,
    graph_id: str,
    embedding_model: str,
) -> None:
    """Atomically cache generated vectors for a validated publish retry."""
    from pathlib import Path

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "graph_id": str(graph_id),
        "embedding_model": str(embedding_model),
        "record_ids": [point.record_id for point in points],
        "points": list(points),
    }
    handle = None
    temporary = None
    try:
        handle, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        with os.fdopen(handle, "wb") as stream:
            handle = None
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if handle is not None:
            os.close(handle)
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def load_prepared_vectors(
    path,
    graph: nx.MultiDiGraph,
    *,
    graph_id: str,
    embedding_model: str,
) -> list[PreparedVector]:
    """Load a cache only when its identity, model, role IDs, and vectors match."""
    from pathlib import Path

    cache_path = Path(path)
    try:
        with cache_path.open("rb") as stream:
            payload = pickle.load(stream)
    except FileNotFoundError:
        return []
    except Exception as exc:
        raise RuntimeError("Prepared-vector cache could not be read") from exc

    expected_ids = sorted(
        str(node_id)
        for node_id, data in graph.nodes(data=True)
        if data.get("type") == "role"
    )
    if (
        payload.get("version") != 1
        or payload.get("graph_id") != str(graph_id)
        or payload.get("embedding_model") != str(embedding_model)
        or payload.get("record_ids") != expected_ids
    ):
        return []
    points = payload.get("points")
    if not isinstance(points, list) or len(points) != len(expected_ids):
        return []
    dimensions = set()
    for expected_id, point in zip(expected_ids, points):
        if not isinstance(point, PreparedVector) or point.record_id != expected_id:
            return []
        vector = np.asarray(point.vector)
        if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
            return []
        dimensions.add(int(vector.size))
    return points if len(dimensions) == 1 else []


def validate_qdrant_settings(settings) -> None:
    missing = [
        name
        for name, value in (
            ("VECTOR_ENDPOINT", settings.vector_endpoint),
            ("VECTOR_PASS", settings.vector_pass),
            ("KG_ID", settings.kg_id),
        )
        if not str(value or "").strip()
    ]
    if missing:
        raise RuntimeError(f"Missing required Qdrant settings: {', '.join(missing)}")
    if not str(settings.qdrant_collection or "").strip():
        raise RuntimeError("QDRANT_COLLECTION must not be empty")
    if settings.qdrant_batch_size < 1:
        raise RuntimeError("QDRANT_BATCH_SIZE must be positive")


def qdrant_point_id(graph_id: str, record_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"career-graph::{graph_id}::{record_id}"))


def create_qdrant_client(settings):
    validate_qdrant_settings(settings)
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(
            url=settings.vector_endpoint,
            api_key=settings.vector_pass,
            timeout=60,
            prefer_grpc=False,
        )
        client.get_collections()
        return client
    except Exception as exc:
        raise RuntimeError("Qdrant Cloud connectivity check failed") from exc


def _payload_scalar(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item") and callable(value.item):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item is not None]
    return str(value)


def role_payload(
    graph_id: str,
    record_id: str,
    data: Mapping[str, Any],
    text: str,
) -> dict[str, Any]:
    payload = {
        "graph_id": graph_id,
        "record_id": record_id,
        "source": str(data.get("source", "")),
        "entity_type": str(data.get("type", "role")),
        "title": str(data.get("title", record_id)),
        "text": text,
        "chunk_id": f"{record_id}::role::0",
    }
    for key in (
        "description",
        "esco_code",
        "isco_group",
        "isco_2digit",
        "job_zone",
        "job_zone_title",
        "typical_education_level",
        "green_share",
        "is_research_occupation",
        "interests",
        "interests_keywords",
    ):
        converted = _payload_scalar(data.get(key))
        if converted is not None and converted != "" and converted != []:
            payload[key] = converted
    return payload


def prepare_role_vectors(
    graph: nx.MultiDiGraph,
    settings,
) -> list[PreparedVector]:
    """Generate one vector per live role using the existing embedding model."""
    validate_qdrant_settings(settings)
    role_rows = sorted(
        (
            (str(node_id), dict(data))
            for node_id, data in graph.nodes(data=True)
            if data.get("type") == "role"
        ),
        key=lambda row: row[0],
    )
    prepared: list[PreparedVector] = []
    batch_size = settings.embed_batch_size
    for offset in range(0, len(role_rows), batch_size):
        batch = role_rows[offset : offset + batch_size]
        texts = [build_role_text(record_id, data) for record_id, data in batch]
        vectors = embed_texts(texts, settings)
        if len(vectors) != len(batch):
            raise RuntimeError("Embedding API returned an unexpected vector count")
        for (record_id, data), text, raw_vector in zip(batch, texts, vectors):
            vector = np.asarray(raw_vector, dtype=np.float32)
            if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
                raise RuntimeError(f"Invalid embedding returned for role {record_id!r}")
            if prepared and vector.size != prepared[0].vector.size:
                raise RuntimeError("Embedding API returned inconsistent vector dimensions")
            prepared.append(
                PreparedVector(
                    record_id=record_id,
                    point_id=qdrant_point_id(str(settings.kg_id), record_id),
                    vector=vector,
                    payload=role_payload(str(settings.kg_id), record_id, data, text),
                )
            )
        print(f"[qdrant_store] Prepared {min(offset + batch_size, len(role_rows))}/{len(role_rows)} role vectors")
    return prepared


class PreparedVectorCollection:
    """Minimal ID-vector surface for alignment before cloud publication."""

    def __init__(self, points: Sequence[PreparedVector]) -> None:
        self._points = {point.record_id: point for point in points}

    def get(self, ids=None, include=None) -> dict[str, list]:
        selected = sorted(self._points) if ids is None else [str(item) for item in ids]
        selected = [record_id for record_id in selected if record_id in self._points]
        return {
            "ids": selected,
            "embeddings": [self._points[record_id].vector.tolist() for record_id in selected],
            "metadatas": [dict(self._points[record_id].payload) for record_id in selected],
            "documents": [self._points[record_id].payload.get("text", "") for record_id in selected],
        }


def alignment_from_prepared_vectors(
    graph: nx.MultiDiGraph,
    points: Sequence[PreparedVector],
    threshold: float,
    block_size: int = 256,
) -> list[tuple[str, str, float]]:
    """Compute O*NET-to-ESCO cosine alignment without re-embedding roles."""
    vectors = {point.record_id: point.vector for point in points}
    onet_ids = sorted(
        str(node_id)
        for node_id, data in graph.nodes(data=True)
        if data.get("type") == "role" and data.get("source") == "onet" and str(node_id) in vectors
    )
    esco_ids = sorted(
        str(node_id)
        for node_id, data in graph.nodes(data=True)
        if data.get("type") == "role" and data.get("source") == "esco" and str(node_id) in vectors
    )
    if not onet_ids or not esco_ids:
        return []
    onet = np.stack([vectors[node_id] for node_id in onet_ids]).astype(np.float32)
    esco = np.stack([vectors[node_id] for node_id in esco_ids]).astype(np.float32)
    onet /= np.linalg.norm(onet, axis=1, keepdims=True) + 1e-9
    esco /= np.linalg.norm(esco, axis=1, keepdims=True) + 1e-9
    pairs: list[tuple[str, str, float]] = []
    for offset in range(0, len(onet_ids), block_size):
        scores = onet[offset : offset + block_size] @ esco.T
        rows, columns = np.where(scores > threshold)
        for row, column in zip(rows.tolist(), columns.tolist()):
            pairs.append((onet_ids[offset + row], esco_ids[column], float(scores[row, column])))
    pairs.sort(key=lambda item: (item[0], item[1]))
    return pairs


class QdrantCollectionAdapter:
    """Expose Qdrant through the query/get surface used by current retrieval."""

    def __init__(self, client, collection_name: str, graph_id: str) -> None:
        self.client = client
        self.collection_name = collection_name
        self.graph_id = graph_id

    def query(self, *, query_embeddings, n_results: int, include=None) -> dict[str, list]:
        from qdrant_client import models

        all_ids, all_metadata, all_documents, all_distances = [], [], [], []
        graph_filter = models.Filter(
            must=[models.FieldCondition(key="graph_id", match=models.MatchValue(value=self.graph_id))]
        )
        try:
            for raw_vector in query_embeddings:
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=list(raw_vector),
                    query_filter=graph_filter,
                    limit=n_results,
                    with_payload=True,
                    with_vectors=False,
                )
                points = response.points
                payloads = [dict(point.payload or {}) for point in points]
                all_ids.append([str(payload.get("record_id", point.id)) for point, payload in zip(points, payloads)])
                all_metadata.append(payloads)
                all_documents.append([str(payload.get("text", "")) for payload in payloads])
                all_distances.append([1.0 - float(point.score) for point in points])
        except Exception as exc:
            raise RuntimeError("Qdrant vector query failed") from exc
        return {
            "ids": all_ids,
            "metadatas": all_metadata,
            "documents": all_documents,
            "distances": all_distances,
        }

    def _scroll_all(self, with_vectors: bool) -> list:
        records = []
        offset = None
        while True:
            batch, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=None,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=with_vectors,
            )
            records.extend(batch)
            if offset is None:
                return records

    def get(self, ids=None, include=None) -> dict[str, list]:
        with_vectors = "embeddings" in (include or ())
        try:
            if ids is None:
                records = self._scroll_all(with_vectors)
                records.sort(key=lambda row: str((row.payload or {}).get("record_id", row.id)))
            else:
                requested = [str(item) for item in ids]
                point_ids = [qdrant_point_id(self.graph_id, record_id) for record_id in requested]
                found = self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=point_ids,
                    with_payload=True,
                    with_vectors=with_vectors,
                )
                by_record = {
                    str((record.payload or {}).get("record_id", record.id)): record for record in found
                }
                records = [by_record[item] for item in requested if item in by_record]
        except Exception as exc:
            raise RuntimeError("Qdrant vector retrieval failed") from exc

        payloads = [dict(record.payload or {}) for record in records]
        result = {
            "ids": [str(payload.get("record_id", record.id)) for record, payload in zip(records, payloads)],
            "metadatas": payloads,
            "documents": [str(payload.get("text", "")) for payload in payloads],
        }
        if with_vectors:
            result["embeddings"] = [record.vector for record in records]
        return result

    def count(self) -> int:
        try:
            return int(
                self.client.count(
                    collection_name=self.collection_name,
                    exact=True,
                ).count
            )
        except Exception as exc:
            raise RuntimeError("Qdrant collection count failed") from exc


class QdrantVectorStore:
    def __init__(self, client, *, collection_name: str, graph_id: str, batch_size: int) -> None:
        self.client = client
        self.collection_name = collection_name
        self.graph_id = graph_id
        self.batch_size = batch_size

    @classmethod
    def from_settings(cls, settings) -> "QdrantVectorStore":
        return cls(
            create_qdrant_client(settings),
            collection_name=settings.qdrant_collection,
            graph_id=str(settings.kg_id),
            batch_size=settings.qdrant_batch_size,
        )

    def rebuild(self, points: Sequence[PreparedVector]) -> dict[str, int]:
        from qdrant_client import models

        if not points:
            raise RuntimeError("Refusing to create an empty Qdrant collection")
        dimensions = {int(point.vector.size) for point in points}
        if len(dimensions) != 1:
            raise RuntimeError("Prepared vectors have inconsistent dimensions")
        dimension = dimensions.pop()
        try:
            if self.client.collection_exists(self.collection_name):
                self.client.delete_collection(self.collection_name)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
            )
            self.ensure_payload_indexes()
            for offset in range(0, len(points), self.batch_size):
                batch = points[offset : offset + self.batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    wait=True,
                    points=[
                        models.PointStruct(
                            id=point.point_id,
                            vector=point.vector.tolist(),
                            payload=point.payload,
                        )
                        for point in batch
                    ],
                )
                print(f"[qdrant_store] Uploaded {min(offset + self.batch_size, len(points))}/{len(points)} vectors")
        except Exception as exc:
            raise RuntimeError("Qdrant collection rebuild failed") from exc
        return {"vectors": len(points), "dimension": dimension}

    def ensure_payload_indexes(self) -> None:
        """Create keyword indexes used by tenant and metadata filters."""
        from qdrant_client import models

        try:
            for field_name in ("graph_id", "record_id", "source", "entity_type"):
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                    wait=True,
                )
        except Exception as exc:
            raise RuntimeError("Qdrant payload-index creation failed") from exc

    def adapter(self) -> QdrantCollectionAdapter:
        return QdrantCollectionAdapter(self.client, self.collection_name, self.graph_id)

    def validate(self, points: Sequence[PreparedVector]) -> dict[str, Any]:
        adapter = self.adapter()
        count = adapter.count()
        if count != len(points):
            raise RuntimeError(f"Qdrant count mismatch: expected {len(points)}, received {count}")
        first = points[0]
        fetched = adapter.get(ids=[first.record_id], include=["embeddings"])
        if fetched.get("ids") != [first.record_id] or not fetched.get("embeddings"):
            raise RuntimeError("Qdrant ID retrieval validation failed")
        result = adapter.query(query_embeddings=[first.vector.tolist()], n_results=5)
        if not result["ids"][0] or first.record_id not in result["ids"][0]:
            raise RuntimeError("Qdrant similarity validation failed")
        return {
            "vectors": count,
            "dimension": int(first.vector.size),
            "sample_query_ids": result["ids"][0],
        }


def load_qdrant_collection(settings) -> QdrantCollectionAdapter:
    store = QdrantVectorStore.from_settings(settings)
    adapter = store.adapter()
    if adapter.count() <= 0:
        raise RuntimeError("Qdrant collection is empty; run python rebuild_databases.py")
    return adapter
