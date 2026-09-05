"""
ChromaDB index management and Azure embedding utilities.

All HTTP calls use urllib.request — no openai SDK dependency.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from config import Settings


_RUNTIME_CHROMA_DIR: Path | None = None
_RUNTIME_CHROMA_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Azure Embedding API
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
            f"[embeddings_index] HTTP {exc.code} from {url}: {body_text}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"[embeddings_index] Connection error calling {url}: {exc.reason}"
        ) from exc


def embed_texts(texts: list[str], settings: "Settings") -> list[list[float]]:
    """Embed a list of texts using the Azure embedding endpoint.

    Batches requests and retries transient timeouts, rate limits, and 5xx errors.
    Returns a list of float vectors in the same order as input texts.
    """
    if not settings.embed_model_api_key:
        raise RuntimeError(
            "[embeddings_index] EMBED_MODEL_API_KEY is not set. "
            "Add it to your .env file."
        )

    url = settings.embed_model_endpoint.rstrip("/") + "/openai/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "api-key": settings.embed_model_api_key,
    }

    all_vectors: list[list[float]] = []
    batch_size = settings.embed_batch_size

    for batch_start in range(0, len(texts), batch_size):
        batch = texts[batch_start : batch_start + batch_size]
        payload = {"model": settings.embed_model, "input": batch}

        # API gateways can occasionally time out during an otherwise healthy
        # full rebuild. Retry only transient errors; invalid requests still fail.
        max_retries = 6
        for attempt in range(max_retries):
            try:
                result = _post_json(url, payload, headers)
                break
            except RuntimeError as exc:
                message = str(exc)
                retryable = (
                    any(f"HTTP {status}" in message for status in (408, 429, 500, 502, 503, 504))
                    or "Connection error" in message
                )
                if retryable and attempt < max_retries - 1:
                    wait = min(2 ** attempt, 16)
                    print(
                        f"[embeddings_index] Transient embedding failure; "
                        f"retrying in {wait}s ({attempt + 1}/{max_retries - 1}) ...",
                        flush=True,
                    )
                    time.sleep(wait)
                else:
                    raise

        # data is a list of {index, embedding} dicts
        sorted_items = sorted(result["data"], key=lambda x: x["index"])
        all_vectors.extend(item["embedding"] for item in sorted_items)

    return all_vectors


# ---------------------------------------------------------------------------
# ChromaDB index
# ---------------------------------------------------------------------------

def _queryable_chroma_dir(source_dir: Path) -> Path:
    """Return a writable Chroma directory for the current runtime.

    Vercel packages application files on a read-only filesystem, while Chroma
    updates its SQLite store when a ``PersistentClient`` opens it. Copy the
    bundled index to the function's writable temporary directory once per
    process. Local development continues to use the configured directory.
    """
    source_dir = Path(source_dir)
    if not os.getenv("VERCEL"):
        return source_dir

    global _RUNTIME_CHROMA_DIR
    with _RUNTIME_CHROMA_LOCK:
        if _RUNTIME_CHROMA_DIR is not None:
            return _RUNTIME_CHROMA_DIR
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Packaged ChromaDB directory not found: {source_dir}")

        runtime_dir = (
            Path(tempfile.gettempdir()) / f"career-kg-chroma-{os.getpid()}"
        )
        shutil.copytree(source_dir, runtime_dir, dirs_exist_ok=True)
        _RUNTIME_CHROMA_DIR = runtime_dir
        return runtime_dir

def _build_role_text(node_id: str, data: dict) -> str:
    parts = [data.get("title", node_id)]
    if data.get("description"):
        parts.append(data["description"])
    if data.get("job_zone_title"):
        parts.append(f"Preparation: {data['job_zone_title']}")
    if data.get("typical_education_level"):
        parts.append(f"Education: {data['typical_education_level']}")
    if data.get("isco_group"):
        parts.append(f"ISCO group: {data['isco_group']}")
    return "\n".join(parts)


def build_role_text(node_id: str, data: dict) -> str:
    """Public role-document builder shared by indexing and candidate expansion."""
    return _build_role_text(node_id, data)


def _safe_metadata(data: dict) -> dict:
    """ChromaDB requires metadata values to be str/int/float/bool — coerce lists."""
    safe = {}
    for k, v in data.items():
        if isinstance(v, (str, int, float, bool)):
            safe[k] = v
        elif isinstance(v, list):
            safe[k] = ", ".join(str(i) for i in v)
        elif v is None:
            pass  # omit None values
        else:
            safe[k] = str(v)
    return safe


def build_chroma_index(G: nx.MultiDiGraph, settings: "Settings") -> None:
    """Embed all role nodes and store them in a persistent ChromaDB collection."""
    try:
        import chromadb
    except ImportError:
        raise RuntimeError(
            "[embeddings_index] chromadb is not installed. "
            "Run: pip install chromadb"
        ) from None

    Path(settings.chroma_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    collection = client.get_or_create_collection(
        "roles_collection",
        metadata={"hnsw:space": "cosine"},
    )

    role_nodes = [
        (nid, data)
        for nid, data in G.nodes(data=True)
        if data.get("type") == "role"
    ]
    print(f"[embeddings_index] Embedding {len(role_nodes)} role nodes …")

    ids = [nid for nid, _ in role_nodes]
    texts = [_build_role_text(nid, data) for nid, data in role_nodes]
    metadatas = [_safe_metadata(data) for _, data in role_nodes]

    batch_size = settings.embed_batch_size
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i : i + batch_size]
        batch_texts = texts[i : i + batch_size]
        batch_meta = metadatas[i : i + batch_size]

        vectors = embed_texts(batch_texts, settings)
        collection.add(
            ids=batch_ids,
            embeddings=vectors,
            metadatas=batch_meta,
            documents=batch_texts,
        )
        print(f"[embeddings_index]   indexed {min(i + batch_size, len(ids))}/{len(ids)}")

    print("[embeddings_index] ChromaDB index built.")


def load_chroma_collection(settings: "Settings"):
    """Load the pre-built ChromaDB collection for querying."""
    try:
        import chromadb
    except ImportError:
        raise RuntimeError(
            "[embeddings_index] chromadb is not installed. "
            "Run: pip install chromadb"
        ) from None

    try:
        chroma_dir = _queryable_chroma_dir(settings.chroma_dir)
        client = chromadb.PersistentClient(path=str(chroma_dir))
        collection = client.get_collection("roles_collection")
        return collection
    except Exception as exc:
        raise RuntimeError(
            f"[embeddings_index] Failed to load ChromaDB collection from "
            f"{settings.chroma_dir}: {exc}. "
            "Ensure the packaged ChromaDB index is present and compatible."
        ) from exc


# ---------------------------------------------------------------------------
# ONET–ESCO semantic alignment
# ---------------------------------------------------------------------------

def compute_alignment_edges(
    G: nx.MultiDiGraph,
    settings: "Settings",
) -> list[tuple[str, str, float]]:
    """Compute cosine similarity between ONET and ESCO role embeddings.

    Returns a list of (onet_id, esco_id, similarity) tuples where
    similarity > settings.similarity_threshold.
    """
    try:
        import numpy as np
    except ImportError:
        raise RuntimeError(
            "[embeddings_index] numpy is not installed. Run: pip install numpy"
        ) from None

    onet_roles = [
        (nid, data)
        for nid, data in G.nodes(data=True)
        if data.get("type") == "role" and data.get("source") == "onet"
    ]
    esco_roles = [
        (nid, data)
        for nid, data in G.nodes(data=True)
        if data.get("type") == "role" and data.get("source") == "esco"
    ]

    if not onet_roles or not esco_roles:
        print("[embeddings_index] No roles to align.")
        return []

    print(
        f"[embeddings_index] Computing alignment: "
        f"{len(onet_roles)} ONET × {len(esco_roles)} ESCO roles …"
    )

    onet_ids = [nid for nid, _ in onet_roles]
    esco_ids = [nid for nid, _ in esco_roles]
    onet_texts = [_build_role_text(nid, d) for nid, d in onet_roles]
    esco_texts = [_build_role_text(nid, d) for nid, d in esco_roles]

    onet_vecs = np.array(embed_texts(onet_texts, settings), dtype=np.float32)
    esco_vecs = np.array(embed_texts(esco_texts, settings), dtype=np.float32)

    # Normalise for cosine similarity
    onet_norm = onet_vecs / (np.linalg.norm(onet_vecs, axis=1, keepdims=True) + 1e-9)
    esco_norm = esco_vecs / (np.linalg.norm(esco_vecs, axis=1, keepdims=True) + 1e-9)

    # Process in blocks to avoid huge matrix
    threshold = settings.similarity_threshold
    pairs: list[tuple[str, str, float]] = []
    block = 500

    for i in range(0, len(onet_ids), block):
        sim_block = onet_norm[i : i + block] @ esco_norm.T  # (block, n_esco)
        rows, cols = np.where(sim_block > threshold)
        for r, c in zip(rows, cols):
            pairs.append((onet_ids[i + r], esco_ids[c], float(sim_block[r, c])))

    print(f"[embeddings_index] Found {len(pairs)} alignment pairs above threshold.")
    return pairs
