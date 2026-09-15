"""Read-only graph/vector loading for offline research commands."""

from __future__ import annotations

from config import Settings
from src.graph_build import load_graph
from src.neo4j_store import load_graph_from_neo4j
from src.qdrant_store import load_qdrant_collection
from src.transition_embedding import load_live_esco_embeddings


def load_research_graph(settings: Settings):
    """Prefer a local snapshot; otherwise read the authoritative Aura graph."""
    if settings.graph_path.is_file():
        return load_graph(settings.graph_path), "local_graph_snapshot"
    return load_graph_from_neo4j(settings), "auradb_read_only"


def load_research_embeddings(settings: Settings, graph):
    """Load live role vectors from Qdrant without modifying the collection."""
    collection = load_qdrant_collection(settings)
    embeddings, report = load_live_esco_embeddings(collection, graph)
    return embeddings, {**report, "source": "qdrant_read_only"}
