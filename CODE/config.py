"""
Central configuration.  All settings are read from environment variables
or a .env file in the project root.  No secrets are hardcoded here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(root: Path) -> None:
    """Minimal .env loader — sets vars that are not already in os.environ."""
    env_path = root / ".env"
    if not env_path.exists():
        return
    with env_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


@dataclass
class Settings:
    # --- Chat model (OpenAI Responses API via APIM proxy) ---
    chat_model_api_key: str | None
    chat_model_endpoint: str          # full URL ending in /openai/v1/responses
    chat_model: str                   # model name sent in the request body

    # --- Embedding model (same APIM gateway) ---
    embed_model_api_key: str | None
    embed_model_endpoint: str         # base URL; code appends /openai/v1/embeddings
    embed_model: str

    # --- Cohere reranker (via Azure APIM) ---
    cohere_rerank_api_key: str | None
    cohere_rerank_endpoint: str
    cohere_rerank_model: str

    # --- File paths ---
    onet_dir: Path
    esco_dir: Path
    graph_path: Path
    chroma_dir: Path

    # --- Tuning ---
    embed_batch_size: int
    similarity_threshold: float
    retrieval_top_k: int
    rerank_top_n: int
    onet_importance_threshold: float

    @classmethod
    def from_env(cls, root: Path | None = None) -> "Settings":
        if root is None:
            root = Path(__file__).parent
        _load_dotenv(root)

        def _path(key: str, default: str) -> Path:
            return root / os.getenv(key, default)

        return cls(
            # Chat
            chat_model_api_key=os.getenv("CHAT_MODEL_API_KEY"),
            chat_model_endpoint=os.getenv(
                "CHAT_MODEL_ENDPOINT",
                "https://career.azure-api.net/career-graph-ai/openai/v1/responses",
            ),
            chat_model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
            # Embeddings
            embed_model_api_key=os.getenv("EMBED_MODEL_API_KEY"),
            embed_model_endpoint=os.getenv(
                "EMBED_MODEL_ENDPOINT",
                "https://career.azure-api.net/career-graph-ai",
            ),
            embed_model=os.getenv("EMBED_MODEL", "text-embedding-3-large"),
            # Cohere reranker
            cohere_rerank_api_key=os.getenv("COHERE_RERANK_API_KEY"),
            cohere_rerank_endpoint=os.getenv(
                "COHERE_RERANK_ENDPOINT",
                "https://career.azure-api.net/career-graph-ai/providers/cohere/v2/rerank",
            ),
            cohere_rerank_model=os.getenv("COHERE_RERANK_MODEL", "Cohere-rerank-v4.0-pro"),
            # Paths
            onet_dir=_path("ONET_DIR", "Data/ONET"),
            esco_dir=_path(
                "ESCO_DIR",
                "Data/ESCO dataset - v1.2.1 - classification - en - csv",
            ),
            graph_path=_path("GRAPH_PATH", "graph/graph.gpickle"),
            chroma_dir=_path("CHROMA_DIR", "index/chroma"),
            # Tuning
            embed_batch_size=int(os.getenv("EMBED_BATCH_SIZE", "100")),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.85")),
            retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "50")),
            rerank_top_n=int(os.getenv("RERANK_TOP_N", "5")),
            onet_importance_threshold=float(
                os.getenv("ONET_IMPORTANCE_THRESHOLD", "3.0")
            ),
        )
