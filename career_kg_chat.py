"""
Legacy monolith — replaced by the modular src/ package.

This file is intentionally empty.  All functionality has been moved to:
  config.py                    Settings dataclass and env-var loading
  src/onet_preprocessing.py   ONET Excel -> nodes/edges DataFrames
  src/esco_preprocessing.py   ESCO CSV -> nodes/edges DataFrames
  src/graph_build.py          NetworkX graph construction and persistence
  src/embeddings_index.py     Azure embeddings + ChromaDB index
  src/inference_pipeline.py   GraphRAG inference loop (routing -> retrieval
                               -> reranking -> traversal -> generation)
  build_graph.py              Offline graph build CLI
  career_kg_web.py            Flask web server

To rebuild the knowledge graph:
  python build_graph.py           # graph only
  python build_graph.py --embed   # graph + ChromaDB index + alignment edges

To start the web server:
  python career_kg_web.py
"""
