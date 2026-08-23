# Agentic GraphRAG Career Intelligence Platform

An agentic career recommendation system combining a multi-taxonomy knowledge graph (ONET + ESCO), empirical career transitions (Karrierewege), LangGraph agent orchestration, link prediction, and LLM-grounded generation to provide evidence-based career guidance.

## Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│                LangGraph Supervisor                        │
│                                                            │
│  [Intent Router] → [Retrieval + Transitions]              │
│        │                    │                              │
│        ▼                    ▼                              │
│  [Cohere Reranker] → [Skill-Gap Ranking]                 │
│        │                    │                              │
│        ▼                    ▼                              │
│  [Graph Traversal] → [Effort Scoring]                    │
│        │                    │                              │
│        ▼                    ▼                              │
│  [LLM Generation] → [Faithfulness Check]                 │
│        │                    │                              │
│        ▼                    ▼                              │
│  [Explanation Chains] → [Course Search]                  │
└──────────────────────────────────────────────────────────┘
    │
    ▼
Structured Career Recommendation
(roles, effort bands, skill gaps, evidence chains, courses)
```

## Key Features

- **Knowledge Graph**: 19,225 nodes, 224,711 edges spanning ONET + ESCO taxonomies with 18,907 empirical career transition edges
- **Transition Effort Score**: Multi-factor difficulty metric (IDF-weighted skill gap, ISCO domain distance, empirical support, transferability)
- **Link Prediction**: LightGBM with 9 graph-structural features (CV AUC 0.9344) achieving 100% source-role coverage
- **Faithfulness Verification**: Post-generation check that LLM-cited entities exist in the graph and are reachable from anchor nodes
- **Explanation Chains**: Typed-edge evidence paths (TRANSITIONS_TO, SIMILAR_TO, REQUIRES) for each recommendation
- **Hybrid Retrieval**: Vector search + empirical transitions + link prediction with priority-based fallback
- **Embedding-Smoothed Transitions**: Semantic-neighbour propagation for cold-start roles (+8.2% coverage)

## Inference Pipeline

| Step | Component | Method |
|------|-----------|--------|
| 1 | Intent Routing | LLM classifies user type, extracts skills/role/goal |
| 2 | Retrieval | ChromaDB cosine (top-50) + transition augmentation |
| 3 | Reranking | Cohere rerank-v4.0-pro → top-8 candidates |
| 4 | Skill-Gap Ranking | Deterministic accessibility ordering (owned ∩ required / required) |
| 5 | Graph Traversal | REQUIRES, SIMILAR_TO, TRANSITIONS_TO triples |
| 6 | Effort Scoring | TES = 0.35×SkillGap + 0.15×Domain + 0.25×(1-Empirical) + 0.25×(1-Transfer) |
| 7 | Generation | LLM grounded in graph triples (2-3 sentences) |
| 8 | Faithfulness | Entity extraction → graph match → BFS reachability |
| 9 | Explanations | Typed-edge path tracing per recommended role |
| 10 | Courses | Coursera search for skill gaps |

## Evaluation Results

| Method | Hits@5 | Hits@10 | MRR | Coverage |
|--------|--------|---------|-----|----------|
| Direct edges (baseline) | 0.3702 | 0.5014 | 0.2591 | 73.9% |
| + Embedding smoothing | 0.3722 | 0.5058 | 0.2617 | 95.3% |
| + Combined (direct > LP) | 0.3704 | 0.5017 | 0.2580 | **100%** |

| Model | Metric | Value |
|-------|--------|-------|
| Link Prediction | 5-fold CV AUC | 0.9344 |
| Link Prediction | 5-fold CV AP | 0.8186 |
| Faithfulness | Live test score | 1.0 (7/7 reachable) |

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Knowledge Graph | NetworkX MultiDiGraph |
| Vector Store | ChromaDB |
| Embeddings | Azure OpenAI text-embedding-3-large |
| Reranker | Cohere rerank-v4.0-pro |
| LLM | Azure OpenAI gpt-5.4-nano (Responses API) |
| Link Prediction | LightGBM |
| Agent Orchestration | LangGraph StateGraph |
| Backend | Flask (Python 3.13) |
| Frontend | Vanilla JS + Cytoscape.js |

## Run Locally

1. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and add API keys.

3. Start the server:

   ```bash
   python career_kg_web.py
   ```

4. Open http://127.0.0.1:8001

Local development reads the graph from `graph/graph.gpickle` and the Chroma index from `index/chroma/`. These are excluded from Git and must be hosted externally for deployment.

## Deploy to Vercel

Connect this repository and deploy the `dev` branch. Vercel detects `app.py` as the Flask entry point.

**Required environment variables:**
- `CHAT_MODEL_API_KEY`
- `EMBED_MODEL_API_KEY`
- `COHERE_RERANK_API_KEY`

**Optional (see `.env.example` for full list):**
- `USE_LANGGRAPH=true` — enable agent orchestration
- `LINK_PREDICTION_ENABLED=true` — enable LP fallback for uncovered roles
- `EFFORT_WEIGHT_*` — tune effort score component weights

The deployment does not ship graph or vector data. Connect `load_graph()` and `load_chroma_collection()` to external hosted databases.

## Project Structure

```
app.py                      Vercel entry point
career_kg_web.py            Flask server (port 8001)
config.py                   Environment-based settings
coursera_client.py          Course search integration

src/
  inference_pipeline.py     10-step inference orchestration
  skill_gap.py              Skill resolution + accessibility ranking
  transition_embedding.py   Semantic-neighbour smoothing
  transition_effort.py      Transition Effort Score (TES)
  link_prediction.py        LightGBM link predictor (9 features)
  kg_enrichment.py          Skill IDF + ISCO codes
  faithfulness.py           Graph-provenance verification
  explainability.py         Typed-edge explanation chains
  hybrid_retrieval.py       Multi-source retrieval fusion
  embeddings_index.py       ChromaDB embed + alignment
  graph_store.py            Graph loading utilities
  text_normalization.py     Label normalization

agents/
  state.py                  CareerAgentState TypedDict
  graph.py                  LangGraph StateGraph (10 nodes)
  tools.py                  6 graph query tools
  nodes/                    One file per pipeline node

templates/                  HTML (chat + graph viewer)
public/                     CSS + JS (chat + graph)
```

## Novel Research Contributions

1. **Empirical Career Transition Edges** — 18,907 frequency-weighted edges from 568,888 real career trajectories
2. **Skill-Gap-Aware Career Path Ranking** — deterministic accessible-to-aspirational ordering
3. **Embedding-Smoothed Transition Inference** — cold-start coverage via semantic neighbours
4. **Transition Effort Score** — multi-factor career switch difficulty metric
5. **KG Link Prediction** — structural feature-based edge prediction (AUC 0.93)
6. **Graph-Provenance Faithfulness** — post-generation verification against graph topology
7. **Provenance-Traced Explanation Chains** — typed-edge evidence paths per recommendation
