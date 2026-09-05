# Agentic GraphRAG Career Intelligence Platform

An agentic career recommendation system combining a multi-taxonomy knowledge graph (ONET + ESCO), empirical career transitions (Karrierewege), LangGraph agent orchestration, link prediction, and LLM-grounded generation to provide evidence-based career guidance.

## Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│                   LangGraph Agent Pipeline (12 nodes)             │
│                                                                    │
│  Intent → Retrieval → Ranking → Qualification → Traversal         │
│                                                                    │
│  → Effort Scoring → Skill Gap Analysis → Course Search            │
│                                                                    │
│  → Learning Plan → Generation → Faithfulness → Explanation        │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
Structured Career Recommendation
(roles sorted by effort, stats tiles, skill gaps, learning roadmap,
 graph-verified evidence chains, clickable Coursera courses)
```

## Key Features

- **Knowledge Graph**: 19,241 nodes and 240,906 relationships in Neo4j AuraDB, spanning ONET + ESCO with 18,907 train-only career transition edges
- **Transition Effort Score (TES)**: Multi-factor difficulty metric (IDF-weighted skill gap, ISCO domain distance, empirical support, transferability) — displayed as percentage with upskill time estimate
- **Qualification Scoring**: IDF-weighted fraction of essential skills the user already has per target role
- **Skill Gap Analysis**: Per-role ranked missing skills by TES reduction impact, quick wins, and blockers
- **Learning Roadmap**: Phased upskill plan with Coursera course groupings and week estimates
- **Link Prediction**: LightGBM over structural, semantic, transition, ISCO, and skill features, exposed only as scored virtual missing-edge backfill
- **Faithfulness Verification**: Post-generation check showing verified/unreachable/unmatched entities with percentage score
- **Explanation Chains**: Typed-edge evidence paths (TRANSITIONS\_TO, SIMILAR\_TO, REQUIRES) per recommendation
- **Hybrid Retrieval**: Vector search + accepted transition smoothing + role-relevant graph overlap, with LP coverage backfill
- **Embedding-Smoothed Transitions**: Semantic-neighbour propagation for cold-start roles (+26.1pp source-role coverage)
- **ISCO Group Edges**: Structural domain-proximity edges between roles sharing a 2-digit ISCO code

## Inference Pipeline

| Step | Component | Method |
|------|-----------|--------|
| 1 | Intent Routing | LLM classifies user type, extracts skills/role/goal |
| 2 | Retrieval | RRF over Qdrant, direct/smoothed transitions, IDF skill overlap; LP missing-edge backfill |
| 3 | Reranking | Cohere rerank-v4.0-pro → top-8 candidates |
| 4 | Qualification | IDF-weighted % of essential skills owned per candidate |
| 5 | Graph Traversal | REQUIRES, SIMILAR\_TO, observed and explicitly inferred transition triples |
| 6 | Effort Scoring | TES = 0.35×SkillGap + 0.15×Domain + 0.25×(1-Empirical) + 0.25×(1-Transfer) |
| 7 | Skill Gap Analysis | Ranked missing skills by TES reduction impact |
| 8 | Course Search | Coursera search for skill gaps |
| 9 | Learning Plan | Phased roadmap from skill gap + courses |
| 10 | Generation | LLM grounded in graph triples (2–3 sentences) |
| 11 | Faithfulness | Entity extraction → graph match → BFS reachability |
| 12 | Explanations | Typed-edge path tracing per recommended role |

## Evaluation Results

| Method | Hits@1 | Hits@3 | Hits@5 | Hits@10 | MRR | Source coverage |
|--------|--------|--------|--------|---------|-----|-----------------|
| Direct edges (cloud rebuild baseline) | 0.1478 | 0.2850 | 0.3702 | 0.5014 | 0.2591 | 73.9% |
| Accepted embedding smoothing | **0.1483** | **0.2857** | **0.3723** | **0.5059** | **0.2618** | **100%** |

| Model | Metric | Value |
|-------|--------|-------|
| Link Prediction | 5-fold CV AUC | 0.9344 |
| Link Prediction | 5-fold CV AP | 0.8186 |
| Link Prediction | Standalone test Hits@5 | 0.0274 (coverage-only; not promoted as top-K ranker) |

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Knowledge Graph | Neo4j AuraDB, loaded as a NetworkX MultiDiGraph snapshot at runtime |
| Vector Store | Qdrant Cloud (cosine, 3,932 role vectors) |
| Embeddings | Azure OpenAI text-embedding-3-large |
| Reranker | Cohere rerank-v4.0-pro |
| LLM | Azure OpenAI gpt-5.4-nano (Responses API) |
| Link Prediction | LightGBM cloud training; portable JSON/NumPy inference on Vercel |
| Agent Orchestration | LangGraph StateGraph (12 nodes) |
| Backend | Flask (Python 3.13) |
| Frontend | React 19 + Vite + Tailwind CSS v4 + Framer Motion |

## Run Locally

1. Install Python dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Install frontend dependencies and build:

   ```bash
   cd frontend
   npm install
   npm run build
   cd ..
   ```

3. Copy `.env.example` to `.env` and add the model, Neo4j Aura, and Qdrant Cloud credentials.

4. Start the server:

   ```bash
   python career_kg_web.py
   ```

5. Open http://127.0.0.1:8001

> **Frontend dev mode** (hot reload): `cd frontend && npm run dev` — proxies `/api/*` to Flask on port 8001.

Local and deployed inference read the graph from Neo4j AuraDB and vectors from Qdrant Cloud. The same embedding model is used for indexing and query-time retrieval. The small trained LP model is bundled, while raw datasets and generated graph/vector stores remain excluded from deployment.

## Train link prediction fully online

GitHub Actions supplies the temporary Linux training computer; Vercel continues to host the application. GitHub Pages is not used because it cannot run the Flask API.

The manual workflow at `.github/workflows/train-link-prediction.yml` reconstructs the production graph from AuraDB, reads existing ESCO vectors from Qdrant, trains LightGBM with source-role-disjoint folds, removes held-out-source transitions from neighbour-evidence features, and reports AUC, average precision, Hits@1/3/5/10, MRR, and NDCG. A candidate must pass all configured gates before the workflow can create a model-update pull request.

Configure these GitHub Actions repository secrets:

- `KG_URI`
- `KG_USER`
- `KG_PASS`
- `KG_ID`
- `VECTOR_ENDPOINT`
- `VECTOR_PASS`
- `NEO4J_DATABASE` only when using a non-default database

Optionally set the repository variable `QDRANT_COLLECTION`; it defaults to `career_roles`. The workflow does not require chat, embedding, or reranking API keys because it reuses vectors already stored in Qdrant.

Because the repository default branch is `main`, the workflow must exist on `main` for GitHub to display its manual **Run workflow** button. Run it with `target_branch=dev` and pull-request publishing enabled; after review, merge the generated model PR so Vercel builds the updated portable model from `dev`.

The repository is configured with all six required cloud secrets, `QDRANT_COLLECTION=career_roles`, read/write workflow permissions, and permission for Actions to create pull requests. `NEO4J_DATABASE` remains optional. The workflow requests only `contents: write` and `pull-requests: write`; it runs only when manually started.

The workflow uploads the candidate model and redacted metrics as a 7-day Actions artifact. Its job timeout is 30 minutes, it never writes predicted relationships to AuraDB, and it does not expose endpoint or credential values in the report.

## Rebuild the cloud databases

The databases are prepared locally and mutated only after preprocessing, vector generation, and held-out evaluation succeed:

```bash
python -m pip install -r requirements-research.txt
python rebuild_databases.py
```

Use `python rebuild_databases.py --dry-run` to perform every preparation and evaluation step without changing either cloud database. The full command recreates the dedicated Aura graph and Qdrant collection, creates indexes, uploads in batches, validates counts and representative queries, and writes redacted reports under `artifacts/cloud_rebuild/`.

## Deploy to Vercel

Connect this repository and deploy the `dev` branch. Vercel detects `app.py` as the Flask entry point.

**Required environment variables:**
- `CHAT_MODEL_API_KEY`
- `EMBED_MODEL_API_KEY`
- `COHERE_RERANK_API_KEY`
- `KG_URI`
- `KG_USER`
- `KG_PASS`
- `KG_ID`
- `VECTOR_ENDPOINT`
- `VECTOR_PASS`

**Optional (see `.env.example` for full list):**
- `USE_LANGGRAPH=true` — enable agent orchestration (default: `true`)
- `LINK_PREDICTION_ENABLED=true` — enabled by default
- `EFFORT_WEIGHT_*` — tune effort score component weights

Set the variables for Preview and Production in Vercel before deploying. Do not paste their values into repository files. `vercel.json` packages the React build and LP model, excludes raw/local graph data, and allows the Python function enough time for its database-backed cold start.

## Project Structure

```
app.py                      Vercel entry point
career_kg_web.py            Flask server (port 8001)
train_link_prediction_cloud.py  Aura/Qdrant cloud training entry point
config.py                   Environment-based settings
coursera_client.py          Course search integration

src/
  inference_pipeline.py     12-step inference orchestration
  skill_gap.py              Skill resolution + accessibility ranking
  transition_embedding.py   Semantic-neighbour smoothing
  transition_effort.py      Transition Effort Score (TES) + upskill time estimate
  link_prediction.py        LightGBM predictor with model-declared feature compatibility
  portable_lightgbm.py      Dependency-free JSON model inference for Vercel
  kg_enrichment.py          Skill IDF + ISCO codes
  isco_edges.py             SAME_ISCO_GROUP structural edges
  faithfulness.py           Graph-provenance verification
  explainability.py         Typed-edge explanation chains
  hybrid_retrieval.py       Multi-source retrieval fusion
  embeddings_index.py       Shared Azure embedding client + legacy offline utilities
  neo4j_store.py            Aura schema, upload, validation, and runtime graph loading
  qdrant_store.py           Qdrant indexing, payload metadata, and runtime vector adapter
  graph_quality.py          Entity normalization, pruning, deduplication, connectivity enrichment
  text_normalization.py     Label normalization
  transition_policy.py      Training-only transition contract

agents/
  state.py                  CareerAgentState TypedDict
  graph.py                  LangGraph StateGraph (12 nodes)
  tools.py                  6 graph query tools
  nodes/
    intent.py               Intent routing
    retrieval.py            Vector + transition retrieval
    ranking.py              Cohere rerank + gap ranking
    qualification_node.py   IDF-weighted skill qualification score
    traversal.py            Graph traversal
    effort.py               TES computation + effort sorting
    skill_gap_node.py       Ranked skill gap per role
    courses.py              Coursera search
    learning_plan_node.py   Phased learning roadmap
    generation.py           LLM generation
    faithfulness_node.py    Post-generation verification
    explanation.py          Evidence chain tracing

frontend/                   React 19 + Vite + Tailwind v4
  src/
    components/
      ChatShell.tsx         Main chat layout + message rendering
      RoleCard.tsx          Role card with effort, qualification, evidence
      StatsTiles.tsx        4-tile stats row per top role
      TransferableSkills.tsx Cross-role shared skill strengths
      SkillGapCard.tsx      Priority skills / quick wins / blockers
      LearningRoadmap.tsx   Phased learning timeline
      EvidencePanel.tsx     Graph-verified entity breakdown + provenance
      FaithfulnessBadge.tsx Verified % badge
      CourseCard.tsx        Clickable Coursera course card
      PipelineIndicator.tsx 5-stage animated pipeline loader
      StatusPanel.tsx       Server status + graph stats
    api.ts                  Typed fetch wrappers
    types.ts                Shared TypeScript interfaces
```

## Novel Research Contributions

1. **Empirical Career Transition Edges** — 18,907 frequency-weighted edges from 568,888 real career trajectories
2. **Skill-Gap-Aware Career Path Ranking** — deterministic accessible-to-aspirational ordering
3. **Embedding-Smoothed Transition Inference** — cold-start coverage via semantic neighbours
4. **Transition Effort Score** — multi-factor career switch difficulty metric with upskill time estimate
5. **KG Link Prediction** — structural/semantic missing-edge prediction with explicit virtual provenance
6. **Graph-Provenance Faithfulness** — post-generation verification against graph topology
7. **Provenance-Traced Explanation Chains** — typed-edge evidence paths per recommendation
