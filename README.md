# Agentic GraphRAG Career Intelligence Platform

An evidence-led career-advising system for the thesis project. It combines an integrated O*NET/ESCO knowledge graph, train-only empirical career transitions, Qdrant semantic retrieval, LangGraph agent orchestration, and graph-grounded LLM generation.

The deployed application is served from [Vercel](https://knowledge-graph-recommendation.vercel.app/). Its authoritative data stores are Neo4j AuraDB (knowledge graph) and Qdrant Cloud (role vectors); raw source data and generated local stores are not deployed.

## Capabilities

- Recommends career roles from semantic, structural, and empirical-transition evidence.
- Extracts a current role, owned skills, and goal from conversational context.
- Scores qualification, skill gaps, transition effort, transferable skills, and estimated upskilling time.
- Produces phased learning plans and real Coursera search results.
- Checks generated role/skill references against graph entities and returns typed evidence paths.
- Exposes LightGBM-predicted links as explicitly virtual evidence; it never writes them back as observed graph relationships.

## Current validated data layer

| Resource | Validated state |
| --- | --- |
| Neo4j AuraDB | 19,241 nodes and 240,906 relationships |
| Qdrant Cloud | 3,932 role vectors, 3,072 dimensions, cosine distance |
| Empirical transitions | 18,907 support-filtered, train-only `TRANSITIONS_TO` edges |
| Runtime | AuraDB, Qdrant, transition smoothing, and link prediction loaded successfully on 2026-09-12 |

AuraDB may be paused by its provider. If `/api/status` reports `graph_loaded: false`, resume the instance, wait for it to become available, then retry the endpoint.

## System architecture

```text
Browser (React 19 / Vite)
  | GET /api/status, POST /api/chat, GET /api/graph-data
  v
Flask application (app.py -> career_kg_web.py)
  | lazy, lock-protected resource loading
  +-------------------------+-------------------------+
  |                         |                         |
  v                         v                         v
Neo4j AuraDB          Qdrant Cloud              Portable LP model
typed MultiDiGraph    role-vector retrieval     JSON + NumPy inference
NetworkX snapshot     Azure embedding queries   virtual edges only
  |                         |                         |
  +-------------------------+-------------------------+
                            |
                            v
             LangGraph workflow or compatibility pipeline
                            |
                            v
            Structured career recommendation response
```

`career_kg_web.py` reads the graph from AuraDB, exposes Qdrant through the retrieval adapter, and caches those read-only resources per process. The committed React production bundle lives in `public_react/` and is served by Flask/Vercel. HTML entry points have `Cache-Control: no-store` to prevent cached HTML from referencing obsolete hashed assets; JavaScript and CSS retain normal static-asset caching.

### Data lifecycle

```text
O*NET + ESCO + Karrierewege training split
  -> typed graph build and quality controls
  -> enrichment and train-only transitions
  -> role embeddings and Qdrant collection
  -> held-out validation/test evaluation
  -> Neo4j AuraDB upload and round-trip validation
  -> Vercel runtime: Aura snapshot + Qdrant retrieval + agents
```

The rebuild validates allowed node/edge types, normalized labels, finite weights, no self/duplicate/held-out transition evidence, cloud counts, representative Cypher queries, deterministic vector retrieval, and similarity search. `Data/` is ignored but retained locally because it is required for a reproducible full rebuild. Local `graph/` and `index/` runtime stores are not authoritative and are excluded from Vercel.

## AI agent workflow

When `USE_LANGGRAPH=true` (the default in `config.py`), `/api/chat` runs `agents/graph.py`:

```text
Full context
intent -> retrieval -> ranking -> qualification -> traversal -> effort
       -> skill gap -> courses -> learning plan -> generation
       -> faithfulness -> explanation -> response

Partial context
intent -> retrieval -> ranking -> qualification -> traversal -> explore -> response
```

The partial route deliberately stops before effort, learning-plan generation, LLM generation, faithfulness, and explanations, so it does not present a complete path without enough user context. `USE_LANGGRAPH=false` selects the maintained compatibility pipeline in `src/inference_pipeline.py`.

| Agent / stage | Responsibility |
| --- | --- |
| Intent | Extracts user type, current role, owned skills, goal, and context sufficiency. |
| Retrieval | Combines Qdrant vectors, observed/smoothed transitions, skill overlap, and LP coverage backfill. |
| Ranking | Applies Cohere reranking and role-aware evidence. |
| Qualification | Calculates the IDF-weighted share of essential skills already owned. |
| Traversal | Builds bounded role, requirement, and transition triples. |
| Effort | Calculates Transition Effort Score (TES) and upskill-time range. |
| Skill gap | Identifies priority skills, quick wins, and blockers. |
| Courses / learning plan | Retrieves course results and groups them into phased plans. |
| Generation | Produces concise guidance grounded in retrieved graph labels. |
| Faithfulness | Checks extracted entities for graph match and bounded reachability. |
| Explanation | Returns typed provenance paths for recommendations. |

## Retrieval, ranking, and provenance

The hybrid retriever uses distinct sources rather than a single score:

1. Qdrant returns semantically related roles for the embedded query.
2. Direct `TRANSITIONS_TO` evidence and embedding-smoothed distributions add plausible destinations for a resolved current role.
3. IDF-weighted role-requirement overlap adds graph-structural candidates.
4. Reciprocal-rank fusion combines the primary sources.
5. LightGBM is appended only as missing-edge coverage backfill. Its `predicted_transition` evidence remains request-local and is never persisted to AuraDB.
6. Cohere reranking, qualification, graph traversal, TES, and skill-gap analysis form the response payload.

`src/transition_policy.py` is the shared leakage guard. Only Karrierewege `TRANSITIONS_TO` edges with `split=train` (or legacy edges with no split marker) may influence smoothing, traversal, TES, explanations, LP indexing/training, or evaluation. Explicit validation/test edges are excluded.

### Transition Effort Score

Higher TES means a more demanding transition:

```text
TES = 0.35 * skill-gap magnitude
    + 0.15 * ISCO domain distance
    + 0.25 * (1 - empirical support)
    + 0.25 * (1 - skill transferability)
```

TES returns low, moderate, or high bands and an estimated week range. Role-relevant requirements are used so optional generic skills do not distort technical-role effort scores.

### Graph-provenance faithfulness

After generation, `src/faithfulness.py` extracts cited role/skill candidates, matches normalized labels to graph entities, and runs bounded reachability from the retrieved anchor roles. The response includes `score`, matched entities, unmatched entities, unreachable entities, and typed explanation chains. It is a post-generation grounding signal, not complete factual verification of prose.

## Evaluation evidence

The transition ranker was selected on validation data and evaluated once on the held-out test split. Validation and test transitions are not runtime evidence.

| Held-out test method | Hits@1 | Hits@3 | Hits@5 | Hits@10 | MRR | Source coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct train-edge baseline | 0.1478 | 0.2850 | 0.3702 | 0.5014 | 0.2591 | 0.7389 |
| Accepted embedding-smoothed ranker | **0.1483** | **0.2857** | **0.3723** | **0.5059** | **0.2618** | **1.0000** |

The accepted production ranker is the semantic smoother: its top-K gains are modest, but source-role coverage rises from 73.9% to 100%.

| Link-prediction evidence | Result | Interpretation |
| --- | --- | --- |
| Five-fold classification CV | AUC 0.9344; AP 0.8186 | Useful classification signal. |
| Standalone held-out transition ranking | Hits@5 0.0274; Hits@10 0.0447; MRR 0.0182 | Not promoted as an equal-weight top-K ranker. |
| Successful cloud-training run | AUC 0.883507; AP 0.702902; Hits@5 0.911111; MRR 0.797201 | Source-grouped sampled-negative gates; not comparable with the official held-out test. |

```powershell
# Deterministic offline tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -q

# Held-out and diagnostic evaluation
.\.venv\Scripts\python.exe evaluation/evaluate_karrierewege.py
.\.venv\Scripts\python.exe evaluation/evaluate_retrieval_strategies.py
.\.venv\Scripts\python.exe evaluation/evaluate_transition_ranker.py
.\.venv\Scripts\python.exe evaluation/pipeline_ablation.py
```

## API

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/` | GET | React career-advisor application. |
| `/graph` | GET | Interactive graph viewer. |
| `/api/status` | GET | Lazy-loads dependencies and reports graph/vector/model readiness. |
| `/api/chat` | POST | Accepts `{"messages": [{"role": "user", "content": "..."}]}` and returns structured guidance. |
| `/api/graph-data` | GET | Returns a bounded graph-viewer sample and aggregate graph statistics. |

Chat responses can contain narrative text, path/explore data, evidence, faithfulness, explanations, skill-gap analysis, learning plan, courses, and metadata. The frontend normalizes supported legacy/current response shapes at its API boundary.

## Local setup

Prerequisites: Python 3.13, Node.js/npm, and credentials for the model, AuraDB, and Qdrant services. In PowerShell, use `npm.cmd` when `npm` is blocked.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

Set-Location frontend
npm.cmd install
npm.cmd run build
Set-Location ..

.\.venv\Scripts\python.exe career_kg_web.py
```

Open `http://127.0.0.1:8001`. For hot reload, run `npm.cmd run dev` from `frontend/` while Flask runs on port 8001.

Create an uncommitted `.env` file; never commit or print credentials.

| Required group | Variables |
| --- | --- |
| Chat | `CHAT_MODEL_API_KEY`, `CHAT_MODEL_ENDPOINT`, `CHAT_MODEL` |
| Embeddings | `EMBED_MODEL_API_KEY`, `EMBED_MODEL_ENDPOINT`, `EMBED_MODEL` |
| Reranking | `COHERE_RERANK_API_KEY`, `COHERE_RERANK_ENDPOINT`, `COHERE_RERANK_MODEL` |
| Neo4j AuraDB | `KG_URI`, `KG_USER`, `KG_PASS`, `KG_ID`, optional `NEO4J_DATABASE` |
| Qdrant Cloud | `VECTOR_ENDPOINT`, `VECTOR_PASS`, optional `QDRANT_COLLECTION` (default: `career_roles`) |

Feature flags include `USE_LANGGRAPH`, `TRANSITION_SMOOTHING_ENABLED`, and `LINK_PREDICTION_ENABLED`. See `config.py` for all defaults, limits, and TES weights.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
.\.venv\Scripts\python.exe -m compileall -q src agents evaluation career_kg_web.py app.py
.\.venv\Scripts\python.exe -m pip check

Set-Location frontend
npm.cmd run build
npm.cmd run lint
Set-Location ..

git diff --check
curl.exe --silent --show-error https://knowledge-graph-recommendation.vercel.app/api/status
```

The known `react(set-state-in-effect)` warning in `PipelineIndicator.tsx` is pre-existing and non-blocking when lint otherwise succeeds.

## Cloud rebuild

`rebuild_databases.py` is an administration command, not an application-startup step. A full run clears and recreates the dedicated AuraDB graph and Qdrant collection only after preparation, held-out evaluation, and cloud connectivity validation pass.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-research.txt

# Safe: prepares and evaluates without cloud mutation.
.\.venv\Scripts\python.exe rebuild_databases.py --dry-run

# Destructive: intentionally replaces both cloud stores.
.\.venv\Scripts\python.exe rebuild_databases.py
```

Redacted preparation and rebuild reports are written under `artifacts/cloud_rebuild/`.

## GitHub Actions link-prediction training

`.github/workflows/train-link-prediction.yml` is manual-only. GitHub-hosted Ubuntu loads the production AuraDB graph and Qdrant vectors, runs source-role-disjoint validation, exports a portable JSON model, uploads a seven-day artifact, and can create a focused update pull request.

Required repository secrets: `KG_URI`, `KG_USER`, `KG_PASS`, `KG_ID`, `VECTOR_ENDPOINT`, and `VECTOR_PASS`. `NEO4J_DATABASE` is optional, and `QDRANT_COLLECTION` is an optional repository variable. The workflow must exist on the default branch for GitHub to offer **Run workflow**; use `target_branch=dev` for the update PR. It never writes predicted relationships to AuraDB.

## Vercel deployment

`app.py` is the Vercel entry point. `vercel.json` packages Flask templates, `public_react/`, and the portable model; it excludes raw data, tests, evaluation output, local graph/vector stores, and training-only dependencies. Configure the same model and cloud variables in Vercel for the intended Preview/Production targets. The function has a 300-second maximum duration for cold starts and model calls. Do not run database rebuilds during a Vercel build or request.

## Repository map

```text
app.py                         Vercel entry point
career_kg_web.py               Flask routes and lazy cloud-resource loading
config.py                      Environment-backed settings
rebuild_databases.py           Controlled cloud rebuild command
train_link_prediction_cloud.py Fully online LightGBM trainer
agents/                        LangGraph state, nodes, and tools
src/                           Graph build, cloud stores, retrieval, scoring, provenance
evaluation/                    Held-out metrics, diagnostics, and ablations
tests/                         Deterministic regression suite
frontend/                      React/Vite source
public_react/                  Committed production bundle
.github/workflows/             Manual cloud-training workflow
HANDOFF.md                     Operational handoff and delivery record
ALL_STEPS.md                   Chronological implementation/experiment record
novelty.md                     Thesis-contribution record
```

## Research boundaries

- Empirical transitions describe population-level observations, not guaranteed or causal outcomes.
- Held-out validation/test evidence is excluded from runtime and training-transition evidence.
- LightGBM is coverage backfill, not a claimed top-K ranking improvement.
- Faithfulness checks entity match/reachability; it is not complete factual verification.
- Coursera and hosted cloud services are external dependencies; `/api/status` is the source of truth for current runtime readiness.
