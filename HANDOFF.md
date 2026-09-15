# GraphRAG Career Advisor — Session Handoff

## Project State (as of 2026-09-11)

GraphRAG career chatbot for a thesis. **React frontend + LangGraph 12-node pipeline. The production UI and HTML cache fix are deployed and browser-verified; cloud runtime readiness must be rechecked.**

- **Branch:** `dev`
- **Remote:** `https://github.com/marvisrego/KnowledgeGraph_Recommendation.git`
- **Server:** `python career_kg_web.py` → `http://127.0.0.1:8001`
- **Frontend:** React 19 + Vite + Tailwind CSS v4 (built to `public_react/`, served by Flask)
- **Knowledge graph:** Neo4j AuraDB; last successful rebuild validation on 2026-09-05 recorded 19,241 nodes and 240,906 relationships. The 2026-09-11 production status check reports `graph_loaded: false`.
- **Vector database:** Qdrant Cloud; last rebuild validation recorded 3,932 role vectors at 3,072 dimensions. The 2026-09-11 production status check reports `qdrant_loaded: true`; vector counts were not revalidated during the UI work.
- **Runtime:** Aura is loaded into a NetworkX compatibility snapshot; semantic retrieval runs against Qdrant
- **Rebuild:** `python rebuild_databases.py` prepares, evaluates, rebuilds, and validates both dedicated cloud databases
- **Vercel:** production is `https://knowledge-graph-recommendation.vercel.app/`. UI PR #11 merged as `42c51e5`; cache-fix PR #12 merged into `main` as `2e37d17` and its production deployment passed. `app.py` serves the bundled React build and LP model, while raw data and local graph/vector artifacts stay excluded.
- **GitHub Actions:** cloud-training secrets, `QDRANT_COLLECTION=career_roles`, read/write workflow permissions, and pull-request creation are configured at repository level

### 2026-09-11 UI polish and white-screen cache repair

- **UI delivery:** `44cab5b` (`Polish career advisor and graph explorer UI`) was pushed to `dev`, then merged through [PR #11](https://github.com/marvisrego/KnowledgeGraph_Recommendation/pull/11). Production deployment `6400355787` corresponds to merge commit `42c51e5`.
- Applied the approved charcoal/cyan palette, retained Outfit and JetBrains Mono, improved text hierarchy, card spacing, tabs, disclosures, composer focus, and graph-viewer controls. Reflowed existing sidebar content above the mobile chat; the page can scroll while the conversation retains its own viewport and existing scroll-position behavior.
- Added restrained hover feedback and a welcome-only 320px cursor glow with at most 6% opacity. Coordinates use a local ref and `requestAnimationFrame`; pointer listeners/frames are cleaned up, and the glow is disabled for touch/coarse pointers and reduced motion. No new dependency was introduced.
- The UI pass preserved API requests, response normalization, recommendation values, graph interactions, and backend logic. Research references were Linear, Raycast, and Resend; only frontend skills were used.
- **UI verification:** production build passed; 150 Python tests and 74 browser checks passed. Browser checks covered desktop/tablet/mobile and 320px/short-height cases, existing controls, matching baseline requests and recommendation content (allowing CSS text-case changes), long content, optional fields, scroll preservation, hover, focus, and reduced/coarse-pointer motion. Sampled advisor, graph, and mobile axe checks reported no WCAG A/AA violations. Lint retained the existing non-blocking `PipelineIndicator.tsx` state-in-effect warning.
- **White-screen diagnosis:** fresh production loads rendered correctly, but a returning browser could reuse stale HTML. Both the old and new entrypoints were 822 bytes, and Vercel normalized their timestamps to `1540000000`. Flask generated the same metadata ETag, returned HTTP 304 for the changed HTML, and left the browser requesting old hashed JS/CSS assets that returned 404. Serving the old HTML against production reproduced a white screen.
- **Authorized cache-only exception:** after the original UI-only scope, the user explicitly approved a narrow Flask frontend-HTML caching fix. Commit `2d5b84b` (`Prevent stale frontend HTML after Vercel deployments`) disables conditional responses and ETags for the entrypoint and sends `Cache-Control: no-store`. This covers `/`, GET/HEAD `/index.html`, and the existing SPA fallback without changing route definitions, APIs, recommendation logic, or static JS/CSS caching.
- **Cache verification:** all 154 Python tests passed, including four new tests in `tests/test_frontend_html_cache.py`. They cover same-size/same-timestamp changes, old ETag/Last-Modified headers, direct HTML and fallback delivery, asset conditional caching, HEAD/OPTIONS, and missing paths/builds. Fresh and repeated local browser loads rendered without runtime errors.
- **Production delivery verified:** [PR #12](https://github.com/marvisrego/KnowledgeGraph_Recommendation/pull/12) merged into `main` as `2e37d17`. Vercel production deployment `6400532594` succeeded after preview `6400513015`. Both `/` and `/index.html` returned HTTP 200, the current asset references, `Cache-Control: no-store`, and no ETag when the old ETag and Last-Modified headers were supplied. Production browser load and reload mounted the advisor without runtime errors. Screenshot: `cache-fix-production-verified.png` in the local evidence directory.
- **Separate readiness limitation:** the latest production `/api/status` returned HTTP 200 with `ready: false`, `graph_loaded: false`, `qdrant_loaded: true`, and zero loaded nodes/edges. Live successful recommendation generation was not verified in this session; no graph/database repair was attempted. Historical successful cloud counts below are not current readiness evidence.
- **Local evidence:** `C:/Users/marvi/AppData/Local/Temp/career-ui-research/` contains reference/before/after screenshots, the white-screen reproduction, browser check scripts, JSON results, and the cache-fix screenshot. Successful result/graph UI captures use browser-only fixtures, not live model output. These temporary artifacts are not committed and may be cleaned up by Windows.

### 2026-09-06 main advisor UI refresh

- Refined the main React advisor surface while preserving the existing API, conversation state, controls, routes, and response rendering behavior.
- Added a clearer evidence-led welcome state, bounded conversation column, larger labeled composer, stronger focus states, and improved desktop/tablet/mobile spacing.
- Reworked role, stats, course, evidence, learning-plan, and pipeline presentation for clearer hierarchy and more readable content density.
- Added semantic status/expanded-state attributes, a skip link, explicit message field metadata, reduced-motion handling, safe-area padding, and improved dark-theme contrast.
- Regenerated the committed `public_react/` Vite bundle for Flask and Vercel.
- Verification: `npm.cmd run build` passed; Playwright interaction checks passed at 1440, 1280, 820, 390, and 320px with no console errors, overflow, or axe violations. `npm.cmd run lint` passed with one non-blocking existing state-in-effect warning in `PipelineIndicator`.
- A local `/api/status` request returned HTTP 200 but reported the external Qdrant Cloud connectivity check as unavailable in this environment; no UI code depends on that local connectivity result.

### 2026-09-05 cloud migration result

- Rebuilt from ONET, ESCO, and train-only Karrierewege source data. JobHop remains an evaluated auxiliary research dataset and is not inserted into the production KG because it did not improve the primary top-K promotion gate.
- Quality controls normalize labels, reject unsupported node/relationship types, remove invalid/self/held-out/duplicate edges and isolates, and add 12 source-backed qualification nodes, 4 O*NET job-zone nodes, and 1,770 connectivity edges.
- Aura schema includes a unique `entity_key` constraint plus graph ID, entity ID, normalized title, type, and source indexes.
- Qdrant uses cosine distance, deterministic point IDs, keyword payload indexes, and payloads containing graph/source/type/title/text/record/chunk metadata.
- The accepted smoother remains the production ranker. Link prediction is scored and returned only as virtual missing-edge evidence; predictions are never persisted as observed transitions.
- Live validation passed: Aura counts and round-trip NetworkX counts both equal 19,241/240,906; Qdrant count is 3,932 and sample similarity search succeeded.
- Held-out test: Hits@1 `0.148318`, Hits@3 `0.285743`, Hits@5 `0.372281`, Hits@10 `0.505858`, MRR `0.261803`, source coverage `1.000000`.
- Vercel deployment is complete and the production configuration uses the external Aura and Qdrant connections. Database credentials remain environment-only and are not stored in Git.

### Implemented Novelty Contributions

See `novelty.md` and `ALL_STEPS.md` for the design, implementation, and evaluation record.

1. **Empirical Career Transition Edges** — 18,907 support-filtered `TRANSITIONS_TO` edges learned from training trajectories only (`src/karrierewege_preprocessing.py`)
2. **Skill-Gap-Aware Career Path Ranking** — deterministic owned-skill evidence, ESCO-comparable requirements, and accessible-to-aspirational ordering (`src/skill_gap.py`)
3. **Embedding-Smoothed Transition Inference** — soft-propagates transition distributions from semantically nearest ESCO role neighbours to source roles with no direct observations; validated on held-out split, locked config applied to test once (`src/transition_embedding.py`)
4. **Transition Effort Score (TES)** — multi-factor effort metric combining IDF-weighted skill gap, ISCO domain distance, empirical support, and skill transferability (`src/transition_effort.py`). Now includes `estimated_weeks_min/max` upskill time. Roles sorted low→high effort in response.
5. **KG Link Prediction** — LightGBM classifier predicting virtual missing `PREDICTED_TRANSITION` edges (`src/link_prediction.py`). The runtime supports the current 11-feature extractor and safely honors the bundled model's original 9-feature schema instead of bypassing LightGBM shape checks.
6. **Graph-Provenance Faithfulness Verification** — post-generation entity extraction + graph reachability check producing a faithfulness score (`src/faithfulness.py`). Now serializes as `score` (not `faithfulness_score`) and includes `matched_entities` list.
7. **Provenance-Traced Explanation Chains** — typed-edge path tracing from source to target role showing TRANSITIONS_TO/SIMILAR_TO/REQUIRES evidence (`src/explainability.py`)
8. **Qualification Scoring** — IDF-weighted fraction of essential skills owned per candidate role (`agents/nodes/qualification_node.py`)
9. **Skill Gap Analysis** — per-role ranked missing skills by TES reduction impact, quick wins, and blockers (`agents/nodes/skill_gap_node.py`)
10. **Learning Roadmap** — phased upskill plan with Coursera course groupings and week estimates (`agents/nodes/learning_plan_node.py`)
11. **ISCO Group Edges** — `SAME_ISCO_GROUP` structural edges between roles sharing a 2-digit ISCO code, capped at top-20 per group (`src/isco_edges.py`)

Direct-edge test baseline: Hits@5 `0.3702`, Hits@10 `0.5014`, MRR `0.2591`. Fine-tuned embedding smoother (locked test): Hits@5 `0.372273`, Hits@10 `0.505849`, MRR `0.261805`, source-role coverage `1.0000`.
Link prediction 5-fold CV (with embeddings): AUC `0.9344`, AP `0.8186`. Top features: Neighbour Evidence, Embedding Cosine Sim, Resource Allocation.

The 2026-08-27 comparison selects the validation-accepted smoothed ranking: test Hits@5 `0.3722`, Hits@10 `0.5058`, MRR `0.2617`, **coverage 0.7389 → 1.0000**. Standalone LP ranking did not pass the top-K promotion gate (test Hits@5 `0.0274`, Hits@10 `0.0447`), so runtime LP is deliberately a virtual coverage backfill after the stronger fused sources, not an equal-weight rank boost.

### 2026-08-27 PR review repair — current result

- One shared `is_training_transition()` predicate now protects smoothing, TES, candidate augmentation, traversal, explanation, evaluation, agent tools, graph API data, and LP training/indexing from validation/test leakage.
- Classic and LangGraph requests both call the same hybrid retriever: vector + direct/smoothed transition + role-relevant skill overlap, followed by LP-only missing-edge backfill.
- LP initialization now loads the saved LightGBM model, all 3,039 live ESCO embeddings, the training-transition index, per-source semantic neighbours, IDF, and graph feature caches. Predicted edges are request-local and are never persisted as observed KG edges.
- Readiness and TES use IDF-weighted, role-relevant requirements. ESCO optional language/comprehension requirements no longer inflate technical-role effort, while essential language skills still count for roles such as teachers. Unaligned ONET roles use only high-importance requirements (default `3.5`).
- Generation instructions require exact graph labels, bold role/skill mentions, explicit ownership before calling a skill a strength, and occupation-specific gaps ahead of generic language/comprehension unless those abilities are central to the role.
- LangGraph partial-context requests now route `traversal → explore → END`; they do not run effort, generation, faithfulness, explanations, or courses.
- Concurrent Aura/Qdrant/model startup is lock-protected. Graph request retries are identity-guarded to prevent stale rendering.
- Skip targets are focusable, graph nodes have a keyboard-accessible selector, and explanation panels use the shared message/evidence layout with a real graph icon.
- `.venv` now satisfies `requirements.txt`; LangGraph/LangChain are capped below `1.0` to avoid the local `langchain 0.3.x` conflict.
- Verified after the cloud migration: 143 unit tests pass; Python compilation, frontend production build, JavaScript syntax, and `pip check` pass. A real cold `/api/status` request returned HTTP 200 with Aura, Qdrant, smoothing, and link prediction all loaded.

The external data layer and Vercel deployment are complete. Required cloud/model values are configured through Vercel environment variables; `.env` itself remains uncommitted.

---

## File Map

```
config.py                     Settings dataclass — reads from .env
src/
  onet_preprocessing.py       ONET Excel → nodes/edges DataFrames
  esco_preprocessing.py       ESCO CSV → nodes/edges DataFrames
  graph_build.py              Merge into typed nx.MultiDiGraph; prune; atomic pickle save/load
  embeddings_index.py         Shared Azure embedding client; legacy offline Chroma utilities
  neo4j_store.py              Aura schema, batched upload, validation, and runtime graph loading
  qdrant_store.py             Qdrant collection rebuild, payload indexes, and runtime adapter
  graph_quality.py            Normalization, pruning, deduplication, and source-backed connectivity
  karrierewege_preprocessing.py  Stream-clean trajectories; aggregate/add transitions
  transition_embedding.py     Semantic-neighbour smoothing over train-derived transitions
  skill_gap.py                Skill ownership, comparable requirements, accessibility ranking
  inference_pipeline.py       Classic GraphRAG orchestration using shared hybrid retrieval
  hybrid_retrieval.py         Multi-source fusion + cached virtual LP edge backfill
  transition_policy.py        Shared training-only transition evidence contract
evaluation/
  evaluate_karrierewege.py    Held-out transition evaluation
  evaluate_embedding_transitions.py  Local-only validation-tuned, locked-test experiment
  ranking_ablation.py         Semantic/transition/skill-gap diagnostic ablation
tests/                        Deterministic offline unit tests
build_graph.py                CLI: graph, alignment, and transition build modes
career_kg_web.py              Flask server — port 8001
coursera_client.py            Coursera search scraper (no API key needed)
templates/graph.html          Interactive Cytoscape.js knowledge graph viewer
frontend/                     React/Vite source; frontend/public contains graph viewer assets
public_react/                 Committed production build served by Flask/Vercel
test_apis.py                  Smoke-test for Azure embed + Cohere rerank
novelty.md                    Research contributions: Part 1 = implemented (A/B/C); Part 2 = proposed for supervisor (P1–P4)
ALL_STEPS.md                  Chronological history, implementation, commands, and results
```

---

## Cloud Graph/Vector Stats (rebuilt 2026-09-05)

| Stat | Value |
|---|---|
| Total nodes | 19,241 |
| Total relationships | 240,906 |
| Role nodes | 3,932 (893 ONET + 3,039 ESCO) |
| SIMILAR_TO edges | 132 (66 pairs × 2 directions) |
| TRANSITIONS_TO edges | 18,907 (train only, support ≥5) |
| Transition source roles | 765 |
| Transition incident roles | 824 |
| Qdrant collection | 3,932 live role vectors; 3,072 dimensions; cosine distance |
| Similarity threshold | 0.65 (cosine, ONET↔ESCO alignment) |
| ONET importance threshold | 3.5 (IM scale) |

**Cloud and reproducibility locations:**
```
Neo4j AuraDB              Authoritative career knowledge graph
Qdrant Cloud              Authoritative role vector collection
Data/                     Ignored source datasets retained locally for a future full rebuild
graph/ and index/         Removed legacy local databases; not used by runtime
```

---

## API Configuration (.env)

```
CHAT_MODEL_API_KEY      = <set in .env>
CHAT_MODEL_ENDPOINT     = https://career.azure-api.net/career-graph-ai/openai/v1/responses
CHAT_MODEL              = gpt-5.4-nano

EMBED_MODEL_API_KEY     = <set in .env>
EMBED_MODEL_ENDPOINT    = https://career.azure-api.net/career-graph-ai
EMBED_MODEL             = text-embedding-3-large

COHERE_RERANK_API_KEY   = <set in .env>
COHERE_RERANK_ENDPOINT  = https://career.azure-api.net/career-graph-ai/providers/cohere/v2/rerank
COHERE_RERANK_MODEL     = Cohere-rerank-v4.0-pro

KG_URI                  = <Neo4j Aura URI>
KG_USER                 = <Neo4j user>
KG_PASS                 = <Neo4j password>
KG_ID                   = <stable project graph ID>
NEO4J_DATABASE          = <optional; leave blank for Aura default>
VECTOR_ENDPOINT         = <Qdrant Cloud URL>
VECTOR_PASS             = <Qdrant API key>
QDRANT_COLLECTION       = career_roles

SIMILARITY_THRESHOLD    = 0.65
ONET_IMPORTANCE_THRESHOLD = 3.5
TRANSITION_MIN_COUNT    = 5
TRANSITION_CHUNK_SIZE   = 200000
TRANSITION_SMOOTHING_ENABLED = true
TRANSITION_SMOOTHING_NEIGHBOURS = 25
TRANSITION_SMOOTHING_DIRECT_WEIGHT = 0.91
TRANSITION_SMOOTHING_TEMPERATURE = 0.06
KARRIEREWEGE_MAX_INVALID_ROW_RATIO = 0.001
LINK_PREDICTION_ENABLED = true
LINK_PREDICTION_MODEL_PATH = artifacts/link_prediction/link_predictor.json
USE_LANGGRAPH = false
```

**API shape details:**
- Chat: OpenAI **Responses API** — POST to full endpoint, body `{"model":..., "input":[...messages...], "max_output_tokens":...}`, response at `output[*].type=="message" → content[0].text`
- Embed: POST to `EMBED_MODEL_ENDPOINT + "/openai/v1/embeddings"`, body `{"model":..., "input":[...]}`
- Rerank: POST to `COHERE_RERANK_ENDPOINT`, headers `{"api-key":...}`, body `{"model":..., "query":..., "documents":[...], "top_n":N}`, response at `results[*].relevance_score`
- All HTTP via `urllib.request` — no OpenAI SDK

**gpt-5.4-nano is a reasoning model** — it consumes tokens on internal chain-of-thought before outputting text. Token budgets in `inference_pipeline.py`:
- Routing: `max_tokens=2000`
- Generation: `max_tokens=1000` (responses are now 2–3 sentences)

**Available chat models on this Azure gateway** (all support Responses API):
`gpt-5.4-nano`, `gpt-5.4-mini`, `gpt-5.4-pro`, `gpt-5.4`, `gpt-5.6-sol`, `gpt-5.6-luna`, `gpt-5.6-terra`, `gpt-5.2`, `gpt-5.1`, `gpt-5-pro`, `gpt-5`, `gpt-chat-latest`
*Avoid `-codex` and `-reasoning` variants.*

---

## 7-Step Inference Pipeline (`src/inference_pipeline.py`)

```
Full conversation history (messages[])
  │
  ▼
1. route_intent()        — Responses API: detect student vs professional,
                           check if enough context; returns follow-up Qs if not.
                           Extract explicit current role, owned skills, and goal.
  │ {"has_context", "user_type", "current_role", "skills", "career_goal"}
  ▼
2. retrieve_candidates() — Embed last query → Qdrant top-50 live role nodes;
  │                        filter stale IDs and add empirical destinations
  ▼
3. rerank_candidates()   — Cohere rerank → top-8 anchors, then order by
  │                        skill accessibility → transition → semantic evidence
  │
  ▼
4. traverse_graph()      — NetworkX: for each anchor collect
                           REQUIRES (ONET IM≥3.5, ESCO essential only)
                           BELONGS_TO, BROADER_THAN, NARROWER_THAN, SIMILAR_TO,
                           and bounded TRANSITIONS_TO evidence
  │
  │  ── If has_context==false: return explore_data (skills + roles panels)
  │     and follow-up questions. Skip steps 5–7.
  ▼
5. generate_response()   — Responses API with user_type-specific system prompt:
                           STUDENT  → 2–3 sentence careers advisor response
                           PROFESSIONAL → 2–3 sentence career coach response
                           Both: bold role/skill names only; no headings/bullets
                           Full history passed so model has conversation context.
  │
  ▼
6. fetch_coursera_courses() — scrape Coursera for anchor role titles +
                              essential skills → return list[dict] (structured, up to 5)
  │
  ▼
7. _build_path_data()    — roles, readiness, have/need skills, transition proof,
                           and SIMILAR_TO pairs → structured frontend data
```

**`run_query()` returns a dict:**
```python
{
  "message": str,
  "courses": list[dict],
  "path":    {"roles": [...], "skills": [...], "similar_pairs": [...]},  # full context
  "explore": {"roles": [...], "skills": [...]},                          # partial context only
  "evidence": {"current_role": {...}, "matched_skills": [...]}           # explicit evidence
}
```
`career_kg_web.py` passes all five fields to the frontend as JSON.

**Partial context flow:** when `has_context==false` (e.g. user only gives degree, no goal),
the pipeline still runs retrieval+traversal and returns:
- `message` — the follow-up questions from the router
- `explore.skills` — top-20 essential skills across matched roles (wrapping chip grid)
- `explore.roles` — all 8 matched role cards (horizontal scrollable track)
This gives the user something visual to explore while they answer the follow-up.

**Conversation memory:** the web layer sends the full `messages[]` array from the client on every request. Both routing and generation receive the full history so the model never re-asks for information already given.

---

## Prompts Summary

### Routing (`_INTENT_SYSTEM_PROMPT`)
- Detects STUDENT / PROFESSIONAL / UNKNOWN from conversation history
- STUDENT needs: degree + skill + goal
- PROFESSIONAL needs: current role + skill + next step
- Returns pure JSON: `{"has_context": bool, "user_type": "...", "followup_questions": "..."}`

### Generation (2–3 sentences — no markdown headings or bullets)
- `_GENERATION_SYSTEM_PROMPT_STUDENT` — 1 short paragraph; names best-fit role in bold; one strength + one skill to develop
- `_GENERATION_SYSTEM_PROMPT_PROFESSIONAL` — 1 short paragraph; names best-fit next role in bold; one strength + one skill gap
- **Strict graph-only constraint:** every role/skill cited must appear in the graph triples

---

## Frontend

### Current React structure and response layout

- `frontend/src/App.tsx` composes `StatusPanel` and `ChatShell`; `frontend/src/index.css` supplies the visual system. Vite writes the committed deployment bundle to `public_react/`.
- `ChatShell` retains the welcome message, user/assistant conversation, pipeline loading indicator, error feedback, New chat, and the labeled composer. Enter sends; Shift+Enter inserts a newline; the existing 4,000-character limit and loading/empty-input disabling remain unchanged.
- When supplied by the existing response, assistant results show faithfulness, top-role statistics, role cards, transferable skills, skill-gap/learning-plan tabs, a horizontal course track, and provenance evidence. Data, ordering, score calculations, field visibility, and disclosure defaults were preserved.
- Role cards remain a two-column desktop grid and one column at narrower widths. Their effort-colored edges, skill chips, evidence disclosures, and course links retain their existing meaning and behavior.
- The legacy `chat.js` helpers referenced in older history are no longer the active frontend. See `ALL_STEPS.md` for that historical implementation.

### Design and motion

- **Palette:** charcoal `#0B0F14`, panel `#12171E`, raised surface `#1A222C`, border `#303D4C`, text `#EDF2F7`, muted text `#A2AFBF`, and cyan `#4CC2EA`. Success/warning/error and graph-category semantics remain distinct.
- **Typography:** Outfit for headings/body, JetBrains Mono for metrics/technical metadata; 16px conversation text and generally 14px supporting result text.
- Existing desktop/sidebar structure, safe-area spacing, visible focus, and touch-sized controls remain. Below 768px the same sidebar reflows above the chat; the page scrolls, and the conversation keeps its internal scrolling viewport so existing near-bottom detection still works.
- Hover transitions use roughly 180-220ms easing; linked course cards lift 2px. The welcome-only cursor glow is decorative, clipped, non-interactive, and disabled for reduced-motion/coarse-pointer users.
- React/Framer Motion and CSS implement presentation; this pass introduced no GSAP or other dependency. Existing content rendering and `AppErrorBoundary` remain unchanged.

### Knowledge Graph Viewer (`/graph`)

- Cytoscape.js interactions remain in `frontend/public/graph.js`; presentation is in `frontend/public/graph.css`. Both are copied into `public_react/` by Vite.
- The 2026-09-11 graph changes are CSS-only: clearer typography, controls, selector/inspector/tooltip surfaces, legend spacing, focus/hover states, and flexible header/workspace sizing.
- Existing graph sampling, canvas category colors, node selection, neighbourhood highlighting, tooltip behavior, zoom/pan/fit, and loading/error/retry behavior were preserved.

### HTML delivery and cache policy

- `_send_react_index()` in `career_kg_web.py` sends the frontend entrypoint with `conditional=False`, `etag=False`, and `Cache-Control: no-store`.
- Existing GET/HEAD `/index.html` requests use that policy through a narrowly scoped pre-request hook; OPTIONS and other static assets keep their prior handling.
- Do not restore metadata-based conditional caching for this HTML: Vercel's normalized timestamps allow same-size builds to share validators while referencing different hashed assets.
- This policy is in `dev` at `2d5b84b` and production through PR #12 merge `2e37d17`. Production headers and browser rendering were verified on 2026-09-11.

---

## Node / Edge Schema

**Node types:** `role` (ONET+ESCO), `element` (ONET), `skill` (ESCO), `skill_group` (ESCO), `isco_group` (ESCO)

**Edge types:**
| relation | meaning | key attrs |
|---|---|---|
| REQUIRES | role → skill/element | `requirement_level` (float ≥3.5 for ONET, `'essential'` only for ESCO), `domain`, `source` |
| BELONGS_TO | ESCO role → isco_group | `source='esco'` — URI-resolved (fixed) |
| BROADER_THAN | concept → broader concept | `pillar`, `source` |
| NARROWER_THAN | concept → narrower concept | `pillar`, `source` |
| SIMILAR_TO | ONET role ↔ ESCO role | `similarity` float ≥0.65, `source='alignment'` |
| TRANSITIONS_TO | ESCO role → empirically observed next ESCO role | `count`, `probability`, `source_total`, `split='train'`, `source='karrierewege'` |

`PREDICTED_TRANSITION` and `SEMANTIC_TRANSITION_BACKOFF` are virtual inference relations added only to request context/response payloads. They are labeled inferred in prompts and UI, never written into `graph.gpickle`, and never described as observed moves.

**Removed edge types (KG quality cleanup):**
- `RELATED_TO` — generic skill-skill taxonomy links; excluded at build time
- ESCO `optional` REQUIRES edges — excluded at build time and inference time
- Work Values / Work Styles ONET element nodes — excluded at build time

---

## KG Quality Improvements (2026-07-27)

| # | File | What changed |
|---|---|---|
| 1 | `onet_preprocessing.py` | Excluded Work Values + Work Styles element nodes (`_EXCLUDED_DOMAINS`) |
| 2 | `esco_preprocessing.py` | Fixed BELONGS_TO edges: `iscoGroup` code → URI via ISCOGroups_en.csv lookup |
| 3 | `esco_preprocessing.py` | `_build_skill_skill_edges` drops RELATED_TO; keeps only BROADER_THAN/NARROWER_THAN |
| 4 | `graph_build.py` | New `prune_graph()`: removes roles with <3 REQUIRES edges (123 removed) + isolates (2,906 removed) |
| 5 | `.env` | `SIMILARITY_THRESHOLD` raised 0.45 → 0.65 (SIMILAR_TO pairs: 12,631 → 66) |
| 6 | `.env` | `ONET_IMPORTANCE_THRESHOLD` added at 3.5 (was 3.0 hardcoded) |
| 7 | `inference_pipeline.py` | ESCO REQUIRES: `essential` only (dropped `essential/optional` and `optional`) |

---

## Running the Project

```bash
# Start server (graph already built)
python career_kg_web.py
# → http://127.0.0.1:8001

# Check health
curl http://127.0.0.1:8001/api/status
# → {"ready": true, "graph_loaded": true, "qdrant_loaded": true, ...}

# Smoke-test APIs
python test_apis.py

# Kill stale server processes
python -c "
import subprocess
out = subprocess.check_output('netstat -ano', shell=True).decode()
for line in out.splitlines():
    if ':8001' in line and 'LISTEN' in line:
        pid = line.strip().split()[-1]
        subprocess.run(['taskkill', '/PID', pid, '/F'], capture_output=True)
        print('Killed', pid)
"
```

**Rebuild only if needed:**
```bash
python build_graph.py             # rebuild graph only (no API calls)
python rebuild_databases.py       # rebuild and validate Neo4j Aura + Qdrant Cloud
python build_graph.py --align-only  # recompute SIMILAR_TO edges only (calls Azure embed API)
python build_graph.py --transitions-only  # replace train-derived transition edges only
python evaluation/evaluate_karrierewege.py  # validation/test metrics
python evaluation/ranking_ablation.py       # fixed-query ranking diagnostic
```

---

## Verification Checklist

Latest session verification (2026-09-11):

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -q
# 154 tests passed after the HTML cache fix.
.venv\Scripts\python.exe -m unittest discover -s tests -p test_frontend_html_cache.py -v
# Four focused cache regression tests.
```

From `frontend/`, `npm.cmd run build` and `npm.cmd run lint` passed during the visual pass; lint retains one known warning. Browser fixtures passed 74 checks and sampled axe audits. The cache-only change did not modify or rebuild the frontend bundle.

Production checks after PR #12 passed for `/` and `/index.html` with prior `If-None-Match` and `If-Modified-Since` headers: HTTP 200, fresh HTML, `Cache-Control: no-store`, and no ETag. Production browser load/reload with the old validators also passed. Retain these checks for future frontend deployments. Check cloud readiness independently; a rendered UI and HTTP 200 from `/api/status` do not establish that `ready` is true.

The commands below are historical graph/API checks. In particular, the local pickle check predates the cloud migration; do not recreate local stores just to run it. Historical expected counts are not current runtime measurements.

```bash
# Graph sanity
python -c "
import pickle; G=pickle.load(open('graph/graph.gpickle','rb'))
print(G.number_of_nodes(), G.number_of_edges())
bad=[e for _,_,e in G.edges(data=True) if e.get('relation')=='REQUIRES' and e.get('source')=='onet' and str(e.get('requirement_level',99)).replace('.','',1).isdigit() and float(e.get('requirement_level',99))<3.5]
print('Bad ONET edges (should be 0):', len(bad))
optional=[e for _,_,e in G.edges(data=True) if e.get('relation')=='REQUIRES' and e.get('source')=='esco' and e.get('requirement_level')!='essential']
print('ESCO non-essential edges (should be 0):', len(optional))
"
# Expected: 19225 224711 / Bad ONET edges: 0 / ESCO non-essential: 0

# API response shape check
curl -s -X POST http://127.0.0.1:8001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"I finished a BSc in Computer Science, I know Python and SQL, I want to go into data science."}]}' \
  | python -c "import sys,json; d=json.load(sys.stdin); print('status:', d['status']); print('courses:', len(d.get('courses',[])));  print('path roles:', len(d.get('path',{}).get('roles',[])));  print('msg[:200]:', d['message'][:200])"
# Expected: status: ok, courses: 3-5, path roles: 3-8, message is 2-3 sentences

# Partial context check (should return explore panels, not full path)
curl -s -X POST http://127.0.0.1:8001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"I studied Computer Science"}]}' \
  | python -c "import sys,json; d=json.load(sys.stdin); print('has_path:', bool(d.get('path',{}).get('roles'))); print('explore_roles:', len(d.get('explore',{}).get('roles',[]))); print('explore_skills:', len(d.get('explore',{}).get('skills',[])))"
# Expected: has_path: False, explore_roles: 8, explore_skills: up to 20
```

---

## Git

```bash
# Commit author: marvisrego / marvisregodab@gmail.com
git config user.name "marvisrego"
git config user.email "marvisregodab@gmail.com"

# Push to dev
git push origin dev
```

---

## Known Issues / TODO

- [x] Cache-fix PR #12 merged as `2e37d17`; production deployment `6400532594` passed. Old conditional headers now receive fresh 200/no-store HTML, and production browser load/reload renders correctly. Ctrl+Shift+R remains a one-time recovery option for an already-open stale page.
- [ ] Investigate the separately observed production readiness state (`graph_loaded: false`, `qdrant_loaded: true`, `ready: false`, checked 2026-09-11) before claiming live recommendation availability. Database/recommendation behavior was not changed in this UI/cache session.

- [ ] Coursera client scrapes HTML (no API key) — may break if Coursera changes their page structure
- [ ] Knowledge graph viewer uses `cose` layout which is slow for >300 nodes — consider pre-computing layout positions and caching as JSON
- [ ] SIMILAR_TO edges are now only 66 pairs (threshold 0.65) — may be too few for cross-framework traversal; monitor recommendation quality and lower threshold to 0.60 if needed
- [ ] ONET roles have NO ISCO codes in the graph (0/893). ESCO roles all have them (3039/3039). Need SOC→ISCO-08 crosswalk from BLS for contribution #3.
- [x] Qdrant was recreated with exactly 3,932 live role records; the 123 stale legacy Chroma records are gone.
- [x] `.agents/`, `.claude/`, `CODE/`, raw data, graph binaries, and the local vector index remain excluded. The deployed runtime is connected to Aura/Qdrant through environment variables.

---

## Additional Datasets (in `data/`)

| Dataset | Location | Rows | Format | Status |
|---|---|---|---|---|
| **Karrierewege** | `data/Karrierewege/` (train/test/validation splits) | 2,480,369 career steps | CSV: _id, experience_order, preferredLabel_en/de, description_en/de, skills | **IMPLEMENTED** — 18,907 train-only empirical transition edges |
| LinkedIn | `data/Linkedin/postings.csv` + subdirs | 3,383,601 postings | CSV: job_id, title, description, skills_desc, salary, company | Skip — skills too coarse (36 categories) |
| Monster.com | `data/monster_com-job_sample.csv` | ~22,000 postings | CSV: job_title, job_description, sector | Skip — small, noisy |
| Job descriptions | `data/job-descriptions/training_data.csv` | ~32,000 | CSV: company, description, title, length | Skip — NLP training only |

### Karrierewege Dataset Details
- **Origin:** "capadict" project; German career transition data with ESCO labels
- **Structure:** Each row = one career step for one person. Grouped by `_id`, ordered by `experience_order`
- **568,888 person-disjoint trajectories** spanning 1,295 ESCO occupations
- **Exact ESCO graph linkage:** 1,295/1,295 normalized English occupation labels (100% row coverage)
- **Prior mapping:** `artifacts/karrierewege_onet_mapping_summary.json` — 77.1% of ESCO titles matched to ONET SOC codes (999/1295 titles, covering 91.6% of data rows)
- **Skills column:** contains Python list of ESCO skill strings per career step
- **Train/test/val split:** ready for quantitative evaluation

---

## Example Prompts That Work

**Student (full context → path visual):**
> "I just graduated with a BSc in Computer Science, I know Python, SQL, and basic machine learning. I want to go into data science or AI roles."

**Professional (full context → path visual):**
> "I am a data analyst with 4 years experience using Python, SQL, and Tableau. I want to switch to a data science or machine learning role."

**Partial context → explore panels + follow-up questions:**
> "I studied Computer Science"
> "I am a software engineer"

**Pattern for full context:** `"I am a [role/degree] with [experience/year], I know [skills], and I want to [goal]."`

---

## Session Changes (2026-08-23)

### Supervisor feedback implemented

The supervisor requested: (1) a metric to measure effort/difficulty of career transitions, (2) KG link prediction to predict missing edges and improve Hit ratios, (3) general KG improvements.

### New contributions (D, E, P2, P4)

1. **Transition Effort Score (TES)** — `src/transition_effort.py`. Multi-factor composite: `TES = w1*SkillGapMagnitude + w2*DomainDistance + w3*(1-EmpiricalSupport) + w4*(1-Transferability)`. IDF-weighted skill gap measures rare vs common skill difficulty. ISCO Jaccard distance measures cross-domain difficulty. Empirical support uses TRANSITIONS_TO edges + smoother. Transferability measures IDF-weighted shared skills. Effort bands: Low (0–0.3), Moderate (0.3–0.6), High (0.6–1.0). Integrated into `inference_pipeline.py` — effort badges displayed on career-path role cards.

2. **KG Link Prediction** — `src/link_prediction.py`, `evaluation/evaluate_link_prediction.py`. 9 graph-structural features (Common Neighbours, Jaccard, Adamic-Adar, Resource Allocation, Preferential Attachment, Embedding Cosine Sim, Same ISCO Group, IDF Skill Overlap, Neighbour Evidence). LightGBM classifier trained on 18,907 positive + 94,535 negative samples. **5-fold CV AUC: 0.9344, AP: 0.8186**. Top features: Neighbour Evidence (426K gain), Embedding Cosine Sim (44K). Production uses the portable `artifacts/link_prediction/link_predictor.json`; the obsolete pickle copy was removed. Source coverage: 100%.

3. **Graph-Provenance Faithfulness Verification** — `src/faithfulness.py`. Extracts bolded plus unbolded role/skill candidates, resolves normalized graph labels and safe short aliases such as parenthetical skill names, then checks BFS reachability within 3 hops. Score = |matched & reachable| / |total entities|.

4. **Provenance-Traced Explanation Chains** — `src/explainability.py`. Traces typed edges from source role to each recommended role (TRANSITIONS_TO → SIMILAR_TO → REQUIRES), producing human-readable evidence chains.

### KG enrichment

- `src/kg_enrichment.py` — computed IDF for 13,570 skills/elements; ISCO 2-digit codes for 3,039 ESCO roles; mapped 893/893 ONET roles to ISCO-08 via `Data/crosswalks/soc_isco08_crosswalk.csv`
- Graph re-saved with enrichment attributes (no API calls needed)

### Configuration additions

```
EFFORT_WEIGHT_SKILL_GAP = 0.35
EFFORT_WEIGHT_DOMAIN = 0.15
EFFORT_WEIGHT_EMPIRICAL = 0.25
EFFORT_WEIGHT_TRANSFERABILITY = 0.25
LINK_PREDICTION_ENABLED = true
LINK_PREDICTION_MODEL_PATH = artifacts/link_prediction/link_predictor.json
```

### New CLI flags

```bash
python build_graph.py --enrich               # Add IDF + ISCO attributes
python build_graph.py --train-link-predictor  # Train LP model
```

### Test suite at that historical checkpoint: 100 tests passing

| Test file | Count |
|-----------|-------|
| `test_kg_enrichment.py` | 11 |
| `test_transition_effort.py` | 25 |
| `test_link_prediction.py` | 12 |
| `test_faithfulness.py` | 15 |
| `test_explainability.py` | 9 |
| (original 6 files) | 28 |

### Frontend changes

- Effort badges on career-path role cards (`.path-effort-badge`, `.effort-low/moderate/high`)
- Green/amber/red styling for Low/Moderate/High effort transitions

### Remaining work after the 2026-08-27 repair

- Externalize graph, vector, and model storage before deployment work.
- Qdrant was fully rebuilt with exactly one vector per live role, resolving the 123-record legacy mismatch.
- Improve/retrain LP before allowing its score to promote top-K results. It is currently exposed only as clearly inferred missing-edge backfill.
- Expand generation-quality and latency evaluation beyond the deterministic/unit and two end-to-end local smoke cases.

---

## Session Changes (2026-09-05): transition-metric improvement

1. **Graph audit** — 19,225 nodes and 224,711 edges passed endpoint, isolate, role-requirement, transition-value, and held-out-edge checks. No broad KG pruning was justified; repeated taxonomy labels occur across legitimate ISCO hierarchy levels.
2. **JobHop v2 acquired locally** — verified CC BY 4.0 Parquet splits under ignored `Data/JobHop_v2/`; added checksum manifest and research-only `pyarrow` dependency.
3. **JobHop cleaning** — direct mapping to 2,973 live ESCO codes; training produced 1,391,276 accepted experiences, 858,913 valid transitions, and 295,825 distinct transition pairs. Official person splits do not overlap. Unknown/unmapped codes and ambiguous same-start pairs are excluded.
4. **Source-grouped LambdaMART** — implemented 5-fold person-disjoint training with 33 direct, semantic, walk, embedding, ISCO, skill, popularity, provenance, and JobHop features. True held-out targets are never injected into candidates.
5. **Ranker outcome** — JobHop raised fold candidate recall from about 83.7% to 84.2–84.5%, and standalone test Hits@10 reached 0.509095, but MRR/Hits@5 regressed. The acceptance gate rejected both standalone and protected-head fusion, so runtime promotion is disabled and the smoother remains authoritative.
6. **Accepted metric improvement** — a 63-configuration validation-only fine search selected 25 neighbours, direct weight 0.91, and temperature 0.06. Frozen test metrics improved over the previous accepted smoother: MRR 0.261703 → 0.261805, Hits@5 0.372183 → 0.372273, Hits@10 0.505809 → 0.505849, and destination coverage 0.953196 → 0.958322.
7. **Evaluation additions** — transition metrics now include observation-weighted and macro NDCG@K. Local artifacts retain full experiment results; raw data, graph, Chroma, and caches remain ignored/local-only, while the small accepted LP model is bundled for runtime inference.

---

## Session Changes (2026-07-27)

1. **Shorter LLM responses** — generation prompts cut to 2–3 sentences (1 paragraph); `max_tokens` dropped 4000 → 1000; supervisor feedback: less text, more visual
2. **Linear career path visual** — `_build_path_data()` in `inference_pipeline.py`; `addCareerPath()` + CSS in `chat.js`/`chat.css`; horizontal scrollable role cards with skill chips + arrow connectors
3. **Explore panels for partial context** — `_build_explore_data()` in `inference_pipeline.py`; `addExplorePanels()` + CSS; when user only gives degree/role with no goal, shows skills grid + roles track alongside follow-up questions
4. **More recommendations** — `rerank_top_n` 5 → 8 in `config.py`; Coursera limit 3 → 5
5. **KG quality — 7 improvements** — see KG Quality Improvements table above; graph rebuilt and re-embedded; node count reduced from 22,259 → 19,225; SIMILAR_TO pairs from 12,631 → 66
6. **Skill chip overflow fix** — `white-space: normal; word-break: break-word` on `.path-skill-chip`; skill titles truncated at 38 chars on backend; scroll track `padding-right` to prevent last card clipping

---

## Session Changes (2026-08-01)

1. **Novelty contributions brainstormed** — 6 candidate research contributions identified and documented in `novelty.md`
2. **Dataset assessment** — evaluated all datasets in `data/` folder. Decision: use Karrierewege (2.48M ESCO-native career steps), skip LinkedIn/Monster/job-descriptions
3. **Top novelty: Empirical Transition Edges** — integrate Karrierewege as `TRANSITIONS_TO` edges (new edge type) weighted by frequency. Strongest contribution because grounded in real data.
4. **Key finding: ONET has no ISCO codes** — verified that 0/893 ONET role nodes have `isco_group` attribute. ESCO has all 3039. Contribution #3 (ISCO alignment validation) requires a SOC→ISCO crosswalk from BLS.
5. **Supervisor guidance:** wants 4-5 novelty ideas to pick from; evaluation comes separately later; KG improvements may continue later
6. **Next research option:** Search for additional structured skills, transitions, or demand datasets that map to ESCO/ONET.

---

## Session Changes (2026-08-17)

1. **novelty.md restructured** — divided into Part 1 (implemented: A, B, C) and Part 2 (proposed for supervisor: P1–P4). Contribution C (Embedding-Smoothed Transition Inference, `src/transition_embedding.py`) surfaced as a third implemented contribution previously embedded in session notes but not counted separately.
2. **Literature gap tables updated** — 2024–2026 papers (AdaptJobRec AAAI 2026, Cui et al. ACM TOIS 2026, Wang et al. ACL 2025, CE2-GLF 2026, MELO 2024, Dawoud 2026) added to novelty tables to strengthen the no-prior-work claim for each contribution.
3. **Supervisor options table added** — four combinations (A: already done; B: +P2 faithfulness; C: +P1+P2; D: +P4 explainability) with effort estimates so the supervisor can confirm which proposals to implement.
4. **HANDOFF.md updated** — implemented contributions list updated to three; file map description for novelty.md updated.

---

## Historical Session Changes (2026-08-15)

This section records the August 15 checkpoints. The current 2026-08-27 metrics, configuration, and 124-test verification are authoritative and are summarized at the top of this handoff.

1. **Leakage-safe Karrierewege integration** — training builds the graph; validation/test are evaluation-only.
2. **Streaming cleaning** — handles chunk boundaries, exact/conflicting duplicates, missing orders, gaps, and self-transitions without loading description fields.
3. **Typed multigraph** — migrated to `MultiDiGraph`, preserving all 130 taxonomy/transition pair collisions.
4. **Graph enrichment** — added 18,907 support-≥5 transition edges while keeping 19,225 nodes and 132 alignment edges.
5. **Skill-gap ranking** — added deterministic possession checks, ESCO requirements, ONET alignment fallback, missing-evidence states, and stable ranking.
6. **Evidence UI** — role cards now show readiness, observed moves, owned skills, and development gaps.
7. **Evaluation** — validation Hits@5 0.3716/MRR 0.2586; test Hits@5 0.3702/MRR 0.2591.
8. **Verification** — 20 offline tests pass; full and partial-context live smoke tests return HTTP 200.

### Production UI/frontend pass

1. **Cohesive visual system** — unified the studio and graph viewer around the deep-space/graphite/slate palette, Outfit UI typography, JetBrains Mono evidence labels, consistent spacing, borders, and controls.
2. **Responsive application shell** — added bounded desktop chat, compact tablet presentation, narrow-screen layout, safe-area padding, scroll-snap evidence tracks, and touch-sized controls.
3. **Safe rendering** — assistant Markdown now passes through a strict allowlist sanitizer; course titles and API error messages use DOM text nodes; CDN failures degrade to usable text or explicit recovery states.
4. **Request integrity** — duplicate submissions are locked, New Chat aborts active work, and stale in-flight responses cannot enter a cleared conversation.
5. **Graph usability** — added responsive navigation/legend, progressive labels, persistent tap/click inspection, viewport-safe hover details, accessible controls, and retryable load failures.
6. **Validation** — JavaScript syntax, Python compilation, 20 offline unit tests, four local HTTP routes, desktop/tablet/mobile render checks, and blocked-CDN fallback renders passed. `/api/chat` was not invoked during the visual pass to avoid unnecessary external model calls.
7. **Deployment boundary** — `HANDOFF.md` and `ALL_STEPS.md` are committed project records but excluded from the Vercel function bundle; raw graph/vector assets remain local because Aura and Qdrant now provide external persistence.

### Embedding-smoothed transition pass

1. **Why embeddings are used** — title mapping was already complete, so raw Karrierewege rows were not re-embedded. Existing `text-embedding-3-large` ESCO role vectors identify semantically related transition-source roles.
2. **Leakage boundary** — only graph edges marked `split='train'` contribute destination distributions; validation selects parameters and the frozen configuration is applied to test once.
3. **Selected configuration** — 20 neighbours, direct-transition weight 0.90, softmax temperature 0.05.
4. **Locked test improvement** — MRR 0.259056 → 0.261703, Hits@5 0.370239 → 0.372183, Hits@10 0.501359 → 0.505809, and destination coverage 0.871500 → 0.953196.
5. **Runtime behavior** — direct observations retain counts/probabilities. Smoothed-only roles are marked `semantic_transition_backoff`, shown as inferred evidence, and never represented as directly observed moves.
6. **Failure behavior** — disabled smoothing, missing vectors, incompatible Qdrant data, or scorer errors fall back to the original direct transition expansion.
7. **Vector scope** — 3,039/3,039 live ESCO vectors loaded in evaluation; runtime preloads only 765 transition-source vectors and caches per-role rankings. No embedding API call or raw person-level embedding is used.
8. **Verification** — 28 offline tests pass; a mocked `/api/chat` request initialized the real-vector smoother, returned a ranked result, and exposed no smoothing error without calling external models.

---

## Post-deployment repair (2026-09-05)

- The deployed `/api/chat` endpoint was healthy, but ESCO skills arrived as `{id, title}` objects while the React role cards expected strings. `frontend/src/lib/normalizeChatResponse.ts` now normalizes those values and the explanation-field aliases at the API boundary.
- `AppErrorBoundary` prevents an unexpected result payload from blanking the entire application and provides a reload recovery action.
- Link prediction now loads the existing trained ensemble from `artifacts/link_prediction/link_predictor.json` through a NumPy-based evaluator. Its predictions were verified exactly equal to the original LightGBM Booster on a 100-row numerical comparison.
- Vercel no longer installs native LightGBM/scikit-learn packages. They remain in `requirements-research.txt` for offline training and evaluation. An existing Vercel value ending in `.pkl` automatically resolves to the sibling JSON model, so older environment configuration remains compatible.

### Fully online link-prediction training (2026-09-05)

- `.github/workflows/train-link-prediction.yml` provides a manual GitHub Actions job; GitHub Pages is not part of the architecture and Vercel remains the web/API host.
- `train_link_prediction_cloud.py` loads the graph read-only from AuraDB and existing ESCO vectors from Qdrant, so no local graph, Chroma store, raw dataset, or embedding API call is required.
- Validation uses source-role-disjoint GroupKFold and excludes held-out source transitions from the neighbour-evidence feature. It reports classification and ranking metrics and refuses publication below AUC 0.87, AP 0.68, Hits@5 0.89, or MRR 0.78.
- The verified cloud-data smoke run used 113,442 pairs (18,907 positive) and produced AUC 0.883384, AP 0.702746, Hits@1 0.718954, Hits@3 0.857516, Hits@5 0.912418, Hits@10 0.950327, and MRR 0.801180. These sampled-negative source-grouped diagnostics are a promotion gate, not a replacement for the separate Karrierewege held-out benchmark.
- An accepted run exports a portable model, uploads the model/report as a GitHub artifact, and can open a model-update PR against `dev`. It never writes predicted edges to AuraDB.
- Required Actions secrets are `KG_URI`, `KG_USER`, `KG_PASS`, `KG_ID`, `VECTOR_ENDPOINT`, and `VECTOR_PASS`; `NEO4J_DATABASE` is optional and `QDRANT_COLLECTION` is an optional repository variable.
- The full verification suite is now 150 tests. All six required Actions secrets and the `QDRANT_COLLECTION` repository variable are configured, and Actions has read/write plus pull-request creation permission. Secret values were never printed or committed.

### 2026-09-05 repository and Actions cleanup

- Reduced the manual cloud-training job timeout from 180 to 30 minutes and candidate artifact retention from 30 to 7 days to stay lightweight on the GitHub Student/Pro allowance.
- Pinned checkout, Python setup, and artifact upload to their current official v7 majors after the first run exposed Node.js 20 deprecation notices; this removes the obsolete action-runtime pins without adding another paid training run.
- Removed the obsolete pickle model, vanilla chat template/assets, unused Vite starter images, unused component scaffolding, and their Radix/Cytoscape npm dependencies.
- Moved the retained graph viewer CSS/JS into `frontend/public/`; Vite now copies them into `public_react/`, the single static root served by Flask and packaged by Vercel.
- Removed generated local `graph/`, `index/`, cloud rebuild cache, cloud smoke output, and the accidental workspace `~/` directory. The hosted Aura/Qdrant databases remain authoritative, and the generated caches can be recreated.
- Preserved ignored `Data/` source files because they are still required by `rebuild_databases.py` for a complete from-source rebuild.
- Delivery completed: PR #7 merged the workflow and deployment cleanup into `main`; Actions run `33975475939` completed successfully in 2m57s; its validated model PR #8 passed Vercel preview and was merged into `dev` as `32ce418`.
- First GitHub-hosted result: AUC `0.883507`, AP `0.702902`, Hits@1 `0.713725`, Hits@3 `0.853595`, Hits@5 `0.911111`, Hits@10 `0.949020`, and MRR `0.797201`. These source-grouped sampled-negative metrics passed every publication gate and remain separate from the official held-out Karrierewege benchmark.
- Final public validation passed: `/` returned the React app, `/api/status` reported Aura, Qdrant, smoothing, and link prediction ready with 19,241 nodes, 240,906 relationships, and 18,907 train transitions, and a representative `/api/chat` request returned HTTP 200 with a structured career path and no error.

### Empirical TES and ranking implementation (2026-09-12)

- `src/tes_calibration.py` validates portable, versioned TES calibration artifacts; absent or invalid artifacts safely use labelled legacy values. `evaluation/calibrate_transition_effort.py` fits train-only, inverse-stratum-weighted trajectory-choice proxy weights and learns effort-band boundaries on validation only.
- TES now treats absent skills, requirements, or ISCO data as unavailable evidence instead of maximum difficulty; it renormalizes available components and exposes calibration metadata plus the existing estimate's `heuristic` basis.
- `SmoothingConfig.prior_strength` enables empirical-Bayes count-aware shrinkage while an unset value retains the accepted fixed-weight behaviour. Tune `k`, temperature, and prior strength only with the offline prefix benchmark.
- `src/sequential_ranking.py` provides dependency-free runtime inference for promoted MLP/STEP artifacts. Internal intent extraction resolves only user-stated ordered career history; sequential candidates are fused inside the transition channel and cannot replace the smoother if the artifact is absent, invalid, or unpromoted.
- `evaluation/export_sequential_embeddings.py`, `evaluation/train_sequential_ranker.py`, `evaluation/evaluate_prefix_ranking.py`, and `evaluation/train_rotate_link_prediction.py` are offline research tools. They use an existing local graph if present, otherwise read AuraDB/Qdrant without writes. No learned artifact was generated or promoted in this delivery, so `SEQUENTIAL_RANKING_ENABLED` remains false.
- `docs/ALGORITHMS.md` contains the required algorithm steps and equations. `Writing/` was not modified.

## 2026-09-14 - Causal trajectory ranking, stability gate, and runtime safety

- The primary evaluation remains a person-disjoint Karrierewege prefix task:
  rank every live ESCO v1.2.1 occupation, exclude the current role, and derive
  all transition evidence from the training split only.
- A causal Transformer with a learned occupation-ID residual was pretrained on
  JobHop v2 training records and fine-tuned on Karrierewege training data. It
  achieved validation MRR `0.310101`, Hits@5 `0.433727`, Hits@10 `0.565431`
  (seed 17, 123,520 prefixes, coverage `1.0`). A validation-only frozen fusion
  reached MRR `0.312508`, Hits@5 `0.435622`, Hits@10 `0.566758`.
- Its recorded standalone test diagnostic was MRR `0.310800`, Hits@5
  `0.433687`, Hits@10 `0.563050` (122,918 prefixes, coverage `1.0`). This test
  result must not be treated as a newly blind promotion test because it was
  observed before multi-seed selection finished.
- Seed 29 completed with validation MRR `0.309984`, Hits@5 `0.432659`, and
  Hits@10 `0.564953`; seed 17 is the provisional validation leader. The fixed
  remaining seeds 41, 53, and 71 are running sequentially. No seed is chosen
  from test performance.
- JobHop personal source fields (dates, tenure, education, companies,
  locations, and individual skills) are not transferred into Karrierewege or
  production. The isolated JobHop feature-aware test benchmark (Hits@10
  `0.318897`, MRR `0.161371`) is not comparable to, and is not fused with, the
  primary Karrierewege metric.
- Added a dependency-free runtime implementation for the causal artifact, a
  checksum-bound promotion tool, and a frozen-fusion test evaluator. Local
  inference for the unpromoted 20.7 MB seed-17 artifact measured p50 `32.1 ms`
  and p95 `32.4 ms` for top-50 ranking; this is runtime-only, not end-to-end
  API latency. `SEQUENTIAL_RANKING_ENABLED` remains `false`.
- Promotion remains blocked until the fixed multi-seed selection, frozen
  full-candidate fusion test, person-bootstrap evidence, artifact/API checks,
  and an explicit blind-test disclosure all pass. The current best Hits@10 is
  still below the requested 60-65% target; do not claim otherwise.

## 2026-09-15 - Seed-41 frozen fusion result

- Completed four validation-only seeds (`17`, `29`, `41`, `53`). The user
  stopped the planned seed `71` before it produced an artifact, so do not
  describe this as a five-seed result. Seed 41 won the declared ordering
  (MRR, Hits@5, Hits@10) with validation MRR `0.310251`, Hits@5 `0.434165`,
  Hits@10 `0.564143`. The four-seed MRR standard deviation was `0.000113`.
- Seed 41's checksum-bound validation fusion selected `(0.5, 0.5, 0.0)` for
  `(neural, smoother, second-order)` on single-role histories and `(1.0, 0.0,
  0.0)` on multi-role histories. It produced validation MRR `0.312526`,
  Hits@5 `0.435962`, Hits@10 `0.565819`.
- The seed-41 test was read once after the manifest freeze: MRR `0.313438`,
  Hits@1 `0.194219`, Hits@3 `0.351234`, Hits@5 `0.435648`, Hits@10
  `0.564807`, NDCG@10 `0.361146`, coverage `1.0` (122,918 prefixes / 56,848
  people / 3,039 live candidates). Against the semantic smoother, deltas were
  MRR `+0.051597`, Hits@5 `+0.063367`, Hits@10 `+0.058950`; their paired
  person-bootstrap 95% intervals exclude zero.
- The 20.7 MB unpromoted seed-41 artifact loaded in `143.3 ms`; local top-50
  ranking p50 was `31.5 ms`, p95 `32.5 ms`. The full unit suite passed (182
  tests), `pip check` passed, and compilation passed. These are local checks,
  not a live API latency claim.
- `SEQUENTIAL_RANKING_ENABLED` remains `false` and no deployable artifact was
  created: final Hits@10 `0.564807` still does not meet the requested 0.60-0.65
  threshold. The semantic smoother remains production behavior.

### GitHub Actions boundary for sequential ranking

The causal trajectory model is deliberately **offline-only**. The existing
manual GitHub Actions workflow remains limited to LightGBM link-prediction
training; it uses a standard GitHub-hosted runner with a 30-minute timeout and
does not contain the ignored JobHop Parquet data or local role-embedding
artifact required by the sequential experiment. A reproducible multi-seed
causal run would require durable data storage/access, long-running compute,
and likely a paid larger or self-hosted runner. It must therefore be described
as offline research infrastructure in the paper, not as part of the current
GitHub Actions deployment pipeline. Any future workflow should be manual-only,
upload experiment artifacts, and never auto-promote or deploy a ranker.
