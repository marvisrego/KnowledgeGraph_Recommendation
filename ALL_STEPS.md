# All Steps — GraphRAG Career Advisor

This is the chronological, reproducible project record requested for the repository. It consolidates the older handoff history, the novelty decision, the 2026-08-15 implementation, commands, results, and known limitations. It intentionally excludes `.env` values, API keys, and person-level Karrierewege records.

## 1. Inherited project foundation

The starting system was a Flask GraphRAG career adviser backed by ONET and ESCO:

- ONET and ESCO preprocessing produced typed role, skill/element, skill-group, and ISCO-group nodes.
- Role embeddings were stored in ChromaDB.
- ONET↔ESCO `SIMILAR_TO` links were created with embedding similarity.
- The online pipeline performed intent routing, vector retrieval, Cohere reranking, graph traversal, grounded response generation, Coursera search, and structured career-path output.
- The web interface already supported full-context career paths and partial-context exploration panels.

### 2026-07-27 inherited quality work

The earlier session:

1. removed ONET Work Values and Work Styles;
2. corrected ESCO occupation-to-ISCO URI linkage;
3. removed generic ESCO `RELATED_TO` links;
4. pruned roles with fewer than three `REQUIRES` edges and then isolates;
5. raised the ONET↔ESCO alignment threshold to 0.65;
6. set ONET importance filtering to 3.5;
7. retained only essential ESCO requirements at inference;
8. shortened generated advice;
9. added career-path and partial-context visual components.

The inherited graph contained 19,225 nodes, 205,804 edges, 3,932 roles, and 132 directed `SIMILAR_TO` edges.

### 2026-08-01 novelty work

Six possible thesis contributions were documented in `novelty.md`. The strongest recommended pairing was:

1. empirical Karrierewege `TRANSITIONS_TO` graph edges;
2. skill-gap-aware career-path ranking.

The design approved on 2026-08-15 is stored at `docs/superpowers/specs/2026-08-15-karrierewege-skill-gap-design.md` and was committed as `cd01890`.

## 2. Repository and data audit

Commands used included file discovery, selective CSV reads, graph-pickle inspection, mapping comparison, split-integrity scans, and transition-frequency profiling. Only aggregate information was emitted.

Verified raw split sizes:

| Split | Rows | Trajectories |
|---|---:|---:|
| Train | 1,984,657 | 455,129 |
| Validation | 248,465 | 56,909 |
| Test | 247,247 | 56,850 |
| Total | 2,480,369 | 568,888 |

Corrections to the older handoff:

- The dataset contains 2,480,369 career steps, not 3,036,454.
- The three splits are person-disjoint; their identifier intersections are all zero.
- All 1,295 dataset occupation labels match ESCO graph roles exactly after conservative normalization, covering every raw row. No fuzzy ESCO mapping is necessary.
- The existing ONET mapping artifact remains a separate result: 999/1,295 labels mapped to ONET, covering 91.6% of rows.

The graph audit also found that a plain `DiGraph` could not preserve the new semantics: at support five, 130 ordered role pairs already carried an ESCO taxonomy relationship. Adding a transition to the same pair would overwrite it.

## 3. Approved methodological decisions

1. Build production transition edges from the training split only.
2. Reserve validation and test strictly for held-out evaluation.
3. Use `MultiDiGraph` so each pair can carry independent typed evidence.
4. Require exact normalized ESCO occupation linkage.
5. Accept only consecutive `experience_order` steps.
6. Collapse exact duplicate steps, discard conflicting same-order steps, and never bridge the resulting gap.
7. Remove same-role transitions.
8. Calculate probability against all valid non-self outgoing training observations before rare-edge filtering.
9. Retain edges with support of at least five.
10. Treat missing skill evidence as unavailable, not as zero readiness.
11. Treat empirical transitions as population-level observations, never causal or guaranteed outcomes.

## 4. Files added

- `src/karrierewege_preprocessing.py` — split discovery, streaming trajectory cleaning, aggregation, probabilities, quality reports, and transition-edge replacement.
- `src/skill_gap.py` — safe skill aliases, possession checks, negation/goal exclusion, ESCO requirements, ONET alignment fallback, role resolution, transition evidence, and ranking.
- `evaluation/transition_metrics.py` — observation-weighted and macro held-out metrics.
- `evaluation/evaluate_karrierewege.py` — validation/test evaluation CLI.
- `evaluation/ranking_ablation.py` — fixed-query semantic/transition/gap diagnostic.
- `tests/test_karrierewege_preprocessing.py`
- `tests/test_graph_build.py`
- `tests/test_transition_metrics.py`
- `tests/test_skill_gap.py`
- `tests/test_inference_evidence.py`
- `ALL_STEPS.md` — this record.

## 5. Files extended

- `config.py` — Karrierewege paths, support/chunk settings, and transition limits.
- `src/graph_build.py` — typed `MultiDiGraph`, legacy migration, deterministic keys, idempotent alignment, and atomic persistence.
- `src/embeddings_index.py` — public role-document builder and multigraph typing.
- `build_graph.py` — `--transitions` and `--transitions-only` modes.
- `src/inference_pipeline.py` — structured role/skill intake, stale-index filtering, empirical candidate expansion, gap ranking, transition traversal/context, and evidence payloads.
- `career_kg_web.py` — evidence response field, transition statistics, and transition viewer edges.
- `public/chat.js` and `public/chat.css` — accessibility meter, observed-transition proof, and owned/development skill states.
- `templates/graph.html` — Karrierewege title/legend and dashed `TRANSITIONS_TO` styling.
- `HANDOFF.md` and `novelty.md` — corrected facts and implementation status.

Existing uncommitted project work was extended in place and not reset or discarded.

## 6. Streaming transformation details

Only `_id`, `experience_order`, and `preferredLabel_en` are read from the multi-gigabyte CSV files. The final person run is carried across chunk boundaries. Completed identifiers are tracked so a later non-contiguous reappearance fails rather than producing a false sequence.

Per-person processing:

1. reject missing identifiers, invalid/non-integral orders, and empty titles;
2. normalize titles with NFKC Unicode normalization, case folding, punctuation handling, and whitespace collapse;
3. sort stably by order;
4. collapse exact duplicate rows;
5. remove conflicting labels at one order;
6. require an order difference of exactly one;
7. remove self-transitions;
8. map both endpoints exactly to ESCO nodes;
9. aggregate ordered pair counts and source totals.

Training cleaning output:

| Measure | Value |
|---|---:|
| Raw rows | 1,984,657 |
| Accepted steps | 1,984,594 |
| Exact duplicate rows removed | 1 |
| Ambiguous positions removed | 31 |
| Ambiguous rows removed | 62 |
| Nonconsecutive adjacent pairs rejected | 6 |
| Self-transitions removed | 542,656 |
| Valid non-self transitions | 986,803 |
| Distinct ordered pairs before support filtering | 90,170 |

Support analysis:

| Minimum count | Retained edges | Retained observations | Ratio |
|---:|---:|---:|---:|
| 1 | 90,170 | 986,803 | 100.0% |
| 2 | 44,658 | 941,291 | 95.39% |
| 3 | 30,221 | 912,417 | 92.46% |
| 5 | 18,907 | 874,391 | 88.61% |
| 10 | 10,301 | 818,640 | 82.96% |
| 25 | 4,644 | 734,843 | 74.47% |

The selected threshold is five.

## 7. Graph enrichment

Before replacing the graph artifact, the original was copied to:

`graph/graph.pre-karrierewege.gpickle`

Enrichment command:

```powershell
python build_graph.py --transitions-only
```

The command loaded the existing graph, migrated it to `MultiDiGraph`, generated training edges, replaced only earlier Karrierewege edges, wrote the quality report, and atomically replaced `graph/graph.gpickle`.

After enrichment:

| Measure | Value |
|---|---:|
| Nodes | 19,225 |
| Roles | 3,932 |
| Total edges | 224,711 |
| `TRANSITIONS_TO` | 18,907 |
| Transition source roles | 765 |
| Roles incident to a transition | 824 |
| Preserved parallel taxonomy/transition collisions | 130 |
| `SIMILAR_TO` edges retained | 132 |
| Self-transition edges | 0 |
| Non-training transition edges | 0 |
| Invalid probabilities | 0 |
| Maximum retained outgoing probability mass | 0.9917 |

Node content did not change, so the embeddings were not regenerated.

## 8. Held-out transition evaluation

Command:

```powershell
python evaluation/evaluate_karrierewege.py
```

Training-edge destinations were ranked by probability, count, title, and identifier. Missing held-out destinations contributed zero to overall ranking metrics.

| Metric | Validation | Test |
|---|---:|---:|
| Valid observations | 123,520 | 122,918 |
| Source-role coverage | 0.7351 | 0.7389 |
| Observation coverage | 0.9942 | 0.9940 |
| Destination coverage | 0.8707 | 0.8715 |
| Hits@1 | 0.1458 | 0.1478 |
| Hits@3 | 0.2852 | 0.2850 |
| Hits@5 | 0.3716 | 0.3702 |
| Hits@10 | 0.5045 | 0.5014 |
| MRR | 0.2586 | 0.2591 |
| Macro MRR | 0.1634 | 0.1658 |

Outputs:

- `artifacts/karrierewege/data_quality.json`
- `artifacts/karrierewege/evaluation.json`

## 9. Skill-gap contribution

The intake model returns a current role, explicit skills, and goal. Deterministic code owns the final evidence decision:

- preferred ESCO skill labels and safe parenthetical aliases are exact-matched;
- possession language must be closer to a mention than goal, learning, or negation language;
- terse skill-list answers are supported;
- assistant-authored text is excluded;
- a desired skill is never counted as owned;
- missing matches remain explicit.

ESCO roles use essential `REQUIRES` skills directly. ONET roles use the strongest aligned ESCO role when available. Unaligned ONET roles receive `null` accessibility with `no_esco_alignment`, avoiding an invalid comparison between granular ESCO skills and broad ONET elements.

For comparable roles:

```text
have = owned skills ∩ required skills
need = required skills − owned skills
accessibility = |have| / |required skills|
gap = 1 − accessibility
```

Semantic retrieval and reranking select relevant roles. Professionals additionally contribute empirically observed destinations from their resolved current role. The final displayed shortlist is ordered by accessibility, direct transition probability/count, semantic score, and deterministic textual ties.

## 10. Ranking diagnostic

Command:

```powershell
python evaluation/ranking_ablation.py
```

The four fixed diagnostic cases compare semantic-only, transition-expanded, and transition-plus-gap stages without generating answers. All four final lists were monotonically ordered by available accessibility evidence. Gap ranking removed five accessibility inversions from the same expanded shortlists. Transition expansion introduced additional candidates in three professional cases, and transition-supported roles survived reranking in two cases.

This is a pipeline diagnostic, not a substitute for a larger human relevance evaluation.

Outputs:

- `artifacts/karrierewege/ranking_ablation.json`
- `artifacts/karrierewege/ranking_ablation.csv`

## 11. Inference and interface behavior

For professionals with a resolvable ESCO current role:

1. semantic retrieval runs;
2. stale Chroma IDs are removed;
3. high-support outgoing empirical roles augment the candidate pool;
4. the combined pool is reranked;
5. skill gaps are calculated;
6. roles are ordered accessible-to-aspirational;
7. graph traversal adds bounded transition triples;
8. the LLM context includes count/probability evidence and a population-level caution;
9. the API returns structured evidence;
10. cards display readiness, observed moves, owned skills, and development needs.

Students and unresolved professionals retain semantic plus skill-gap behavior. Requests with no matched owned skill preserve transition/semantic ordering and show an unavailable evidence state.

The KG viewer displays up to 120 strong sampled transition edges as dashed amber links and reports the full transition-edge count in `/api/status`.

## 12. Verification

Offline checks:

```powershell
python -m compileall -q src evaluation build_graph.py config.py career_kg_web.py
node --check public/chat.js
python -m unittest discover -s tests -v
```

Result: 20 tests passed. Covered cases include chunk boundaries, duplicate/ambiguous steps, malformed-row tolerance, gaps, self-transitions, denominators, support filtering, exact title linkage, held-out edge-construction rejection, legacy graph migration, idempotent typed edges, parallel edge preservation, held-out metric leakage rejection, weighted metrics, skill aliases, negation, goal exclusion, ONET alignment fallback, unavailable evidence, stable ranking, transition traversal/context, stale vector candidates, and JSON-safe path payloads.

Live configured-service checks:

- `/api/status`: HTTP 200, ready, 19,225 nodes, 224,711 edges, 18,907 transitions.
- `/api/graph-data`: HTTP 200, transition edges included in the sampled visualization.
- Professional full-context query: HTTP 200, eight path roles, five courses, current role resolved to ESCO `data analyst`, and only explicitly owned Python/SQL accepted.
- Partial-context query: HTTP 200, zero path roles, eight explore roles, and twenty explore skills.

## 13. Reproduction commands

```powershell
# Unit and syntax checks
python -m compileall -q src evaluation build_graph.py config.py career_kg_web.py
node --check public/chat.js
python -m unittest discover -s tests -v

# Enrich an existing aligned graph without changing embeddings
python build_graph.py --transitions-only

# Override defaults when required
python build_graph.py --transitions-only --transition-min-count 5 --transition-chunk-size 200000

# Held-out evaluation
python evaluation/evaluate_karrierewege.py

# Fixed-query ranking diagnostic (uses configured embedding/reranking services)
python evaluation/ranking_ablation.py

# Run the application
python career_kg_web.py
```

## 14. Production UI/frontend implementation (2026-08-15)

The approved frontend plan was implemented without changing routes, API contracts, recommendation logic, or data flow.

### Design and responsive system

- Reworked `templates/index.html` and `public/chat.css` into a cohesive evidence-led studio using deep space `#070A10`, graphite `#0E141E`, slate `#151E2B`, frost `#EAF1F8`, and cyan `#4CC2EA`.
- Preserved Outfit for interface copy and JetBrains Mono for runtime/evidence metadata.
- Added a bounded two-column desktop shell, compact tablet layout, and a single-column mobile flow with safe-area padding.
- Standardized panels, cards, forms, buttons, focus rings, disabled states, empty/loading/error states, and 44–48px interaction targets.
- Added reduced-motion, reduced-transparency, forced-color, long-content, and horizontal evidence-track handling.

### Frontend integrity and safety

- Rebuilt `public/chat.js` with DOM-based rendering for all structured API data.
- Added a strict allowlist sanitizer around optional Marked output; unsafe tags, attributes, and protocols are removed.
- Made Marked and GSAP optional enhancements so a CDN failure cannot block the page.
- Added one-request-at-a-time locking, `AbortController`, conversation versioning, retry UI, accessible live status, and New Chat cancellation.
- Preserved the response order: narrative, explore evidence, career path evidence, then course recommendations.

### Knowledge graph workspace

- Split graph presentation and interaction into `public/graph.css` and `public/graph.js` while keeping `/graph` and `/api/graph-data` unchanged.
- Added responsive header/legend behavior, accessible zoom/fit controls, progressive skill/orphan labels, and a persistent node inspector for mouse and touch.
- Added safe tooltip positioning plus explicit Cytoscape, HTTP, invalid-payload, and empty-data recovery states.

### Verification

```powershell
node --check public/chat.js
node --check public/graph.js
python -m compileall -q src evaluation build_graph.py config.py career_kg_web.py app.py
python -m unittest discover -s tests -v
```

Results:

- 20/20 offline tests passed.
- `/`, `/graph`, `/api/status`, and `/api/graph-data` returned HTTP 200 locally.
- Runtime status reported ready with 19,225 nodes, 224,711 edges, and 18,907 transition edges.
- Home and graph pages were rendered in Chromium at desktop, tablet, and narrow/mobile review sizes; the render artifacts remain local under `artifacts/ui-final/`.
- Blocked-CDN renders confirmed that chat remains usable without Marked/GSAP and the graph presents a clear retryable state without Cytoscape.
- `/api/chat` was intentionally not called during UI verification because it uses configured external model/reranking services.

### Repository and deployment boundary

- `HANDOFF.md` and `ALL_STEPS.md` are committed as project records and excluded through `.vercelignore`.
- `.agents/`, `.claude/`, raw data, generated artifacts, graph binaries, and the Chroma index remain available locally but excluded from GitHub/Vercel.
- The Vercel deployment is expected to use externally hosted graph/vector data before the data-backed endpoints can be production-ready.

## 15. Embedding-smoothed transition evaluation and integration (2026-08-15)

### Data decision

Karrierewege title mapping already had complete coverage: 1,284/1,284 training titles, 1,132/1,132 validation titles, and 1,121/1,121 test titles mapped exactly to ESCO. Sampled Karrierewege descriptions and skills were repeated ESCO role content. Re-embedding nearly two million rows would therefore duplicate vectors and overweight common occupations without adding coverage.

The accepted alternative uses existing `text-embedding-3-large` role vectors to find semantically related ESCO source roles, then pools only their training-derived transition distributions:

```text
hybrid destination score =
    direct_weight × direct transition probability
    + (1 - direct_weight) × similarity-weighted neighbour probability
```

No raw row, person identifier, validation transition, or test transition enters the scorer.

### Validation tuning and locked test

The validation-only grid evaluated 48 deterministic configurations:

- neighbours: 3, 5, 10, 20;
- direct weight: 0.50, 0.65, 0.80, 0.90;
- temperature: 0.05, 0.10, 0.20.

Validation selected 20 neighbours, direct weight 0.90, and temperature 0.05. The configuration was frozen before the test split was scored.

| Test metric | Direct baseline | Hybrid | Change |
|---|---:|---:|---:|
| MRR | 0.259056 | 0.261703 | +0.002647 |
| Hits@5 | 0.370239 | 0.372183 | +0.001944 |
| Hits@10 | 0.501359 | 0.505809 | +0.004450 |
| Source-role coverage | 0.738878 | 1.000000 | +0.261122 |
| Destination coverage | 0.871500 | 0.953196 | +0.081697 |

The pre-registered gate passed with no rejection reasons. Two identical prediction/metric passes were deterministic.

### Implementation

- `src/transition_embedding.py` contains training-edge extraction, safe vector loading, deterministic cosine neighbours, hybrid scoring, and the cached runtime scorer.
- `evaluation/transition_metrics.py` now exposes a shared prediction-map evaluator; the original direct baseline is exactly reproduced.
- Local `evaluation/evaluate_embedding_transitions.py` performs validation selection, locked test evaluation, acceptance checks, and an aggregate JSON report; it remains ignored with the raw-data research tooling under the deployment-minimal repository policy.
- `career_kg_web.py` initializes smoothing lazily on the first chat request. Errors are isolated from graph/Chroma readiness and fall back to direct transitions.
- `src/inference_pipeline.py` adds only the bounded top hybrid destinations before the existing reranker and skill-gap ranking.
- Direct moves retain observed count/probability. Backoff-only candidates use `SEMANTIC_TRANSITION_BACKOFF` context and `semantic_transition_backoff` UI evidence so inferred moves are never presented as observed.
- Runtime preloads 765 transition-source vectors rather than all 3,039 ESCO vectors and caches rankings by resolved role.

Command:

```powershell
python evaluation/evaluate_embedding_transitions.py
```

The report remains local at `artifacts/karrierewege/embedding_transition_evaluation.json`. It records zero embedding API calls and zero embedded raw Karrierewege rows.

Verification after conditional runtime integration:

- 28/28 offline unit tests passed;
- Python compilation and both frontend JavaScript syntax checks passed;
- importing the deployment runtime loaded neither pandas nor the raw Karrierewege preprocessor;
- a mocked `/api/chat` request exercised real graph/Chroma loading, lazy smoother initialization, and destination ranking without external model calls;
- `/`, `/graph`, `/api/status`, and `/api/graph-data` returned HTTP 200 after restart.

## 16. Known limitations and next steps

1. ChromaDB contains 4,055 records while the pruned graph contains 3,932 roles. The new retrieval-boundary filter removes the 123 stale IDs safely. A future deployment rebuild should recreate the collection cleanly.
2. Only 132 directed alignment edges connect ONET and ESCO, so many ONET roles cannot receive comparable ESCO gap evidence.
3. Accessibility is based on graph-listed essential requirements, not proficiency depth, recency, or transferable-skill similarity.
4. Karrierewege transitions are observational population frequencies and may contain geographic, sectoral, temporal, or sampling bias.
5. Minimum support five was selected from training statistics. Validation/test were not used to construct graph edges.
6. The four-case ranking diagnostic demonstrates mechanics, not statistically general recommendation superiority.
7. Coursera integration remains an HTML scraper and may change independently.
8. A larger expert/user evaluation should assess relevance, trust, usefulness, and fairness.
9. Local `.agents/`, `.claude/`, `CODE/`, and `static/` resources are ignored rather than deleted; the repository root and `public/` are authoritative for GitHub/deployment.
10. Graph binaries and the Chroma index are intentionally absent from GitHub. External graph/vector storage must be connected for a data-backed Vercel deployment.

The pre-enrichment graph backup can be restored manually if needed; no automatic rollback command was added because broad destructive file operations should remain explicit.

---

## 17. Session Changes (2026-08-23) — Transition Effort Score + KG Link Prediction

### Supervisor feedback

The supervisor requested:
1. A new algorithm that measures the **effort/difficulty** needed to switch between roles (e.g., analyst -> AI engineer)
2. **KG edge prediction** (link prediction) to predict missing transitions and improve Hit ratios
3. General **KG improvements** to support better transition evaluation

### New research contributions implemented

#### Contribution D: Transition Effort Score (TES)

**File:** `src/transition_effort.py` | **Tests:** `tests/test_transition_effort.py` (25 tests)

Multi-factor composite score: `TES = w1*SkillGapMagnitude + w2*DomainDistance + w3*(1-EmpiricalSupport) + w4*(1-Transferability)`

Components:
- **SkillGapMagnitude**: IDF-weighted fraction of missing skills (rare skills cost more)
- **DomainDistance**: ISCO 2-digit Jaccard distance (cross-domain transitions are harder)
- **EmpiricalSupport**: Evidence people actually make this transition (from TRANSITIONS_TO edges + smoother)
- **Transferability**: IDF-weighted shared skills between source and target roles

Default weights: `w1=0.35, w2=0.15, w3=0.25, w4=0.25`
Effort bands: 0-0.3 = Low, 0.3-0.6 = Moderate, 0.6-1.0 = High

Integrated into inference pipeline: effort badges displayed on career-path role cards.

#### Contribution E: KG Link Prediction

**File:** `src/link_prediction.py` | **Tests:** `tests/test_link_prediction.py` (12 tests) | **Evaluation:** `evaluation/evaluate_link_prediction.py`

9 graph-structural features per (source, target) role pair:
1. Common Neighbours (shared skills count)
2. Jaccard Coefficient (normalized skill overlap)
3. Adamic-Adar Index (inverse-log-degree weighted)
4. Resource Allocation Index (inverse-degree weighted)
5. Preferential Attachment (degree product)
6. Embedding Cosine Similarity
7. Same ISCO Group (binary domain proximity)
8. IDF-weighted Skill Overlap
9. Neighbour Transition Evidence

Model: LightGBM binary classifier, trained on 18,907 positive + 94,535 negative samples.
5-fold CV: Mean AUC = 0.9344, Mean AP = 0.8186.
Top features: Neighbour Evidence, Embedding Cosine Similarity, Resource Allocation.

#### Contribution P2: Graph-Provenance Faithfulness Verification

**File:** `src/faithfulness.py` | **Tests:** `tests/test_faithfulness.py` (19 tests after the 2026-08-27 repair)

Post-generation verification: extracts bolded entities from LLM output, fuzzy-matches to graph nodes, checks BFS reachability from anchor roles within 3 hops.
Faithfulness score = |matched & reachable| / |total entities mentioned|.

#### Contribution P4: Provenance-Traced Explanation Chains

**File:** `src/explainability.py` | **Tests:** `tests/test_explainability.py` (12 tests after the 2026-08-27 repair)

Traces each recommendation back through typed edges: TRANSITIONS_TO, SIMILAR_TO, REQUIRES.
Produces human-readable chains showing why each role was recommended.

### KG Enrichment

**File:** `src/kg_enrichment.py` | **Tests:** `tests/test_kg_enrichment.py` (11 tests)

- Computed IDF for 13,570 skills/elements (stored as node attribute)
- Computed ISCO 2-digit codes for 3,039 ESCO roles
- Mapped 893/893 ONET roles to ISCO-08 via SOC->ISCO crosswalk
- Crosswalk stored at `Data/crosswalks/soc_isco08_crosswalk.csv`
- Graph enriched and saved (no API calls needed)

### New dependencies

- `lightgbm>=4.0` — offline link-prediction training (`requirements-research.txt`)
- `scikit-learn>=1.5` — offline cross-validation and metrics (`requirements-research.txt`)

### Test suite

Historical checkpoint: **100 tests passing** (28 original + 72 new). The current 2026-08-27 total is 124.
- `tests/test_kg_enrichment.py` — 11 tests
- `tests/test_transition_effort.py` — 25 tests
- `tests/test_link_prediction.py` — 12 tests
- `tests/test_faithfulness.py` — 15 tests
- `tests/test_explainability.py` — 9 tests

### Configuration additions (`config.py`)

```
EFFORT_WEIGHT_SKILL_GAP = 0.35
EFFORT_WEIGHT_DOMAIN = 0.15
EFFORT_WEIGHT_EMPIRICAL = 0.25
EFFORT_WEIGHT_TRANSFERABILITY = 0.25
LINK_PREDICTION_ENABLED = true
LINK_PREDICTION_MODEL_PATH = artifacts/link_prediction/link_predictor.json
```

### New CLI flags (`build_graph.py`)

```bash
python build_graph.py --enrich               # Add IDF + ISCO enrichment attributes
python build_graph.py --train-link-predictor  # Train LP model from existing graph
```

### Frontend changes

- Effort badges on career-path role cards (Low/Moderate/High with green/amber/red styling)
- `.path-effort-badge`, `.effort-low`, `.effort-moderate`, `.effort-high` CSS classes

### Documentation

- `docs/ARCHITECTURE.md` — current vs target architecture
- `docs/RESEARCH.md` — research decisions and trade-offs

### Stage 8: LangGraph Agent Orchestration

- Created `agents/` package with `StateGraph` from LangGraph
- 10 nodes: intent, retrieval, ranking, traversal, effort, generation, faithfulness, explanation, courses, explore
- Conditional routing: `has_context=True` → full pipeline, `has_context=False` → explore panels
- Feature-flagged via `USE_LANGGRAPH=true` in `.env` (default: off, uses existing `run_query()`)
- Created `agents/tools.py` with tool definitions: `search_career_graph`, `find_role_skills`, `find_related_roles`, `calculate_skill_gap`, `calculate_effort`, `get_transition_evidence`

### Stage 9: Hybrid Retrieval + Combined Predictions

- Created `src/hybrid_retrieval.py` — vector, transition, role-relevant graph overlap, and virtual LP retrieval.
- The validation-accepted smoothed ranking is the primary transition signal. LP is a labeled missing-edge coverage backfill and does not receive an equal RRF vote.
- Current held-out test result: **coverage 73.9% → 100%**, Hits@5 `0.3722`, Hits@10 `0.5058`, and MRR `0.2617`.

### Stage 10: Evaluation Framework

- Created `evaluation/benchmark_dataset.py` — 29 test cases across 10 categories
- Created `evaluation/evaluate_retrieval_strategies.py` — 4-method comparison (direct, smoothed, LP, combined)
- Report saved: `artifacts/retrieval_comparison/comparison_report.json`

### Stage 11: Full Pipeline Ablation (P3)

- Created `evaluation/pipeline_ablation.py` — 7-variant ablation
- Results prove Contribution A (TRANSITIONS_TO) is foundational: removing it drops ALL metrics to zero
- The historical direct > LP result below is superseded by the 2026-08-27 validation-selected smoothing result in Section 17.

**Test ablation results (held-out test split):**

| Variant | Hits@5 | Hits@10 | MRR | Coverage |
|---------|--------|---------|-----|----------|
| A: Full pipeline | 0.3702 | 0.5014 | 0.2591 | 0.7389 |
| C: No graph traversal | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E: No TRANSITIONS_TO | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| + Combined (direct>LP) | 0.3704 | 0.5017 | 0.2580 | **1.0000** |

### Verification

```bash
# All 124 tests pass after the 2026-08-27 repair
python -m unittest discover tests -v

# Server runs with all features (effort + explanations + faithfulness)
python career_kg_web.py
# → http://127.0.0.1:8001

# Enable LangGraph orchestration (optional)
# Set USE_LANGGRAPH=true in .env

# Enrich graph (no API calls)
python build_graph.py --enrich

# Train link predictor (~2 minutes)
python build_graph.py --train-link-predictor

# Run pipeline ablation (~5 minutes, uses LP model)
python evaluation/pipeline_ablation.py

# Run retrieval strategy comparison
python evaluation/evaluate_retrieval_strategies.py

# Save benchmark dataset
python evaluation/benchmark_dataset.py
```

### File map additions

```
agents/
  __init__.py              Package init
  state.py                 CareerAgentState TypedDict
  graph.py                 LangGraph StateGraph + run_career_workflow()
  tools.py                 Graph query tools (6 functions)
  nodes/
    __init__.py            Node package
    intent.py              Intent routing node
    retrieval.py           Vector + transition retrieval node
    ranking.py             Cohere rerank + gap ranking node
    traversal.py           Graph traversal node
    effort.py              TES computation node
    generation.py          LLM generation node
    courses.py             Coursera search node
    faithfulness_node.py   Post-generation verification node
    explanation.py         Evidence chain tracing node

src/hybrid_retrieval.py    Multi-source fusion and virtual LP edge backfill
evaluation/
  pipeline_ablation.py     7-variant full pipeline ablation (P3)
  evaluate_retrieval_strategies.py   4-method retrieval comparison
  benchmark_dataset.py     29 test cases / 10 categories
  benchmark_cases.json     Serialized dataset
artifacts/
  link_prediction/
    link_predictor.pkl     Offline trained LightGBM model
    link_predictor.json    Portable runtime export for Vercel
    evaluation_report.json LP metrics
  retrieval_comparison/
    comparison_report.json 4-method Hits@K comparison
  pipeline_ablation/
    ablation_report.json   7-variant ablation results
```

## 17. PR review repair, role-aware effort, and live missing-edge inference (2026-08-27)

### Scope and storage boundary

The GitHub Copilot review was checked against `HANDOFF.md`, the local graph, the Chroma index, and the executable request paths. All correctness/accessibility findings were implemented locally except the deployment-data finding, which is intentionally deferred: `graph/`, `index/`, and `artifacts/` remain ignored because production will use external storage. No graph, Chroma, or model binary was added to Git, and no predicted edge was persisted into the KG.

The `public/**` Vercel warning did not require a code change in this local-first pass. Flask continues to use `public` as its static directory, while deployment packaging will be validated when the external stores are designed.

The approved implementation design is recorded in `docs/superpowers/specs/2026-08-27-local-graphrag-repair-design.md`.

### Training-only transition contract

Added `src/transition_policy.py` with one `is_training_transition()` predicate. A transition is runtime/training evidence only when:

- `relation == "TRANSITIONS_TO"`;
- `source == "karrierewege"`; and
- `split` is absent or `"train"`.

The predicate is now used by transition smoothing, transition candidates, skill-gap evidence, TES empirical support, classic traversal, agent tools/traversal, explainability, LP training/index construction, transition metrics, graph visualization data, and runtime status. Held-out edges are skipped rather than disabling the smoother.

### Role-relevant readiness and effort

Readiness, graph-overlap retrieval, and TES now compare skills through the same role-requirement policy:

1. ESCO roles use essential requirements only.
2. Aligned ONET roles use the strongest aligned ESCO role's essential requirements.
3. Unaligned ONET roles retain only requirements at or above `ONET_IMPORTANCE_THRESHOLD` (default `3.5`).
4. Accessibility and TES skill/transferability components use stored IDF weights, while UI counts remain raw and explainable.

This prevents optional generic English/comprehension requirements from dominating technical transitions. It does not globally suppress language: essential English still contributes for an English teacher or another language-centered role. Negation handling also recognizes normalized contractions and explicit `no`, so statements such as “I don't use Python” and “I have no Python experience” do not create false ownership.

The student and professional generation prompts now require exact graph labels, bold every named role/skill, prioritize occupation-specific gaps, treat generic language/comprehension as primary only when central or no stronger supported gap exists, and call a skill a strength only when the user explicitly claimed it.

### Shared hybrid runtime and predicted missing edges

Both `run_query()` and the LangGraph retrieval node now call `hybrid_retrieve()`.

The request-time candidate flow is:

1. semantic Chroma retrieval with the requested vector limit;
2. direct and embedding-smoothed training transitions;
3. IDF-weighted role-relevant skill overlap;
4. reciprocal-rank fusion of those accepted sources; and
5. LightGBM missing-edge candidates as a labeled coverage backfill.

`career_kg_web.py` lazily and once-per-process loads a complete `LinkPredictionRuntime`: saved model, all 3,039 normalized ESCO embeddings, training transition index, semantic-neighbour evidence, IDF, skill/degrees, and ISCO caches. Prediction uses all nine training-time feature families, excludes the source role and existing observed destinations, and is protected by a prediction lock. The result is surfaced as `PREDICTED_TRANSITION` with a model score, never as `TRANSITIONS_TO`.

LP does not receive an equal rank vote. The held-out promotion gate shows the accepted embedding smoother is materially stronger for top-K ranking, so predicted links remain available for missing-edge coverage without being allowed to inflate recommendations merely because their classifier score is high.

### Corrected held-out comparison

`evaluation/evaluate_retrieval_strategies.py` was repaired to use the current `hybrid_prediction_map()` signature, compute neighbours for evaluated source roles against training-transition sources, preserve full empirical/smoothed rankings for MRR, and label the combination as priority fallback rather than RRF.

| Test metric | Direct | Accepted smoothed / combined | Delta |
|---|---:|---:|---:|
| Hits@5 | 0.370239 | 0.372183 | +0.001944 |
| Hits@10 | 0.501359 | 0.505809 | +0.004450 |
| MRR | 0.259056 | 0.261703 | +0.002647 |
| Source-role coverage | 0.738878 | 1.000000 | +0.261122 |

Standalone LP test ranking measured Hits@5 `0.0274`, Hits@10 `0.0447`, MRR `0.0182`, coverage `1.0000`. It therefore failed top-K promotion even though the classifier's training CV remains AUC `0.9344` and AP `0.8186`. This distinction is now explicit: classification quality is not treated as proof of held-out ranking superiority.

### LangGraph, concurrency, and UI correctness

- Partial context now routes `retrieval_partial → ranking → traversal → explore → END`.
- Graph/Chroma initialization, smoothing initialization, LP initialization, and model prediction are process-lock protected.
- Graph fetch retries retain a per-request controller identity, preventing an aborted request from clearing/rendering over the current request.
- Skip-link destinations use `tabindex="-1"`.
- The graph provides an accessible role/skill selector that drives the same inspector as pointer selection; the canvas is no longer presented as keyboard-operable by itself.
- Explanation output uses shared `.message`/`.evidence-panel` styling and a defined graph icon.
- Predicted and semantic transitions have explicit inferred UI copy and explanation relations.
- Faithfulness extraction covers unbolded cue-based candidates, slash-separated technologies, and safe short aliases for parenthetical ESCO titles while still penalizing unsupported entities.

### Dependency and local verification record

`requirements.txt` now uses `langgraph>=0.3,<1.0` and `langchain-core>=0.3.85,<1.0`, resolving the local `langchain 0.3.x` incompatibility introduced by an unconstrained LangGraph install.

Verified commands and results:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m compileall -q src agents evaluation career_kg_web.py config.py
node --check public/chat.js
node --check public/graph.js
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -u -m evaluation.evaluate_retrieval_strategies
```

Results:

- `pip check`: no broken requirements;
- 124/124 unit tests passed;
- graph: 19,225 nodes and 224,711 edges;
- Chroma: 4,055 records;
- smoother: 765 training-transition source vectors loaded;
- LP context: 3,039/3,039 ESCO vectors loaded and predictions produced locally;
- `/`, `/graph`, `/api/status`, `/api/graph-data`: HTTP 200;
- classic full-context `/api/chat`: HTTP 200, current role excluded from recommendations;
- LangGraph partial-context `/api/chat`: HTTP 200, eight explore roles, zero path roles, zero courses.

The remaining deployment task is architectural rather than a local bug fix: provide external graph, vector, and model storage, then validate the actual Vercel bundle and production endpoints.

---

## 18. Session Changes (2026-08-30) — React Frontend + Agent Pipeline Enhancements + LP Improvements

### Frontend migration: React 19 + shadcn/ui + Tailwind CSS v4

The vanilla JS/CSS frontend (`templates/index.html`, `public/chat.js`, `public/chat.css`) is superseded by a Vite-built React SPA in `frontend/`. Flask detects `public_react/` (the built output) at startup and falls back to `public/` if it is absent.

**New components:**
- `ChatShell` — main two-column layout (hero/status left, chat right)
- `RoleCard` — role card with large effort %, qualification bar, upskill time, "Why recommended?" collapsible
- `EffortBadge` — effort band + % + upskill time label
- `PipelineIndicator` — 5-stage animated pipeline loader (Intent → Retrieve → Rank → Effort → Generate)
- `FaithfulnessBadge` — Graph-verified % badge (fixed from NaN — see below)
- `EvidencePanel` — three-tier entity breakdown (verified / unreachable / unmatched) + provenance chains
- `SkillGapCard` — per-role priority skills, quick wins, blockers
- `LearningRoadmap` — phased learning timeline with course cards
- `StatsTiles` — 4-tile stats row: skill match %, upskill time, observed transitions, effort %
- `TransferableSkills` — skills appearing across ≥2 recommended roles with count badges
- `CourseCard` — fully clickable card (whole card = `<a href>`) with workload + type badges
- `StatusPanel` — server status, graph node/edge counts, tech badges
- `YourPathPanel` — tabbed panel combining SkillGapCard + LearningRoadmap

**UX changes:**
- Roles displayed in 2-column vertical grid sorted low→high effort (not horizontal scroll)
- Effort legend above role grid: "● 3 low · 2 moderate · 1 high"
- All roles shown (no arbitrary cap)
- Auto-scroll respects user scroll position (no drag-to-bottom on panel append)

### Bug fixes

1. **Faithfulness NaN%** — `FaithfulnessResult.to_dict()` emitted `faithfulness_score`; frontend read `.score`. Fixed: key renamed to `"score"` in `src/faithfulness.py`. Added `"matched_entities"` field (`[m.text for m in self.matches if m.reachable]`).

2. **Course links never rendered** — `coursera_client.py` returns `canonical_url`; frontend expected `url`. Fixed in `fetch_coursera_courses()` (`src/inference_pipeline.py`) with normalization: `canonical_url → url`, `partner_names[0] → provider`, `estimated_workload` and `content_type` passed through.

3. **Effort badges never showed for students** — `effort_node` gated scoring on `current_role_id is not None`, which is always `None` for students. Fixed: fallback to `path_data["roles"][0]` as synthetic source. Tag `effort_source: "relative"` on fallback scores.

### New LangGraph agent nodes (12-node pipeline)

| Node | Position | What it adds |
|---|---|---|
| `qualification_node` | ranking → traversal | IDF-weighted % of essential skills owned per candidate |
| `skill_gap_node` | effort → courses | Ranked missing skills by TES reduction impact, quick wins, blockers |
| `learning_plan_node` | courses → generation | Phased upskill roadmap from skill gaps + Coursera output |

Updated pipeline graph edges:
```
ranking → qualification → traversal → effort → skill_gap → courses → learning_plan → generation → faithfulness → explanation → END
```

API response now includes `skill_gap_analysis` and `learning_plan` fields.

### Transition Effort Score improvements

- `EffortResult` now includes `estimated_weeks_min` and `estimated_weeks_max` (based on missing skill count × 5 weeks/skill, fallback to effort band map)
- Both `effort_node` and `run_query()` re-sort `path_data["roles"]` by `effort_score` ascending after annotation

### Link Prediction improvements (11 features, was 9)

New features added to `src/link_prediction.py`:
1. **`isco_group_distance`** — replaces binary `same_isco_group`; values: 0.0 (same), 0.5 (adjacent ≤5 numeric diff), 1.0 (different)
2. **`title_tfidf_sim`** — TF-IDF bigram cosine similarity between role title strings (sklearn, precomputed at training time)
3. **`meta_path_count`** — log2(1 + count of roles sharing ≥3 skills with both source and target)

Hard negative sampling: 50% of training negatives are now semantic near-neighbours (top-40 cosine similarity with no training transition). `build_neighbour_index` uses k=40 (was k=10).

Re-train command: `python build_graph.py --train-link-predictor`

### ISCO group edges

New file: `src/isco_edges.py`
- `add_isco_group_edges(G)` — bidirectional `SAME_ISCO_GROUP` edges between roles sharing a 2-digit ISCO code, capped at top-20 per group by degree
- `isco_group_stats(G)` — diagnostic `{isco_2digit: member_count}` mapping
- New CLI flag: `python build_graph.py --isco-edges`

### ONET ISCO coverage (confirmed)

`src/kg_enrichment.add_onet_isco_codes()` already writes `isco_2digit` to 893/893 ONET nodes via `Data/crosswalks/soc_isco08_crosswalk.csv`. `domain_distance()` reads it correctly. No changes needed.

### Test record

Test suite: **124 passed** (no regressions). Updated tests:
- `tests/test_faithfulness.py` — `test_to_dict` now checks `"score"` and `"matched_entities"` keys
- `tests/test_link_prediction.py` — `test_isco_group_distance` replaces `test_same_isco`; expects 0.0/0.5/1.0 instead of 1.0/0.0
- `tests/test_hybrid_retrieval.py` — feature shape assertion updated to `len(FEATURE_NAMES)` = 11

### New CLI flags

```bash
python build_graph.py --isco-edges         # Add SAME_ISCO_GROUP edges
python build_graph.py --train-link-predictor  # Retrain LP with 11 features
```

### File additions

```
src/isco_edges.py               SAME_ISCO_GROUP edge builder
agents/nodes/qualification_node.py  IDF-weighted qualification scoring
agents/nodes/skill_gap_node.py      Per-role ranked skill gap analysis
agents/nodes/learning_plan_node.py  Phased learning roadmap
frontend/                       React 19 + Vite + shadcn/ui + Tailwind v4
  src/components/
    ChatShell.tsx, RoleCard.tsx, StatsTiles.tsx, TransferableSkills.tsx
    SkillGapCard.tsx, LearningRoadmap.tsx, EvidencePanel.tsx
    FaithfulnessBadge.tsx, CourseCard.tsx, PipelineIndicator.tsx, StatusPanel.tsx
  src/api.ts, src/types.ts
  vite.config.ts, tailwind.config reference in index.css
  components.json (shadcn registry config)
```

### Verification

```bash
PYTHONPATH=. .venv/Scripts/pytest tests/ -q
# → 124 passed

cd frontend && npm run build
# → ✓ built in ~500ms, no errors

curl http://127.0.0.1:8001/api/status
# → {"ready": true, "graph_loaded": true, ...}

curl -s http://127.0.0.1:8001/ | head -3
# → <!doctype html> ... Career Graph Studio (React SPA)
```

---

## 2026-09-05: Transition ranking, JobHop, and accepted smoother refinement

- Audited the KG without destructive pruning: no isolates, invalid endpoints, invalid transition probabilities, missing role requirements, or held-out transition edges were found.
- Downloaded and checksum-verified JobHop v2 locally under ignored `Data/JobHop_v2/` (CC BY 4.0), then cleaned and mapped it to the live ESCO graph.
- JobHop train quality result: 1,391,276 accepted experiences, 858,913 valid non-self transitions, 295,825 distinct pairs, and zero person overlap across official splits.
- Added a 5-fold person-disjoint LambdaMART experiment. Candidate generation uses only other-fold transition evidence and never injects missing true targets. Features cover direct support, semantic smoothing, 2/3-hop walks, embeddings, ISCO, skills, popularity, provenance, and JobHop-train priors.
- JobHop raised fold candidate recall from roughly 83.7% to 84.2–84.5%. The ranker improved tail metrics (standalone test Hits@10 0.509095) but regressed MRR/Hits@5, so the promotion gate rejected it and runtime behavior was not changed.
- Added observation-weighted and macro NDCG@K to the shared evaluator.
- Ran a 63-configuration fine search around the accepted smoother. Validation selected `neighbours=25`, `direct_weight=0.91`, `temperature=0.06`.
- Frozen test result: MRR 0.261805, Hits@5 0.372273, Hits@10 0.505849, destination coverage 0.958322. All improve on the prior accepted smoother (0.261703 / 0.372183 / 0.505809 / 0.953196).
- Updated runtime and `.env.example` defaults to the accepted fine-tuned configuration. Data, graph, vector, fold-cache, model, and evaluation artifacts remain local-only.

---

## 2026-09-05: Neo4j AuraDB and Qdrant Cloud migration

### Decision and reuse boundary

The project now uses Neo4j AuraDB as the authoritative knowledge graph and Qdrant Cloud as the authoritative vector store. Existing ONET/ESCO preprocessing, Karrierewege train-only transition aggregation, role-text construction, Azure `text-embedding-3-large` model, transition smoothing, hybrid retrieval, and NetworkX-based recommendation algorithms were retained. The storage boundary changed; the recommendation pipeline was not unnecessarily redesigned.

JobHop v2 remains a useful research/long-tail dataset but was not inserted into the production graph. Its person-disjoint LambdaMART experiment improved candidate recall and Hits@10 but regressed the primary MRR/Hits@5 promotion gate, so adding its noisy transition prior to the permanent KG would not be justified.

### Reusable cloud modules

- `src/neo4j_store.py` centralizes secret-safe Aura driver creation, schema setup, batched node/relationship writes, count and representative-query validation, and reconstruction into the existing `NetworkX MultiDiGraph` interface.
- `src/qdrant_store.py` centralizes Qdrant connection handling, deterministic point IDs, role payload construction, vector preparation, cosine collection creation, keyword payload indexes, batch upload, count/ID/search validation, and the existing `query`/`get` compatibility surface.
- `src/graph_quality.py` enforces allowed node and relationship types, normalized titles, finite weights/probabilities, no self-loops, no held-out transition leakage, deterministic edge deduplication, isolate removal, and source-backed qualification/job-zone connectivity.
- `rebuild_databases.py` provides one end-to-end command and a non-mutating `--dry-run`. Cloud mutation starts only after local preparation and held-out evaluation pass and both cloud connections are verified.

### Rebuild flow

```text
ONET + ESCO source preprocessing
→ prune thin roles and isolates
→ enrich skill IDF and ISCO attributes
→ aggregate/add Karrierewege train-only transitions
→ normalize, validate, deduplicate, and improve connectivity
→ generate one role vector per live role with the existing model
→ compute cross-taxonomy alignment from the same vectors
→ evaluate validation/test splits
→ verify both cloud connections
→ rebuild and validate Neo4j Aura
→ recreate, index, batch-upload, and validate Qdrant
```

Prepared vectors are cached only in ignored local artifacts and are reusable only when the graph ID, embedding model, ordered role IDs, dimensions, and finite values all match. The existing Azure embedding client now retries transient HTTP 408/429/5xx and connection failures with bounded exponential backoff; invalid requests still fail immediately.

### KG quality result

The final graph contains 19,241 nodes and 240,906 relationships:

| Node type | Count |
|---|---:|
| Role | 3,932 |
| Skill | 13,939 |
| Element | 95 |
| ISCO group | 619 |
| Skill group | 640 |
| Qualification | 12 |
| Job zone | 4 |

| Relationship type | Count |
|---|---:|
| REQUIRES | 155,490 |
| TRANSITIONS_TO | 18,907 |
| SAME_ISCO_GROUP | 12,634 |
| SIMILAR_TO | 132 |
| BROADER_THAN | 24,467 |
| NARROWER_THAN | 24,467 |
| BELONGS_TO | 3,039 |
| TYPICALLY_REQUIRES_QUALIFICATION | 877 |
| IN_JOB_ZONE | 893 |

The improvement pass added 12 normalized qualification nodes, 4 job-zone nodes, and 1,770 source-backed connectivity relationships. No course or industry nodes were fabricated because the repository has no stable source-backed course/industry dataset. The Coursera integration remains query-time retrieval.

### Link and next-node prediction

Observed training transitions and inferred candidates are ranked together in hybrid retrieval, but inferred links remain request-local. Common-neighbour, Jaccard, Adamic-Adar, resource-allocation, skill overlap, embeddings, ISCO distance, and neighbouring-transition evidence are available to the LP method. The bundled LightGBM artifact declares the original 9-feature schema; runtime now passes exactly that declared prefix rather than disabling LightGBM shape validation. The current 11-feature extractor remains available for a future train-only retraining experiment.

No predicted relationship is written to Aura as `TRANSITIONS_TO`. Prediction responses explicitly use `evidence_type=predicted_transition` and `persisted=false`.

### Held-out recommendation metrics

The final configuration was frozen before the test split was evaluated:

| Method | Hits@1 | Hits@3 | Hits@5 | Hits@10 | MRR | Source coverage | Destination coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct train edges | 0.147830 | 0.284979 | 0.370239 | 0.501359 | 0.259056 | 0.738878 | 0.871500 |
| Accepted smoother on cloud-ready build | 0.148318 | 0.285743 | 0.372281 | 0.505858 | 0.261803 | 1.000000 | 0.958143 |
| Previous accepted local smoother | — | — | 0.372273 | 0.505849 | 0.261805 | 1.000000 | 0.958322 |

The rebuilt graph/vector result matches the accepted local method within roughly two millionths MRR while slightly improving Hits@5 and Hits@10 by about 0.000008. This is effectively parity, with the key improvement being clean cloud storage, exact vector coverage, richer useful connectivity, and elimination of the 123 stale Chroma records.

### Live cloud validation

- Neo4j Aura: 19,241 nodes and 240,906 relationships.
- Representative role-to-skill and high-support transition Cypher queries returned results.
- Aura-to-NetworkX reconstruction returned exactly 19,241 nodes and 240,906 relationships.
- Qdrant: 3,932 vectors, 3,072 dimensions, cosine distance.
- Deterministic ID retrieval and a five-result similarity search succeeded.
- Qdrant requires a keyword payload index for filtered search; the rebuild now creates indexes for `graph_id`, `record_id`, `source`, and `entity_type` before batch upload.
- Redacted reports are written to ignored `artifacts/cloud_rebuild/`; no credential or endpoint values are stored in them.

### Runtime and Vercel deployment

`career_kg_web.py` now lazy-loads the graph from Aura and vectors from Qdrant. The Qdrant adapter preserves the retrieval interface used by the classic and LangGraph pipelines, so query text is embedded with the same model and searched remotely without changing downstream ranking semantics.

`vercel.json` packages `templates/`, the committed `public_react/` production build, and the small portable LP model. It excludes raw datasets, local graph/index directories, tests, evaluation outputs, and agent tooling. ChromaDB was removed from runtime dependencies. The Python function duration is set to 300 seconds for cloud-backed cold starts and model calls.

The GitHub `dev` branch is connected and deployed on Vercel. Required model/cloud values are supplied through Vercel Project Settings and `.env` remains uncommitted. The database rebuild is an offline administration command and must not be run during a Vercel build or request.

### Final verification record

- `python -m unittest discover -s tests -q`: 150 tests passed after the cloud-training automation.
- Python compilation, `node --check` for legacy browser scripts, and the React TypeScript/Vite production build passed.
- `python -m pip check`: no broken requirements.
- A real cloud-backed `/api/status` cold request returned HTTP 200 with Aura, Qdrant, transition smoothing, and link prediction loaded; reported counts were 19,241 nodes and 240,906 relationships.
- A natural-language query for `machine learning engineer` was embedded with the configured model and returned relevant Qdrant matches led by artificial intelligence engineer, data engineer, and data scientist.
- `vercel.json` parses successfully. The earlier local CLI dry-run check was account-blocked, but deployment was subsequently completed through Vercel using the GitHub `dev` branch.

### Deployment completion

- Vercel deployment: complete from the repository's `dev` branch.
- Neo4j AuraDB connection: complete and validated with 19,241 nodes and 240,906 relationships.
- Qdrant Cloud connection: complete and validated with 3,932 vectors at 3,072 dimensions.
- Runtime database loading: validated with Aura, Qdrant, transition smoothing, and link prediction all reporting ready.
- Secrets remain outside Git and are supplied through Vercel environment variables.

### Post-deployment response-rendering and LP portability fix

1. Reproduced the deployed black screen and verified that AuraDB, Qdrant, and `/api/chat` remained healthy.
2. Identified the frontend contract mismatch: `path.roles[*].have/need` contained `{id, title}` skill objects, but the React components treated them as strings.
3. Added API-boundary normalization for mixed skill shapes and backend explanation aliases, plus an application error boundary with a visible reload action.
4. Exported the existing LightGBM Booster to JSON and added dependency-free NumPy inference for Vercel. A configured legacy `.pkl` path falls back automatically to the sibling JSON artifact.
5. Moved LightGBM and scikit-learn from runtime requirements to research requirements; the trained model and ranking behavior remain enabled.
6. A cloud-backed Flask status smoke test with LightGBM imports deliberately blocked returned HTTP 200 with AuraDB, Qdrant, transition smoothing, and portable link prediction loaded.

### Fully online link-prediction retraining

1. Added `train_link_prediction_cloud.py`, which reconstructs the production graph from AuraDB and loads all live ESCO vectors from Qdrant without local datasets or embedding calls.
2. Added source-role-disjoint GroupKFold validation. Held-out source transitions are removed from fold-specific neighbour-evidence features.
3. Added classification and ranking promotion gates: AUC >= 0.87, AP >= 0.68, Hits@5 >= 0.89, and MRR >= 0.78.
4. The production-data smoke run trained on 113,442 sampled pairs and passed with AUC 0.883384, AP 0.702746, Hits@5 0.912418, and MRR 0.801180. These internal sampled-negative metrics are not directly comparable to the official Karrierewege test split.
5. Added `.github/workflows/train-link-prediction.yml`. A manual run installs training-only dependencies on Ubuntu, tests, trains, uploads a 7-day artifact, and optionally opens a validated model PR against `dev`.
6. GitHub Pages is not used. Vercel remains the React/Flask deployment, while GitHub Actions supplies ephemeral training compute.
7. Configured the six cloud credentials as GitHub Actions secrets, set `QDRANT_COLLECTION=career_roles`, and enabled Actions read/write plus pull-request creation. Secret values were not printed or committed. The workflow still needs to be merged into the default `main` branch before its first manual run.

### GitHub/Vercel repository cleanup and delivery

1. Reduced the manual trainer timeout from 180 to 30 minutes and artifact retention from 30 to 7 days, avoiding scheduled runs and unnecessary Student/Pro Actions usage.
2. Removed the obsolete LightGBM pickle, legacy vanilla chat page, unused Vite starter assets, unused UI scaffolding, and unused Radix/Cytoscape packages.
3. Kept the `/graph` viewer and moved its CSS/JS to `frontend/public/`; the Vite build copies both into `public_react/`, which is now Flask's only static root.
4. Removed generated local graph/index stores, cloud preparation/smoke caches, and the accidental workspace `~/` directory after verifying every target was inside the repository workspace.
5. Preserved ignored `Data/` because it is the only complete source input for `python rebuild_databases.py`; it remains excluded from Git and Vercel.
6. Rebuilt the production frontend and verified the resulting `public_react/` bundle contains the SPA and graph-viewer assets.
7. Pushed cleanup commit `b58de79` to `dev`; Vercel preview passed, then PR #7 merged the workflow and cleanup into default `main`.
8. Triggered GitHub Actions run `33975475939` with `target_branch=dev`. The standard Ubuntu job completed successfully in 2m57s, including focused tests, Aura/Qdrant loading, grouped validation, model export, and artifact upload.
9. The cloud run produced AUC `0.883507`, AP `0.702902`, Hits@1 `0.713725`, Hits@3 `0.853595`, Hits@5 `0.911111`, Hits@10 `0.949020`, and MRR `0.797201`; every configured publication gate passed.
10. The workflow opened PR #8 containing only the validated portable model. Its Vercel preview passed and the PR was merged into `dev` as `32ce418`.
