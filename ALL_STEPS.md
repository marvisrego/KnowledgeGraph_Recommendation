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

## 15. Known limitations and next steps

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
