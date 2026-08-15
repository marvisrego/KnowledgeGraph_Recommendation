# Novelty Contributions — GraphRAG Career Advisor Thesis

## Purpose
This document records the candidate contributions and their implementation status.
Contributions #1 and #2 were implemented together on 2026-08-15.

## Implementation Status (2026-08-15)

- **#1 implemented:** 18,907 train-only `TRANSITIONS_TO` edges at minimum support five, retaining 88.6% of valid non-self training transition observations.
- **#2 implemented:** explicit owned-skill matching, ESCO-comparable gap scores, ONET alignment fallback, transition-aware candidate expansion, and accessible-to-aspirational ordering.
- **Held-out results:** validation Hits@5 0.3716/MRR 0.2586; test Hits@5 0.3702/MRR 0.2591.
- **Verified dataset:** 2,480,369 career-step rows across 568,888 split-disjoint trajectories. All 1,295 occupation labels map exactly to ESCO after normalization.
- Full reproducibility and limitations are recorded in `ALL_STEPS.md`.

---

## Summary Table

| # | Contribution | Originality | Feasibility | Best For |
|---|---|:---:|:---:|---|
| 1 | Empirical Career Transition Edges (Karrierewege) | ★★★★★ | Implemented | 2.48M career steps; 18,907 graph edges |
| 2 | Skill-Gap-Aware Career Path Ranking | ★★★★★ | Implemented | Primary method contribution; combines with #1 |
| 3 | Empirically-Optimized Cross-Taxonomy Alignment (ISCO Ground Truth) | ★★★★ | 1 week | Technical contribution with clear publishable figure |
| 4 | Graph-Provenance Faithfulness Verification | ★★★★ | 1 week | LLM reliability / hallucination prevention |
| 5 | Multi-Stage Pipeline Ablation | ★★★ | 3-5 days | Architecture validation (supports other contributions) |
| 6 | Provenance-Traced Explanation Chains | ★★★ | 1.5-2 weeks | Explainability (merges well with #4) |

**Recommended lead contributions:** #1 + #2 (strongest combination — real-world transitions + personalized ranking)

---

## Contribution 1: Empirical Career Transition Edges (Karrierewege Dataset)

### One-Line Claim
Derive empirical `TRANSITIONS_TO` edges from 2.48M career steps in 568,888 Karrierewege trajectories, enriching the static taxonomy graph with real-world career mobility patterns weighted by transition frequency.

### What's Novel (vs. Existing Work)
| Paper | What they do | What they DON'T do |
|---|---|---|
| Gugnani 2018 (Candidate Skill Graph) | Builds skill graphs from resumes | No empirical transition counts between occupations |
| Oentaryo 2018 (Talent Flow Analytics) | Analyses talent flow on LinkedIn | Uses social network, not occupational taxonomy KG |
| Nguyen 2024 (Graph Data Warehouse) | Graph warehouse for learning paths | No real career transition data integrated into KG |
| Striebel 2026 (HetKG Courses) | Heterogeneous KG for courses | Static taxonomy edges only, no empirical mobility |

**The gap:** No paper integrates real-world career trajectory data (person-level transitions) as empirical edges into an occupational knowledge graph alongside ONET/ESCO taxonomy edges, then uses these for GraphRAG recommendation.

### The Dataset
```
Location: data/Karrierewege/
Format:   3 CSV files (train/test/validation split)
Rows:     2,480,369 career steps across 568,888 individual trajectories
Columns:  _id, experience_order, preferredLabel_en, preferredLabel_de,
          description_en, description_de, preferredLabel_skills, skills

Key properties:
  - ESCO-native: titles ARE ESCO preferred labels (direct graph linkage)
  - 100% exact normalized linkage to ESCO graph roles; prior ONET mapping remains 77.1% (999/1295 titles)
  - Contains ESCO skill lists per career step
  - Bilingual (EN + DE)
  - Pre-split for ML evaluation (train/test/validation)
```

### How It Works
```
Raw career trajectories (person-level):
  Person A: Software Developer → Senior Developer → Tech Lead → CTO
  Person B: Data Analyst → Data Scientist → ML Engineer
  Person C: Data Analyst → Business Analyst → Product Manager
    │
    ▼
Aggregate transitions across the 455,129 training trajectories:
  Data Analyst → Data Scientist:     count=847, probability=0.12
  Data Analyst → Business Analyst:   count=623, probability=0.09
  Software Developer → Senior Dev:   count=2341, probability=0.18
    │
    ▼
Create TRANSITIONS_TO edges in the graph:
  - src: ESCO role node (matched by preferredLabel_en)
  - dst: ESCO role node (next step in trajectory)
  - attrs: {relation: "TRANSITIONS_TO", count: int, probability: float,
            source: "karrierewege", avg_steps: float}
    │
    ▼
Inference: when recommending career paths, weight by BOTH:
  - Semantic relevance (current pipeline)
  - Empirical transition likelihood (new signal)
```

### Implemented Components
- `src/karrierewege_preprocessing.py` — new module:
  - `load_transitions(data_dir)` — parse CSVs, group by _id, extract consecutive role pairs
  - `aggregate_transitions(pairs)` — count frequencies, compute probabilities, filter noise
  - `match_to_graph(transitions, G)` — match ESCO preferredLabel_en to graph node IDs
  - `add_transition_edges(G, transitions)` — add TRANSITIONS_TO edges to the graph
- Modify `build_graph.py` — add `--transitions` flag to integrate Karrierewege data
- Modify `traverse_graph()` in `inference_pipeline.py` — include TRANSITIONS_TO edges in traversal
- Modify `_build_path_data()` — use transition probability to order the career path
- Modify `_build_context_block()` — include transition counts in LLM context

### How To Validate
- **Coverage:** What fraction of graph roles have at least one TRANSITIONS_TO edge?
- **Prediction accuracy:** Using test split — given a person's current role, does the system recommend roles that people actually transition to?
- **Comparison:** Recommendations with vs. without transition edges (ablation variant)
- **Path quality:** Do transition-informed paths match real career progressions better than relevance-only paths?

### Feasibility
Implemented with streaming aggregation, support filtering, typed multigraph integration, and held-out evaluation. See `ALL_STEPS.md` for measured runtime outputs.

### Why This Is The Strongest Contribution
- Grounded in 2.48M real career-step rows (not synthetic or theoretical)
- Transforms a static taxonomy into a dynamic mobility graph
- The train/test split enables quantitative evaluation without extra work
- Combines naturally with Contribution #2 (skill-gap ranking uses transitions as validation)
- No other career KG paper has this — most use only taxonomy relationships

---

## Contribution 2: Skill-Gap-Aware Career Path Ranking

### One-Line Claim
Score and rank recommended career roles by structural skill overlap in the knowledge graph — ordering the career path from most-accessible to most-aspirational based on the user's existing skills.

### What's Novel (vs. Existing Work)
| Paper | What they do | What they DON'T do |
|---|---|---|
| Gugnani 2018 (Candidate Skill Graph) | Builds skill graphs from resumes | Doesn't compute gap scores or rank by accessibility |
| Boshkoska 2024 (Market-Driven Skills) | Identifies skill demand from market | Doesn't personalize paths or rank by gap |
| Du 2024 (LLM Job Rec) | Uses LLM for recommendations | No graph-structural ranking signal |
| Striebel 2026 (Heterogeneous KG Courses) | Recommends courses via KG | Doesn't score career paths by skill gap |

**The gap:** No paper combines (a) graph-structural skill matching via REQUIRES edges with (b) personalized gap scoring to (c) order a career path within a GraphRAG pipeline.

### How It Works
```
User states skills (from conversation)
    │
    ▼
Extract skill mentions → fuzzy-match to graph skill nodes
    │
    ▼
For each candidate role (top-8 from reranking):
    ├─ Collect all REQUIRES edges (essential skills)
    ├─ Compute: overlap = |user_skills ∩ role_skills|
    ├─ Compute: gap = |role_skills \ user_skills|
    └─ Score = overlap / |role_skills|  (0.0 = no match, 1.0 = fully qualified)
    │
    ▼
Rank roles by gap score (ascending = most accessible first)
    │
    ▼
Career path visual: left = most accessible → right = most aspirational
Each role card shows: skills you HAVE (green) vs. skills to LEARN (amber)
```

### Implemented Components
- `src/skill_gap.py` — new module:
  - `extract_user_skills(query, history, G, settings)` — use LLM or embedding match to identify user-mentioned skills in the graph
  - `compute_skill_gap(user_skill_ids, role_id, G)` — returns overlap score + gap list
  - `rank_roles_by_gap(user_skill_ids, anchor_ids, G)` — returns roles sorted by accessibility
- Modify `run_query()` in `inference_pipeline.py` — add gap-aware ranking after reranking
- Modify `_build_path_data()` — annotate each role with "have" vs. "need" skills
- Frontend: colour-code skill chips (green = user has, amber = to develop)

### How To Validate
- Compare ranking quality: gap-aware vs. pure relevance ranking (is the order more logical?)
- Measure: does the most-accessible role genuinely require fewer new skills?
- Include in ablation: full pipeline WITH gap ranking vs. WITHOUT

### Feasibility
Implemented with deterministic exact skill evidence, goal/negation exclusion, ESCO requirement scoring, ONET alignment fallback, and evidence-aware frontend output.

---

## Contribution 3: Empirically-Optimized Cross-Taxonomy Alignment (ISCO Ground Truth)

### One-Line Claim
Validate and optimize the ONET↔ESCO embedding alignment threshold using ISCO classification codes as silver-standard ground truth, producing a precision-recall curve that justifies the chosen threshold.

### What's Novel (vs. Existing Work)
| Paper | What they do | What they DON'T do |
|---|---|---|
| Guru Rao 2022 (ESCO-ONET Matching) | Aligns ESCO↔ONET via semantic similarity + rules | No threshold optimization; no ground truth validation |
| JobBERT (Decorte 2021/2025) | Job title understanding via skills | Doesn't align full taxonomies or produce cross-framework edges |
| Boshkoska 2024 | Uses ONET/ESCO for skill identification | Doesn't create alignment edges between the two |

**The gap:** No paper has used ISCO codes as silver-standard ground truth to empirically validate cross-taxonomy alignment quality or optimize the similarity threshold.

### How It Works
```
Both ONET and ESCO roles map to ISCO-08 codes:
  - ESCO: native isco_group attribute (4-digit, e.g. "2654")
  - ONET: derivable via SOC→ISCO crosswalk (published by BLS/ILO)

Ground truth rule:
  If ONET role and ESCO role share the same ISCO code (or parent) → SHOULD be aligned
  If they have different ISCO codes → SHOULD NOT be aligned

Experiment:
  For threshold in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    1. Compute all ONET×ESCO pairs above threshold (from similarity matrix)
    2. For each pair: check if ISCO codes match (at 2-digit and 4-digit level)
    3. Precision = pairs_with_matching_ISCO / total_pairs_above_threshold
    4. Recall = pairs_with_matching_ISCO / total_ISCO_matching_pairs_in_data
    5. Plot precision-recall curve

Output: optimal threshold with P-R tradeoff analysis
```

### What To Build
- Download SOC→ISCO-08 crosswalk (BLS publishes this as Excel/CSV)
- `evaluation/alignment_analysis.py`:
  - Load crosswalk, map ONET SOC codes to ISCO codes
  - Sweep thresholds on the precomputed similarity matrix
  - Compute precision/recall at each threshold
  - Generate P-R curve figure
- Add ISCO codes to ONET role nodes in `onet_preprocessing.py`

### Important Note
**Current state:** ONET roles do NOT have ISCO codes in the graph (verified). ESCO roles DO have them (all 3,039 have 4-digit ISCO codes). A SOC→ISCO crosswalk needs to be added — this is publicly available from the BLS/O*NET resource center (SOC-to-ISCO-08 mapping table).

### How To Validate
- Precision-recall curve (the contribution itself IS the validation)
- Compare optimal threshold vs. current threshold (0.65)
- Show qualitative examples: high-quality alignments vs. false positives at different thresholds
- Compare results against Guru Rao 2022's reported alignment quality

### Feasibility
1 week. The similarity matrix computation already exists in `compute_alignment_edges()`. Main work is: (a) obtain and integrate SOC→ISCO crosswalk, (b) write the sweep + analysis script.

---

## Contribution 4: Graph-Provenance Faithfulness Verification

### One-Line Claim
A post-generation verification mechanism that checks every entity cited by the LLM against the knowledge graph topology, computing a graph-provenance faithfulness score and flagging hallucinated entities.

### What's Novel (vs. Existing Work)
| Paper | What they do | What they DON'T do |
|---|---|---|
| RAGAS (Es et al. 2024) | RAG faithfulness vs. retrieved text chunks | Doesn't verify against structured KG with typed relations |
| KGGLM (Balloccu 2024) | Generates recommendations with KG | Doesn't post-hoc verify citations against graph |
| Upadhyay 2021 (Explainable Job KG) | Uses NER to link entities to KG | Doesn't compute faithfulness scores or verify generation output |

**The gap:** No paper measures LLM generation faithfulness against a typed knowledge graph (node existence + edge connectivity) rather than flat document chunks.

### How It Works
```
LLM generates: "Based on your background, **Data Scientist** would be
an excellent next step. You already have Python, and developing skills
in machine learning frameworks would bridge the gap."
    │
    ▼
Entity Extraction:
  - Parse for bold text + known patterns
  - Fuzzy match against all graph node titles (roles + skills)
  - Result: ["Data Scientist" → node_id, "Python" → node_id,
             "machine learning" → node_id_or_MISSING]
    │
    ▼
Graph Verification:
  - For each extracted entity: does it exist as a node? (existence check)
  - For each entity pair: is there a path of length ≤ 3? (connectivity check)
  - Faithfulness score = verified_entities / total_extracted_entities
    │
    ▼
Output: {"faithfulness": 0.95, "verified": [...], "hallucinated": [...]}
```

### What To Build
- `src/faithfulness.py`:
  - `extract_entities(text, G)` — parse LLM output, fuzzy match to graph nodes
  - `verify_entities(entity_ids, anchor_ids, G)` — check existence + connectivity
  - `compute_faithfulness_score(text, anchor_ids, G)` — full pipeline → score
- Optionally integrate into `run_query()` as a post-generation check
- Can compare: with graph constraint in prompt vs. without → faithfulness difference

### How To Validate
- Run 20-30 test queries, compute faithfulness scores
- Ablation: remove "only cite graph entities" from prompt → measure faithfulness drop
- Report: "With graph constraint: X% faithfulness. Without: Y%."
- Qualitative: show 5 examples of hallucinated vs. verified entities

### Feasibility
1 week. Entity extraction via fuzzy string matching against graph node titles is straightforward. The `_build_context_block` function already maps node IDs to titles (reverse mapping is trivial).

---

## Contribution 5: Multi-Stage Pipeline Ablation

### One-Line Claim
Systematic ablation study removing each pipeline stage independently (reranker, graph traversal, SIMILAR_TO edges, importance filtering, graph-only prompt constraint) to quantify its marginal contribution to recommendation quality.

### What's Novel (vs. Existing Work)
| Paper | What they do | What they DON'T do |
|---|---|---|
| KGAT/KGCN (Wang 2019) | Ablate model components (attention heads, layers) | Don't ablate pipeline stages of a retrieval system |
| Du 2024 (LLM Job Rec) | LLM-enhanced job rec | No pipeline ablation |
| Striebel 2026 (HetKG Courses) | Ablates feature types | Doesn't ablate retrieval+graph+generation pipeline |

**The gap:** No career recommendation paper ablates a multi-stage GraphRAG architecture (retrieve → rerank → traverse → generate).

### How It Works
```
Variants to test:
  A. FULL PIPELINE (baseline)
  B. No reranker — pass top-8 ChromaDB results directly to graph traversal
  C. No graph traversal — pass retrieval results directly to LLM (vector-only RAG)
  D. No SIMILAR_TO edges — remove cross-taxonomy edges before traversal
  E. No importance filtering — set ONET threshold to 0, include ESCO optional skills
  F. No graph constraint — remove "only cite graph entities" from system prompt

Run each variant on same test suite → compare metrics
```

### What To Build
- `evaluation/ablation.py` — orchestrator with flags to disable each stage
- Depends on metrics from Contribution #3 (faithfulness) or external evaluation
- Test suite of 20-30 queries (mix of student/professional, various domains)

### How To Validate
- Table: variant × metric scores
- Expected: full pipeline > any ablation, with specific stages contributing most
- Statistical: paired comparison per query

### Feasibility
3-5 days (after metrics from #3 are built). Each ablation is a parameter change in the existing modular pipeline.

### Note
This is more "evaluation methodology" than "novel method." Best used as supporting evidence for other contributions (#1, #2, #3) rather than standalone novelty. Include in the paper but don't lead with it.

---

## Contribution 6: Provenance-Traced Explanation Chains

### One-Line Claim
A post-generation explainability layer that traces each recommended entity back through the knowledge graph, producing human-readable evidence chains showing the typed edges that justify each recommendation.

### What's Novel (vs. Existing Work)
| Paper | What they do | What they DON'T do |
|---|---|---|
| Upadhyay 2021 | Explainable job rec with KG + NER | Doesn't trace graph paths for explanations |
| Gutierrez 2019 | Explanation interfaces for job recs | Uses collaborative filtering, not graph paths |
| KGGLM (Balloccu 2024) | KG-based LLM recommendation | Doesn't produce separate path-based explanations |
| JobRecoGPT (Ghosh 2023) | Explainable LLM job recs | Uses prompt-based explanations, not graph provenance |

**The gap:** No paper produces graph-topology-traced explanation chains for career LLM recommendations.

### How It Works
```
For each entity mentioned in LLM response:
  1. Find entity in graph (fuzzy match → node_id)
  2. Find shortest path from user's anchor roles to that entity
  3. Format as human-readable chain:
     "Recommended because: Your background in [Software Engineering]
      → REQUIRES → Python (which you have)
      → Also REQUIRES → Machine Learning
      → SIMILAR_TO → Data Scientist (ESCO)
      → REQUIRES → Deep Learning Frameworks (skill to develop)"
```

### What To Build
- `src/explainability.py`:
  - `find_explanation_paths(entity_id, anchor_ids, G)` — nx.shortest_path with edge labels
  - `format_explanation(paths, G)` — human-readable chain strings
- Modify response to include optional `explanations` field
- Frontend: collapsible "Why this?" section per role card

### How To Validate
- Explanation coverage: what fraction of recommended entities have valid graph paths?
- Path length distribution (shorter = more direct = more convincing)
- Optional: small user study (5-10 people) — trust/understanding Likert scale

### Feasibility
1.5-2 weeks. NetworkX shortest_path is trivial on 19K nodes. The harder part is entity disambiguation (same challenge as #3) and meaningful path formatting. Consider merging with #3 — faithfulness verification IS the foundation for explanations.

---

## Recommended Combinations

### Option A: Data-Driven + Personalized (STRONGEST)
**Lead with #1 (Transition Edges) + #2 (Skill-Gap Ranking)**
- #1 enriches the graph with real-world mobility data (unique dataset advantage)
- #2 uses those transitions + skill structure for personalized path ordering
- Together: "We integrated empirical career mobility into a KG, then used graph structure to personalize recommendations by skill gap"
- #5 (ablation) validates both
- Total: ~3-4 weeks

### Option B: Data-Driven + Technical Validation
**Lead with #1 (Transition Edges) + #3 (ISCO Alignment Validation)**
- #1 is the primary novelty (data contribution)
- #3 validates the cross-taxonomy infrastructure that enables #1
- Clean publishable outputs: transition statistics + P-R curve
- Total: ~2-3 weeks

### Option C: Maximum Differentiation (3 contributions)
**#1 (Transitions) + #2 (Skill-Gap) + #4 (Faithfulness)**
- Three distinct angles: data enrichment + method innovation + reliability
- #5 (ablation) validates all three
- Total: ~4-5 weeks

### Option D: Quick Wins + Depth
**#1 (Transitions) + #3 (ISCO Alignment) + #4 (Faithfulness)**
- All three are 1-week implementations
- Together they cover: data, infrastructure, reliability
- #5 (ablation) ties them all together
- Total: ~3-4 weeks

---

## Available Datasets Assessment

| Dataset | Rows | Verdict | Why |
|---|---|---|---|
| **Karrierewege** | 2,480,369 | **IMPLEMENTED** | ESCO-native, empirical transitions, person-disjoint train/validation/test splits |
| LinkedIn | 3.3M | Skip | Skills too coarse (36 categories), would need heavy NLP for marginal KG benefit |
| Monster.com | 22K | Skip | Too small, noisy HTML, no structured skills or transitions |
| job-descriptions | 32K | Skip | Only useful as NLP training data, tangential to KG thesis |

---

## What's Already Done (Don't Claim as New Contribution)

These are already implemented and should be described as system features, not novelty:

- **Adaptive Progressive Disclosure** — partial/full context routing with explore panels. Novel UX pattern but hard to evaluate without user study. Describe in system design section.
- **Cross-taxonomy alignment via embedding similarity** — the mechanism itself exists. What's novel is the ISCO validation (#3), not the basic approach.
- **Graph-only prompt instruction** — already in place. What's novel is measuring its effect (#4), not the instruction itself.

---

## Next Steps

1. Use the held-out and ablation outputs in the thesis evaluation chapter.
2. Decide whether to extend the work with #3 (ISCO alignment validation) or #4 (faithfulness verification).
3. For #3, obtain a defensible SOC→ISCO-08 crosswalk and keep it separate from Karrierewege test data.
4. Rebuild the stale ChromaDB role collection before a deployment-focused evaluation.
