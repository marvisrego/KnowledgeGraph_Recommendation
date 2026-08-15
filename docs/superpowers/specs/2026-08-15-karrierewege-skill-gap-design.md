# Karrierewege Transition Graph and Skill-Gap Ranking Design

Date: 2026-08-15
Status: Approved for implementation planning
Scope: Novelty contribution 1 (empirical career transitions), novelty contribution 2 (skill-gap-aware career-path ranking), held-out evaluation, and a chronological implementation ledger

## 1. Objective

Enrich the existing ONET/ESCO GraphRAG career knowledge graph with empirical occupation transitions derived from the Karrierewege dataset, then personalize the ordering of shortlisted roles using the user's explicitly stated skills and graph `REQUIRES` relationships.

The production graph will be trained only from the Karrierewege training split. Validation and test trajectories will remain held out for leakage-free evaluation.

## 2. Verified Starting Point

Local inspection established the following facts:

- The raw dataset contains 2,480,369 rows, not the 3,036,454 stated in the earlier handoff documents.
- It contains 568,888 person-disjoint trajectories:
  - train: 1,984,657 rows and 455,129 people;
  - validation: 248,465 rows and 56,909 people;
  - test: 247,247 rows and 56,850 people.
- No person identifier appears in more than one split.
- All 1,295 normalized Karrierewege English occupation labels map exactly to retained ESCO occupation nodes in the current graph.
- The training split contains 31 ambiguous duplicate `(person, experience_order)` positions and one exact duplicate row.
- After removing ambiguous positions, exact duplicates, nonconsecutive steps, and same-role transitions, the training data contains 986,803 valid non-self transition observations across 90,170 distinct ordered role pairs.
- A minimum support threshold of five retains approximately 18,907 edges and 88.6% of valid non-self transition observations.
- At that threshold, at least 130 transition pairs overlap an existing directed ESCO taxonomy pair. A plain `DiGraph` would overwrite one of the relationships.
- The current graph has 19,225 nodes, 205,804 edges, and 3,932 role nodes. Existing working-tree changes implement KG pruning, stricter requirement filtering, partial-context exploration, and career-path UI output; these changes must be preserved.

## 3. Architectural Decision

Use a NetworkX `MultiDiGraph` with deterministic typed edge keys. This allows the same ordered node pair to carry independent taxonomy, alignment, requirement, and empirical-transition evidence.

Each graph edge will retain a single semantic relation. Parallel relations will not be packed into a list-valued attribute because doing so would complicate traversal, provenance, filtering, and evaluation.

Legacy `DiGraph` pickles will be upgraded in memory without changing node identifiers. The graph node set is unchanged by transition enrichment, so the existing role-only ChromaDB index remains valid and will not be rebuilt.

## 4. Offline Data Pipeline

### 4.1 Split discovery

`src/karrierewege_preprocessing.py` will recursively locate exactly one train, validation, and test CSV below the configured Karrierewege directory. Discovery will tolerate the existing `karrierewege`/`karierewege` filename inconsistency but will reject missing or duplicate split matches.

Only these columns will be read:

- `_id`
- `experience_order`
- `preferredLabel_en`

Descriptions, German labels, and row-level skill lists are unnecessary for occupation-transition construction and will not be loaded into memory.

### 4.2 Streaming and trajectory boundaries

The approximately 4 GB training file will be processed in configurable chunks. Because a person's rows may cross a chunk boundary, the final person's rows from each chunk will be carried into the next chunk before cleaning.

The processor will verify that each person occupies one contiguous run. Reappearance of a completed person identifier will be treated as a structural data error rather than silently generating an incorrect transition.

### 4.3 Cleaning rules

For each person:

1. Require non-null person identifier, integer `experience_order`, and non-empty English occupation label.
2. Normalize labels with Unicode normalization, case folding, whitespace collapse, and conservative punctuation handling.
3. Sort stably by `experience_order`.
4. Collapse exact duplicate rows at the same order.
5. If one order contains conflicting occupation labels, remove that order as ambiguous.
6. Do not bridge across an ambiguous or missing order; only order difference `+1` forms a transition.
7. Remove consecutive same-role transitions.
8. Match both labels exactly against the normalized ESCO occupation-title index.
9. Count the remaining ordered source/destination pairs.

No fuzzy occupation mapping is permitted because exact normalized coverage is already 100%.

### 4.4 Aggregation and probabilities

For every ordered non-self pair, retain its training count. For each source role, define `source_total` as the count of all valid non-self outgoing training observations before rare-edge filtering.

The empirical probability is:

`probability = pair_count / source_total`

Edges with `pair_count < transition_min_count` are omitted. Because the denominator is calculated before filtering, retained outgoing probabilities may sum to less than one and explicitly preserve the probability mass removed as low-support noise.

Each retained edge will have:

```python
{
    "relation": "TRANSITIONS_TO",
    "source": "karrierewege",
    "count": int,
    "probability": float,
    "source_total": int,
    "split": "train",
    "min_support": int,
}
```

`avg_steps` will not be stored. Accepted transitions are consecutive by definition, so this value would always be one and would not add information.

### 4.5 Data-quality report

Every split will produce a report containing raw rows, accepted rows, trajectories, missing values, exact duplicates, ambiguous positions, nonconsecutive pairs, self-transitions, mapped and unmapped labels, valid transition observations, distinct pairs, and support-threshold retention.

The combined report will be written as JSON under `artifacts/karrierewege/` and summarized in `ALL_STEPS.md`. It will contain aggregate counts only, never person-level records.

## 5. Graph Construction and Persistence

`src/graph_build.py` will:

- construct new graphs as `nx.MultiDiGraph`;
- provide a legacy `DiGraph` to `MultiDiGraph` migration helper;
- assign deterministic edge keys based on source and relation;
- upsert rather than duplicate alignment and transition edges;
- remove only existing `source="karrierewege"`, `relation="TRANSITIONS_TO"` edges before re-enrichment;
- preserve pruning behavior and all existing node attributes;
- continue loading legacy and new graph pickles;
- save graph pickles atomically through a temporary file followed by replacement.

`build_graph.py` will support:

- `--transitions`: include training-derived transition edges in a full graph build;
- `--transitions-only`: load the existing graph, migrate it if necessary, replace Karrierewege edges, and save it without rebuilding nodes, embeddings, or alignment;
- configurable chunk size and minimum support, with settings-backed defaults.

The current workspace will use `--transitions-only` so the 132 existing `SIMILAR_TO` edges and ChromaDB collection remain intact.

## 6. Skill Extraction and Matching

The intake response will be extended with:

```json
{
  "has_context": true,
  "user_type": "professional",
  "current_role": "data analyst",
  "skills": ["Python", "SQL"],
  "career_goal": "move into data science"
}
```

The prompt will require `current_role`, `skills`, and `career_goal` to be copied or conservatively summarized only from user-provided information. Skills mentioned only by the assistant will never be treated as user-owned.

`src/skill_gap.py` will build a deterministic index from graph skill titles. Index aliases will include normalized preferred labels and safe parenthetical bases, such as `Python` for `Python (computer programming)`. Short aliases will be ignored except for a small explicit allowlist such as `R`, `C`, `ML`, and `AI` to avoid substring false positives.

Router-provided skill phrases will be matched exactly after normalization. A fallback will inspect only user-authored statements with possession/experience patterns. It will not scan arbitrary assistant text or infer skills from a desired role. Negated skills will not be marked as owned.

When one normalized label maps to multiple graph skill identifiers, all exact equivalents may satisfy an overlap check; the API will expose a stable representative label.

## 7. Skill-Gap Computation

For an ESCO role, the requirement set is its outgoing `REQUIRES` edges with `requirement_level="essential"`.

For an ONET role, the scorer will use the highest-similarity aligned ESCO role when a `SIMILAR_TO` edge is available. If no aligned ESCO role exists, gap evidence is marked unavailable rather than comparing granular ESCO user skills with broad ONET elements as if they were equivalent.

For a role with usable requirements:

- `have = user_skill_ids ∩ required_skill_ids`
- `need = required_skill_ids - user_skill_ids`
- `accessibility = len(have) / len(required_skill_ids)`
- `gap = 1 - accessibility`

The score uses the complete essential requirement set. The UI may show a capped, deterministic selection of owned and missing skills, but the displayed counts and score will still refer to the full set.

If the user has no matched skills or the role has no comparable requirements, `accessibility` and `gap` will be `null`, with an explicit evidence-status value. The system will not convert missing evidence into a zero score.

## 8. Candidate Expansion and Ranking

### 8.1 Current-role resolution

For professionals, the router-provided current role will be resolved against ESCO role titles by:

1. exact normalized title;
2. exact safe alias;
3. high-threshold fuzzy match only when the best match is unambiguous.

An unresolved role disables transition expansion without failing the request. Students do not require a current occupation and normally skip transition expansion.

### 8.2 Candidate pool

The semantic retrieval pool will be augmented with the strongest outgoing Karrierewege destinations from the resolved current role. Candidates will be deduplicated by graph node identifier. Transition-derived candidates will receive graph-generated role documents before the combined pool is sent through the existing reranker, so reranker scores remain comparable.

### 8.3 Final path order

Semantic retrieval and reranking define a relevant shortlist. The visible career path then orders shortlisted roles from most accessible to most aspirational using:

1. accessibility descending when skill evidence is available;
2. direct transition probability descending;
3. empirical transition count descending;
4. semantic reranker score descending;
5. normalized role title and node identifier for deterministic ties.

If no role has skill evidence, the empirical and semantic order is preserved. Roles with unavailable gap evidence remain eligible but are never presented as more accessible merely because data is absent.

## 9. Traversal, Generation Context, and API Contract

Graph traversal will include outgoing `TRANSITIONS_TO` relations for selected role nodes, bounded to a configurable maximum and sorted by probability/count. Existing requirement, taxonomy, and alignment filtering remains in effect.

The generation context will describe empirical evidence explicitly, for example:

`(Data analyst) --[TRANSITIONS_TO]--> (Data scientist) [count=..., probability=...]`

The LLM prompt will distinguish observed population-level transitions from guaranteed individual outcomes.

Each path role returned to the frontend will include:

```json
{
  "id": "...",
  "title": "...",
  "source": "esco",
  "accessibility": 0.25,
  "gap": 0.75,
  "gap_evidence": "direct_esco",
  "required_skill_count": 20,
  "have": [{"id": "...", "title": "Python (computer programming)"}],
  "need": [{"id": "...", "title": "machine learning"}],
  "transition": {
    "from_role_id": "...",
    "count": 847,
    "probability": 0.12
  },
  "semantic_score": 0.91
}
```

Fields without evidence will be `null` or empty rather than fabricated. Existing `message`, `courses`, `path`, and `explore` response keys will remain backward-compatible.

## 10. Frontend Behavior

The existing career-path component will be extended, not redesigned.

- Owned skill chips use a restrained positive/green state.
- Skills to develop use the existing amber emphasis.
- Roles with a calculated score show an accessibility percentage.
- Direct empirical moves show compact transition count/probability evidence.
- Missing gap evidence uses neutral explanatory copy rather than a misleading score.
- Partial-context explore panels retain their current behavior.
- Existing responsive scrolling, title truncation, and visual hierarchy remain intact.

## 11. Evaluation

### 11.1 Transition prediction

An evaluation module will build predictions only from training edges and evaluate cleaned validation and test transitions.

For each held-out transition whose source role exists in the graph, outgoing training destinations are ranked by probability, count, then stable title. Metrics will include:

- source-role coverage;
- held-out observation coverage;
- Hits@1, Hits@3, Hits@5, and Hits@10;
- mean reciprocal rank, with missing destinations contributing zero;
- covered-only versions of ranking metrics;
- macro metrics across source roles where meaningful.

Validation metrics may guide configuration reporting. Test metrics are reported once as final held-out results and are never used to build or tune the graph.

### 11.2 Skill-gap validation

Tests and an offline evaluation will verify that:

- adding a genuinely required owned skill cannot reduce accessibility;
- removing an owned required skill cannot increase accessibility;
- `have` and `need` are disjoint and partition the requirement set;
- unmatched and negated skills do not become owned evidence;
- ranking is stable under ties;
- unavailable evidence remains distinct from a zero-overlap result.

An ablation output will compare semantic-only ordering with transition expansion and skill-gap-aware ordering on a fixed local query fixture set. External API-dependent comparisons will be separated from deterministic unit tests.

Evaluation summaries will be stored under `artifacts/karrierewege/` as JSON and CSV where tabular output is useful.

## 12. Failure Handling

The build must stop before graph mutation when:

- a required split cannot be uniquely discovered;
- required columns are absent;
- order values are non-integral beyond the documented rejected-row path;
- a completed person identifier reappears later in a split;
- normalized occupation mapping is not unique;
- any transition endpoint is absent from the ESCO graph;
- calculated counts or probabilities violate invariants.

API-time extraction, role resolution, or transition lookup failures degrade to the existing semantic pipeline. External chat, embedding, reranking, and Coursera failures retain their current fallbacks.

## 13. Testing and Acceptance Criteria

Unit tests will use tiny synthetic CSV and graph fixtures and require no external services. They will cover split discovery, chunk-boundary trajectories, duplicates, conflicting steps, gaps, self-transitions, probability denominators, threshold filtering, exact ESCO mapping, idempotent edge replacement, parallel-edge preservation, skill aliases, negation, ONET alignment fallback, ranking, context formatting, and response shape.

The implementation is accepted when:

- all 2,480,369 rows and 568,888 disjoint trajectories are accounted for in the quality report;
- only training rows contribute graph transitions;
- no self or nonconsecutive transition is stored;
- support-five output is consistent with approximately 18,907 transition edges and 88.6% observation retention;
- all known taxonomy/transition collisions preserve both relations;
- node identifiers and role count remain unchanged;
- the existing 132 `SIMILAR_TO` edges remain present after transition-only enrichment;
- probability and skill-gap invariants pass;
- held-out validation/test metrics are generated reproducibly;
- API and frontend outputs expose evidence without breaking partial-context behavior;
- the server health and representative full/partial-context smoke tests pass, or any external-service blocker is recorded precisely.

## 14. Documentation Ledger

Create `ALL_STEPS.md` at the repository root. It will be chronological and include:

1. the inherited project state and relevant historical changes summarized from `HANDOFF.md`;
2. the original novelty proposals and the selected contributions;
3. the verified dataset audit and corrections;
4. design decisions and rejected alternatives;
5. files added or changed;
6. preprocessing and graph-build commands;
7. cleaning and graph statistics;
8. unit, evaluation, and smoke-test commands and results;
9. limitations and reproducible next steps.

The ledger will never copy API keys, `.env` values, or person-level raw data. `HANDOFF.md` and `novelty.md` will be corrected where their dataset counts or implementation status become outdated.

## 15. Scope Boundaries

This implementation does not:

- train on validation or test data;
- introduce fuzzy Karrierewege-to-ESCO occupation mappings;
- rebuild role embeddings when node content is unchanged;
- claim causal career mobility from observational transitions;
- infer that a user owns a skill merely because it is relevant to a desired role;
- add unrelated datasets or redesign unrelated frontend areas;
- overwrite or discard existing uncommitted work.
