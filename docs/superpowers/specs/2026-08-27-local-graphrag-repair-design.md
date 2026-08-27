# Local GraphRAG Repair and Metric Improvement Design

**Date:** 2026-08-27
**Status:** Approved for local-first implementation
**Target branch:** `dev`

## Objective

Repair the non-deployment findings from the GitHub Copilot review, make the
GraphRAG application work reliably with the local graph, Chroma index, and link
prediction model, and attempt a defensible improvement to transition-ranking
metrics. Preserve the academic boundary between observed training transitions,
semantic inference, and link-predicted missing edges.

## Scope

### Included

- Enforce training-only transition evidence throughout runtime and model code.
- Correct Transition Effort Score (TES) and skill readiness so they use
  role-relevant, comparable requirements.
- Fully wire hybrid retrieval and optional link prediction into both the classic
  and LangGraph application paths.
- Repair LangGraph partial-context routing.
- Tune and evaluate transition fusion using validation data and report honest
  Hits@K, MRR, and coverage results.
- Fix the remaining correctness, faithfulness, explainability, concurrency,
  accessibility, and small frontend findings from the review.
- Add regression tests and verify the application locally.

### Excluded

- Committing `graph/` or `index/` to Git.
- Packaging those data stores into the Vercel function.
- Implementing the external graph/vector database adapter.
- Declaring the Vercel deployment ready before that adapter exists.

The local graph and Chroma index remain the runtime sources for this work.
Deployment storage will be handled separately.

## Verified Starting Point

- Local graph: 19,225 nodes and 224,711 edges.
- Observed transitions: 18,907 `TRANSITIONS_TO` edges, all marked `split=train`.
- Enrichment: IDF on 13,570 skill/element nodes, ISCO mappings on 3,039 ESCO
  roles and 893 O*NET roles.
- Missing-edge model: LightGBM artifact at
  `artifacts/link_prediction/link_predictor.pkl`.
- Existing tests: 100 `unittest` tests pass, but the reviewed regressions are
  not covered.
- Existing link prediction is evaluated offline but is not connected to live
  retrieval. The graph contains no persisted predicted-transition edges.
- The project virtual environment lacks `langgraph`, although it is declared in
  `requirements.txt`.

## Architecture

### 1. Central transition evidence policy

Introduce one shared predicate for an eligible observed transition:

- relation is `TRANSITIONS_TO`;
- source is `karrierewege`; and
- split is absent for legacy training graphs or is exactly `train`.

All direct-transition consumers must call this policy: candidate augmentation,
graph traversal, skill-gap evidence, TES empirical support, explanations,
smoothing distributions, link-prediction training indexes, and evaluation
prediction maps.

Validation/test transitions are ignored by runtime consumers. They remain input
only to evaluation functions. Smoothing should skip ineligible edges and report
the count rather than disabling all smoothing for an otherwise valid graph.

### 2. Role-aware requirements and effort scoring

Use one canonical requirement resolver for readiness and TES:

- ESCO target: essential ESCO requirements only.
- O*NET target with alignment: essential requirements from the strongest aligned
  ESCO role.
- O*NET target without alignment: O*NET requirements at or above the configured
  importance threshold only.

This replaces TES's current use of every raw `REQUIRES` edge, which includes
optional ESCO requirements and low-importance O*NET abilities.

IDF weighting remains the deterministic relevance signal. English,
comprehension, writing, and communication requirements are not globally
penalized or assumed to be owned. They receive their graph-derived importance:
they can remain central for teachers, writers, translators, and communication
roles, while occupationally specific skills dominate technical-role gaps.
Both the readiness percentage and the TES skill-gap/transferability components
use the same IDF-weighted requirement set, while the UI continues to show raw
matched and required counts for transparency.

The response prompts will instruct the LLM to:

- prioritize occupation-specific, high-signal skill gaps;
- mention general language/comprehension abilities as the principal gap only
  when they are central to the destination role or no stronger graph-supported
  gap exists;
- bold every role and skill it names; and
- never reinterpret a predicted transition as an observed move.

The LLM explains deterministic readiness and TES evidence; it does not generate
the numerical score.

### 3. Hybrid retrieval and missing-edge prediction

Create a lazily initialized runtime link-prediction context containing:

- the loaded LightGBM model;
- live ESCO embeddings loaded from the local Chroma collection;
- a training-only transition index;
- the semantic-neighbour index used during training; and
- IDF data required by the model's nine-feature contract.

When `LINK_PREDICTION_ENABLED=true`, model inference must receive the same feature
families used in training. Missing model/vector inputs produce an explicit
diagnostic and a safe non-LP fallback.

The shared hybrid retriever will combine:

1. vector candidates, honoring the requested vector limit;
2. observed direct training transitions;
3. embedding-smoothed transition candidates;
4. graph skill-overlap candidates; and
5. link-predicted missing transitions when enabled.

Sources are fused using deterministic reciprocal-rank or validation-selected
weighted rank fusion. Candidate provenance is preserved. Predicted transitions
are virtual runtime evidence labelled `predicted_transition`; they are not
persisted as observed edges in the knowledge graph.

The validation-selected fusion governs transition ordering. If LP does not pass
the metric acceptance gate, it remains available as explicitly low-priority
missing-edge evidence for otherwise unsupported candidates and cannot displace
the accepted direct/smoothed ranking.

Both `run_query()` and `agents/nodes/retrieval.py` will call this shared path so
the documented feature does not remain dead code.

### 4. LangGraph routing

Keep shared retrieval, ranking, and traversal, then route conditionally after
traversal:

- full context: `effort -> generation -> faithfulness -> explanation -> courses
  -> END`;
- partial context: `explore -> END`.

Partial requests must return follow-up questions and explore panels without
generation, effort, explanations, or courses.

### 5. Correctness and evidence repairs

- Pass an intentional limit to transition smoother ranking in TES.
- Recognize normalized contractions such as `don t` and explicit `no` negation
  when resolving owned skills.
- Expand faithfulness entity extraction beyond bold phrases using conservative
  quoted, capitalized, and skill-cue candidates while deduplicating normalized
  entities.
- Select the maximum-similarity alignment edge for explanations.
- Ensure explanation transition evidence is training-only.
- Make model and graph resource initialization process-local and synchronized.
- Surface non-fatal initialization diagnostics rather than silently swallowing
  all failures.

### 6. Frontend and accessibility repairs

- Use shared assistant-message and evidence-panel classes for explanation output.
- Add the missing explanation graph icon.
- Add `tabindex="-1"` to both skip-link destinations.
- Protect graph retries with request identity so an older aborted request cannot
  clear or overwrite the active request.
- Add a compact native graph-node selector populated from the sampled nodes.
  Selecting a node invokes the same inspection/highlighting behavior as pointer
  selection, giving keyboard and screen-reader users access without pretending
  that a focusable canvas makes individual nodes operable.

Top-level `public/**` remains tracked. It is intentionally outside the Python
function bundle because Vercel serves that directory as static assets; this is
not part of the local repair.

## Metric Improvement Method

### Baselines

Existing locked-test results:

- direct: Hits@5 0.370239, Hits@10 0.501359, MRR 0.259056, source coverage
  0.738878;
- embedding smoothing: Hits@5 0.372183, Hits@10 0.505809, MRR 0.261703,
  source coverage 1.0;
- current direct/smooth/LP priority combination: Hits@5 0.370426, Hits@10
  0.501700, MRR 0.258038, source coverage 1.0.

The existing evidence therefore supports a smoothing improvement and an LP
coverage improvement, but not a meaningful LP Hits@K improvement.

### Tuning protocol

- Build all model evidence from training transitions only.
- Compare direct, smoothing, standalone LP, priority fallback, weighted fusion,
  and RRF fusion on validation data.
- Search only a small, documented deterministic weight grid.
- Select a fusion only when it improves validation Hits@5 or Hits@10 over the
  accepted smoothing baseline while keeping the other Hits metric and MRR within
  0.001 absolute of that baseline.
- Freeze the selected configuration before reporting test results.
- Report absolute and relative deltas for Hits@5, Hits@10, MRR, source-role
  coverage, and destination coverage.
- If no LP fusion passes the acceptance gate, retain smoothing for ranking and
  describe LP honestly as missing-edge/coverage evidence rather than claiming it
  improves Hits@K.

The historical test split has already been evaluated in prior sessions, so the
new report must not describe it as previously unseen. New tuning decisions will
still use validation data only.

## Error Handling and Degradation

- Missing graph or Chroma remains a clear local readiness error.
- Missing/invalid LP model disables only the LP source and records why.
- Missing vectors disable feature-dependent LP safely; they are never replaced
  with silent zero features presented as equivalent inference.
- Failure of one retrieval source does not discard successful sources.
- Empty retrieval produces the existing recoverable user-facing response.
- No exception path may expose validation/test transitions as direct evidence.

## Testing and Verification

Add tests for:

- held-out edges being excluded from every runtime/evidence/model index;
- the smoother fallback receiving its required limit;
- `don't`, `dont`, `don t`, and `no ... experience` negation;
- essential/aligned/high-importance requirement selection for technical and
  communication-centric roles;
- role-aware TES and readiness behavior;
- unbolded skill entity extraction;
- strongest similarity-edge selection;
- vector limit enforcement and use of every advertised hybrid source;
- LP feature parity between training and runtime;
- full and partial LangGraph routes;
- synchronized resource initialization;
- graph retry identity and static JavaScript syntax;
- skip-link focus targets and accessible node selection.

Verification sequence:

1. install the missing declared local runtime dependency if necessary;
2. run the complete offline `unittest` suite;
3. run Python compilation and JavaScript syntax checks;
4. run local status, full-context chat, partial-context chat, and graph-data
   smoke checks using local graph/index resources;
5. run validation fusion tuning and the frozen evaluation report;
6. confirm `graph/` and `index/` remain untracked and excluded from Git.

External model/API smoke calls should be minimized and clearly identified.

## Completion Criteria

- All non-deployment Copilot findings are fixed or documented with evidence when
  a finding is not applicable.
- The application works locally in both classic and LangGraph modes.
- No held-out transition can enter runtime evidence or LP training.
- TES/readiness use role-relevant requirements and no longer inflate technical
  transitions with optional or low-importance generic requirements.
- LP is live when enabled and uses training-compatible features.
- Metric claims match the saved evaluation report and acceptance gate.
- Existing local graph/Chroma data are not committed.
