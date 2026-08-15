# Embedding-Smoothed Karrierewege Transitions

**Date:** 2026-08-15

**Status:** Implemented and accepted by the pre-registered gate

**Target branch:** `dev`

## 1. Objective

Use the existing `text-embedding-3-large` ESCO role vectors to improve Karrierewege next-role recommendations without weakening the existing leakage boundary or claiming that a semantically inferred move was directly observed.

The enhancement is accepted only if a validation-tuned, test-locked comparison improves the current transition baseline. If it fails the acceptance gate, the production recommendation path remains unchanged and the result is documented as a negative experiment.

## 2. Evidence for the chosen approach

Karrierewege integration already has complete normalized title coverage:

- training: 1,284 of 1,284 unique titles mapped;
- validation: 1,132 of 1,132 unique titles mapped;
- test: 1,121 of 1,121 unique titles mapped.

The dataset's English descriptions and skills are role-level ESCO content repeated across observations. Re-embedding nearly two million training rows would duplicate existing ESCO information, overweight common occupations, increase cost, and add no mapping coverage.

The current graph nevertheless has a sparsity boundary: support filtering retains 18,907 direct training edges, source-role coverage is about 74%, and held-out destination coverage is about 87%. Existing role vectors can provide a principled backoff by borrowing training-derived transition distributions from semantically similar ESCO source roles.

## 3. Considered approaches

### 3.1 Embed every Karrierewege record — rejected

This duplicates identical role descriptions and skills many times, creates frequency-biased vectors, adds person-level processing risk, and cannot improve the already complete title mapping.

### 3.2 Add destination titles to role embedding documents — rejected for this experiment

Embedding a source role together with its popular destinations would mix occupational meaning with transition popularity. It would also require rebuilding the vector index and could degrade ordinary semantic role retrieval, making attribution difficult.

### 3.3 Smooth transition distributions with existing role vectors — selected

Use semantic similarity only to identify related source occupations. Continue deriving every destination distribution exclusively from training `TRANSITIONS_TO` edges. This isolates the experimental signal, reuses the existing index, and supports deterministic held-out evaluation.

## 4. Model

For an ESCO source role `s` and destination `d`, the graph provides the direct training distribution:

```text
P_direct(d | s) = training_count(s, d) / training_source_total(s)
```

Let `N_k(s)` be the `k` most similar ESCO source roles that:

- are valid nodes in the current graph;
- have a stored role vector;
- have at least one retained training transition;
- are not `s` itself.

Cosine similarity is converted into normalized non-negative neighbour weights with a temperature-controlled softmax:

```text
w_i = softmax(cosine(vector(s), vector(n_i)) / temperature)
```

The neighbour transition distribution is:

```text
P_neighbour(d | s) = Σ_i w_i × P_direct(d | n_i)
```

The hybrid score is:

```text
score(d | s) = λ × P_direct(d | s)
             + (1 − λ) × P_neighbour(d | s)
```

Candidates are the union of direct destinations and neighbour destinations. The current source role is excluded. Deterministic ties are broken by direct probability, direct count, neighbour support, normalized title, and node identifier.

When a source has no direct retained transitions, the neighbour distribution becomes the complete score rather than being reduced by an unavailable direct component. If vectors or neighbours are unavailable, evaluation and runtime fall back to the current direct transition ranking.

## 5. Vector access

No new embedding API request is required for the experiment:

1. Load stored role embeddings from the existing Chroma `roles_collection`.
2. Retain only live ESCO role identifiers from the graph.
3. L2-normalize vectors once.
4. Calculate ESCO-to-transition-source neighbours in deterministic NumPy blocks.

The implementation must not modify the Chroma collection. It must tolerate the known stale records by intersecting collection identifiers with live graph identifiers.

The neighbour computation is a replaceable component: offline evaluation may materialize it in memory, while runtime can cache neighbours by resolved ESCO role. The interface returns role IDs, similarities, and training-only destination scores without exposing Chroma implementation details to ranking code.

## 6. Leakage-safe evaluation

### 6.1 Baseline

Recompute the current direct-edge metrics through the existing `evaluate_transition_predictions` path. The recorded test reference is:

- MRR: `0.2590564108`;
- Hits@5: `0.3702386957`;
- Hits@10: `0.5013586293`.

### 6.2 Validation tuning

Only validation observations select hyperparameters. The initial deterministic grid is:

- neighbours `k`: `3, 5, 10, 20`;
- direct weight `λ`: `0.50, 0.65, 0.80, 0.90`;
- softmax temperature: `0.05, 0.10, 0.20`.

Select the configuration with the highest validation MRR. Ties are resolved by validation Hits@5, Hits@10, destination coverage, then the simpler configuration: larger direct weight, fewer neighbours, and higher temperature.

The selected configuration is frozen before the test split is evaluated. The test split must be evaluated exactly once by the normal command path. No test-driven parameter adjustment is permitted.

### 6.3 Acceptance gate

Runtime integration occurs only when all of the following hold:

1. validation MRR is greater than the direct baseline;
2. test MRR is greater than `0.2590564108`;
3. test Hits@5 or Hits@10 exceeds its recorded baseline;
4. neither test Hits@5 nor Hits@10 decreases by more than `0.005` absolute;
5. source and destination coverage do not decrease;
6. results are deterministic across two identical metric runs.

Failure leaves production ranking unchanged. The experiment report still records parameters, metrics, deltas, coverage, and the rejection reason.

## 7. Conditional runtime integration

If the gate passes, professional recommendations with a resolved ESCO current role receive a bounded set of smoothed transition candidates before the existing reranker and skill-gap ordering. Existing behavior is preserved as follows:

- direct observed destinations remain first-class empirical evidence with their original counts and probabilities;
- smoothed-only candidates are labelled `semantic_transition_backoff` and never described as directly observed moves;
- the existing semantic retrieval pool, reranker, skill-gap calculation, path traversal, course lookup, response schema, and routes remain intact;
- candidate counts remain bounded by the current transition-expansion limit;
- direct evidence wins deterministic ties;
- missing Chroma data, vector errors, or insufficient neighbours fall back to direct transitions without failing the request.

The LLM context may describe smoothed-only candidates as destinations inferred from career histories of semantically related roles. It must retain the existing population-level caution and distinguish this from a direct source-to-destination observation.

If the gate fails, no runtime integration or response-schema change is made.

## 8. Components and boundaries

### `src/transition_embedding.py`

A focused, runtime-safe module responsible for:

- extracting training transition distributions from a `MultiDiGraph`;
- validating and normalizing live ESCO embeddings;
- computing nearest transition-source roles;
- producing deterministic hybrid destination rankings;
- exposing no person-level data and making no embedding API calls.

### `evaluation/evaluate_embedding_transitions.py`

An offline experiment command responsible for:

- loading the graph, Chroma vectors, validation, and test aggregates;
- reproducing the direct baseline;
- tuning only against validation;
- applying the frozen configuration to test once;
- enforcing the acceptance gate;
- writing a JSON report under `artifacts/karrierewege/`.

### `evaluation/transition_metrics.py`

Extend the existing evaluator with a prediction-map entry point so direct and hybrid rankings use exactly the same metric calculations. Preserve the training-split rejection guard.

### Conditional files

Only after the gate passes:

- `config.py` receives bounded smoothing settings/defaults;
- `src/inference_pipeline.py` consumes the smoothed backoff candidates;
- `HANDOFF.md` and `ALL_STEPS.md` record accepted metrics and behavior.

## 9. Error handling

- Missing or incompatible Chroma data: report the experiment as not runnable; runtime direct behavior remains available.
- Missing source vectors: use direct predictions for that source.
- No eligible neighbours: use direct predictions.
- Non-finite or dimensionally inconsistent vectors: reject those vectors and report counts.
- Empty candidate distribution: return no smoothed candidates.
- Validation/test mapping failures: stop evaluation rather than silently dropping titles.
- A failed acceptance gate is a valid experiment result, not an implementation error.

## 10. Testing

Unit tests must cover:

- normalized transition distributions;
- exclusion of self and non-ESCO/stale roles;
- deterministic neighbour and destination ties;
- missing-vector and no-neighbour fallback;
- direct-only, neighbour-only, and hybrid scoring;
- test/training leakage guards;
- metric equivalence between the refactored direct path and the existing baseline;
- acceptance and rejection decisions at threshold boundaries;
- JSON-safe experiment output;
- conditional inference fallback if runtime integration is accepted.

Verification commands:

```powershell
python -m compileall -q src evaluation config.py career_kg_web.py
python -m unittest discover -s tests -v
python evaluation/evaluate_embedding_transitions.py
```

The evaluation command may read configured local Chroma and graph assets but must not mutate them. If Chroma's client cannot provide a read-only open, evaluation must work against a temporary copy.

## 11. Privacy, deployment, and reproducibility

- No `_id` values, person trajectories, raw rows, or person-authored text enter vectors, graph attributes, reports, or Git.
- Only aggregate training transitions and taxonomy role vectors participate in scoring.
- Validation/test remain evaluation-only.
- Graph, Chroma, data files, and evaluation artifacts remain local and ignored by Git/Vercel.
- Runtime source, design, and aggregate metrics are versioned on `dev`. Raw-data evaluation code and tests remain local under the repository's deployment-minimal ignore policy.
- External graph/vector storage remains the deployment boundary for Vercel.

## 12. Completion criteria

The experiment is complete when:

1. direct baseline equivalence is verified;
2. validation tuning is deterministic;
3. the frozen test evaluation is recorded;
4. the acceptance gate produces an explicit decision;
5. runtime behavior changes only for an accepted result;
6. tests and syntax checks pass;
7. `HANDOFF.md` and `ALL_STEPS.md` document the outcome;
8. accepted source/documentation changes are committed and pushed to `dev`.

## 13. Measured outcome

Validation selected `k=20`, direct weight `0.90`, and softmax temperature `0.05` from the pre-registered 48-configuration grid. The frozen configuration passed the locked test gate:

| Metric | Direct baseline | Hybrid | Absolute change |
|---|---:|---:|---:|
| MRR | 0.259056 | 0.261703 | +0.002647 |
| Hits@5 | 0.370239 | 0.372183 | +0.001944 |
| Hits@10 | 0.501359 | 0.505809 | +0.004450 |
| Source-role coverage | 0.738878 | 1.000000 | +0.261122 |
| Destination coverage | 0.871500 | 0.953196 | +0.081697 |

All 3,039 live ESCO role vectors loaded successfully at 3,072 dimensions. Runtime initialization uses only the 765 vectorized source roles with retained training transitions, and role-specific rankings are cached. No raw Karrierewege row was embedded and no embedding API call was made during evaluation.
