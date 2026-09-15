# Ranking and TES Improvement Handoff

**Updated:** 2026-09-15
**Purpose:** Continue empirical transition-ranking and Transition Effort Score
(TES) improvement work in a new session without losing the evidence, safety
boundaries, or current results.

## Executive summary

The earlier production transition ranker is an embedding-smoothed one-step
model. Its held-out test result is:

| Method | MRR | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Source coverage |
|---|---:|---:|---:|---:|---:|---:|
| Direct train transitions | 0.259056 | 0.147830 | 0.284979 | 0.370239 | 0.501359 | 0.738878 |
| Accepted embedding smoother | 0.261805 | 0.148318 | 0.285776 | 0.372273 | 0.505849 | 1.000000 |

The LightGBM link-prediction classifier must remain **coverage backfill only**:
despite strong sampled-negative diagnostics, its held-out ranking result is
MRR `0.0182`, Hit@5 `0.0274`, and Hit@10 `0.0447`.

## 2026-09-14 update: causal Transformer result and remaining decision gate

The strongest completed neural candidate is a causal Transformer with a
learned occupation-ID residual. It is pretrained only on JobHop v2 training
sequences and fine-tuned only on Karrierewege training sequences. The primary
benchmark remains person-disjoint Karrierewege prefixes, the full set of 3,039
live ESCO roles, current-role exclusion, and train-only evidence.

| Split / method | MRR | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| Validation: selected causal ID-residual, seed 41 | 0.310251 | 0.189872 | 0.345993 | 0.434165 | 0.564143 | 1.000000 |
| Validation: checksum-frozen fusion, seed 41 | 0.312526 | 0.191985 | 0.351206 | 0.435962 | 0.565819 | 1.000000 |
| Held-out test: checksum-frozen fusion, seed 41 | 0.313438 | 0.194219 | 0.351234 | 0.435648 | 0.564807 | 1.000000 |
| Held-out test: semantic smoother | 0.261841 | 0.148318 | 0.285743 | 0.372281 | 0.505858 | 1.000000 |

The validation fusion used weights `(neural, smoother, second-order) =
(0.5, 0.5, 0.0)` for single-role histories and `(1.0, 0.0, 0.0)` for
multi-role histories. These weights are validation-selected; they are not to
be adjusted after reading test results.

Completed seeds 17, 29, 41, and 53 were selected by validation MRR, then
Hit@5, then Hit@10. Seed 41 won; its MRR `0.310251` was the maximum and the
four-seed MRR standard deviation was `0.000113`. The planned fifth seed 71 was
stopped before completion at the user's direction, so this is a four-seed
stability record, not a five-seed claim.

The seed-17 standalone test was observed before this selection completed and
remains only a diagnostic. Seed 41's weights were checksum-bound on validation
before its test split was read once. Its test improvement over the semantic
smoother was MRR `+0.051597` (person-bootstrap 95% CI `[0.050245, 0.052923]`),
Hit@5 `+0.063367`, and Hit@10 `+0.058950` (CI `[0.057016, 0.060774]`).

The final held-out fusion Hits@10 is **56.48%**, below the requested 60-65%
target. The JobHop feature-aware test benchmark (Hits@10 `0.318897`, MRR
`0.161371`) is a separate result and does not justify transferring JobHop
person-level dates, tenure, education, company/industry, location, or skills
into Karrierewege. Such fields must remain out of the production path.

The causal artifact is not promoted. `SEQUENTIAL_RANKING_ENABLED=false`
remains mandatory because the user target has not been reached. The unpromoted
20.7 MB seed-41 artifact loaded in `143.3 ms`; local top-50 ranking p50 was
`31.5 ms` and p95 `32.5 ms`. This is not end-to-end API evidence. Full
regression verification passed: 182 tests, `pip check`, and compilation.

## New measured improvement: ordered trajectory context

A new offline, train-only second-order transition experiment was added:

`evaluation/evaluate_variable_order_transitions.py`

It retains the existing transition policy but uses the previous role as
additional history when it has enough evidence:

\[
P(t\mid p,s) = \frac{c_{pst} + \alpha P(t\mid s)}{N_{ps}+\alpha},
\qquad
S(t)=(1-\lambda)P(t\mid s)+\lambda P(t\mid p,s).
\]

The model falls back to the one-step distribution `P(t|s)` for a single-role
history or an insufficiently supported two-role context. This is deliberately
not a generic KGE score and does not add inferred edges to AuraDB.

### Protocol

- Training counts: Karrierewege `train` split only.
- Selection: validation split only.
- Frozen evaluation: test split, after validation selection.
- Input cleaning: the same conservative streaming cleaner as the existing
  benchmark: contiguous experience orders only, duplicate-position handling,
  no self transitions, no validation/test training evidence.
- Evaluation unit: ordered career prefix. Unscored destinations receive zero
  credit, matching the existing direct transition benchmark's treatment of
  destinations with no train-derived transition evidence.
- This experiment is **not deployed** and is not yet promoted.

### Validation-selected configuration

| Parameter | Selected value |
|---|---:|
| Second-order mixture weight `lambda` | 0.25 |
| Empirical-Bayes prior strength `alpha` | 10 |
| Minimum two-role context support | 5 |

Validation has 123,520 examples; 53.93% have a usable two-role history.

### Locked test result

| Metric | Direct baseline | Second-order ranker | Delta |
|---|---:|---:|---:|
| MRR | 0.259056 | 0.278393 | +0.019336 |
| Hit@1 | 0.147830 | 0.161823 | +0.013993 |
| Hit@3 | 0.284979 | 0.312680 | +0.027701 |
| Hit@5 | 0.370239 | 0.399681 | +0.029442 |
| Hit@10 | 0.501359 | 0.529150 | +0.027791 |

The direct baseline reproduced exactly to the precision shown above, so the
new result is comparable to the historical direct benchmark. It also exceeds
the prior accepted smoother numerically (MRR +0.016588, Hit@5 +0.027409,
Hit@10 +0.023300), but it is **not yet a promotion result** because the
required head-to-head fusion, complete candidate evaluation, and person-level
bootstrap intervals have not yet been run.

Full report:

`artifacts/karrierewege/variable_order_evaluation.json`

## Interpretation of the supervisor/pasted recommendations

### Already implemented and should be preserved

- Entity title normalization, typed nodes and typed relations.
- Invalid/duplicate edge removal, isolation pruning, normalized-label collision
  reporting, and source-backed graph quality checks.
- Train-only transition evidence: a shared policy excludes validation/test
  transitions from graph edges, smoothing, TES, candidate construction, link
  prediction training, and explanations.
- ESCO role requirements and IDF-weighted skills; these matter for TES and
  explanations but do not by themselves predict the next occupation in the
  Karrierewege prefix benchmark.
- Semantic embedding smoothing for cold-start coverage.

### Highest-value path now

1. Integrate the validated second-order score as a portable transition-channel
   component.
2. Combine it with the accepted embedding smoother using validation-only
   weighted reciprocal-rank fusion, separately for single-role and multi-role
   histories.
3. Run complete live ESCO v1.2.1 candidate ranking and person-bootstrap 95%
   confidence intervals.
4. Promote only if the documented MRR/Hit@5/Hit@10/coverage gate passes.

### Do not treat as an improvement path for the primary benchmark

- **Filtered KGC scores:** report only as a KGE diagnostic. The deployed task
  is an unfiltered, full-candidate next-role ranking task; filtered metrics
  answer an easier question.
- **Sampled-negative AUC/AP:** classifier diagnostics only, not a ranking
  promotion gate.
- **Adding predicted links to AuraDB:** prohibited. Predictions remain
  request-local inferred evidence; persistent inferred edges can pollute the
  observed-transition evidence and create leakage/noise.
- **Graph growth for its own sake:** adding generic skills, vague relations, or
  low-quality nodes cannot improve this transition target without a validated
  scoring path.
- **Repeat LambdaMART unchanged:** it slightly raised Hit@10 but regressed MRR
  and Hit@5, so it was rejected.

### Secondary experiments, only after the trajectory fusion

- A 128-dimensional RotatE `TRANSITIONS_TO` benchmark with type/structure-aware
  hard negatives and self-adversarial sampling. Include it in fusion only if it
  improves validation beyond the history-plus-smoother baseline.
- The already scaffolded reduced MLP/GRU rankers with 768-dimensional
  outcome-independent PCA role representations. Use five seeds; pretrain with
  JobHop v2 only as external support and select models on Karrierewege
  validation.
- Do not describe the STEP preprint's published results as directly comparable
  unless its ESCO version and protocol are reproduced separately.

## Existing implementation from the empirical-TES/ranking plan

The following code is already present but has not yet produced a promoted
artifact:

| Area | Relevant files | Status |
|---|---|---|
| TES artifact validation and calibration | `src/tes_calibration.py`, `evaluation/calibrate_transition_effort.py` | Implemented; no calibration artifact generated |
| Missing-evidence TES handling and hierarchical ISCO distance | `src/transition_effort.py` | Implemented |
| Reliability-adaptive smoother capability | `src/transition_embedding.py`, `config.py` | Implemented; not validation-tuned yet |
| Ordered user career history | `src/career_history.py`, intent/retrieval integration | Implemented |
| Portable sequential runtime | `src/sequential_ranking.py` | Implemented but disabled without a promoted artifact |
| Prefix benchmark utilities | `src/trajectory_benchmark.py`, `evaluation/evaluate_prefix_ranking.py` | Implemented |
| Offline embedding export and MLP/GRU trainer | `evaluation/export_sequential_embeddings.py`, `evaluation/train_sequential_ranker.py` | Implemented; Qdrant/Aura export attempt was slow and no artifact was generated |
| Research-only RotatE trainer | `evaluation/train_rotate_link_prediction.py` | Implemented; not run |
| New second-order experiment | `evaluation/evaluate_variable_order_transitions.py` | Run successfully; report generated |

Runtime safety remains unchanged: `SEQUENTIAL_RANKING_ENABLED` is false and
the existing smoother stays production behavior until a promoted artifact is
available.

## Required promotion work

The new result is promising but it must complete these steps before deployment:

1. Extend the prefix benchmark so it scores every live ESCO v1.2.1 occupation
   and excludes only the current role under the no-self-transition policy.
2. Implement a production-safe variable-order runtime artifact built only from
   training data. It must remove consecutive duplicate history roles, cap
   explicit history at ten roles, and align its last resolved role to
   `current_role`.
3. Tune fusion of the second-order ranker and adaptive embedding smoother on
   validation only. Use a 0.25 simplex grid and separate single-role/multi-role
   weighting.
4. Compute test MRR, Hit@1/3/5/10, NDCG@10, coverage, and person-bootstrap 95%
   confidence intervals.
5. Apply the gate:
   - Test MRR and Hit@5 both exceed the adaptive-smoother baseline.
   - At least one improvement's person-bootstrap 95% interval excludes zero.
   - The other regresses by no more than `0.001`.
   - Hit@10 does not regress; coverage remains `1.0`.
   - Artifact loading, Vercel memory, and API latency integration tests pass.
6. If the gate fails, retain smoothing in production and record the trajectory
   model as a rejected experiment. Do not claim an improvement.

## Validation commands and safeguards

Run from the repository root with the project virtual environment:

```powershell
.\.venv\Scripts\python.exe evaluation\evaluate_variable_order_transitions.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
.\.venv\Scripts\python.exe -m py_compile evaluation\evaluate_variable_order_transitions.py
git diff --check
```

The variable-order experiment needs several minutes because it reads and
cleans 1.98M train rows using the same trajectory rules as production. Its
report is written atomically only after validation selection and the final test
evaluation.

Before any deployment, also run the full test suite, `pip check`, the frontend
build from `frontend` using `npm.cmd run build`, API contract tests, and the
full-candidate/person-bootstrap evaluation.

## Key files and data

- `HANDOFF.md`: broader project architecture and historical metrics.
- `ALL_STEPS.md`: project chronology; do not overwrite unrelated history.
- `docs/ALGORITHMS.md`: TES, smoothing, sequential-ranker pseudocode and
  equations.
- `artifacts/karrierewege/embedding_transition_fine_evaluation.json`: accepted
  semantic-smoother benchmark.
- `artifacts/karrierewege/variable_order_evaluation.json`: new measured
  second-order result.
- `Data/Karrierewege/`: raw person-disjoint train/validation/test career
  trajectories. Keep it for reproducibility.
- `Writing/`: explicitly out of scope; do not modify it.

## Research references used for methodology decisions

- Z. Sun et al., “A Re-evaluation of Knowledge Graph Completion Methods,” ACL
  2020. This supports avoiding sampled-negative or filtered KGC metrics as the
  primary deployment claim: https://aclanthology.org/2020.acl-main.489/
- Z. Sun et al., “RotatE: Knowledge Graph Embedding by Relational Rotation in
  Complex Space,” ICLR 2019. This supports the secondary directed-KGE
  experiment and self-adversarial negative sampling:
  https://arxiv.org/abs/1902.10197
- OECD, *Handbook on Constructing Composite Indicators*. This motivates
  empirical calibration, sensitivity analysis, and transparent TES weights:
  https://www.oecd.org/en/publications/handbook-on-constructing-composite-indicators-methodology-and-user-guide_9789264043466-en.html

## Working-tree caution

The worktree already contains many user/session changes. Preserve unrelated
edits and do not use destructive Git commands. The new variable-order script
is currently untracked; the generated report may be ignored by `.gitignore`.
No commit or deployment was requested. `Writing/` was not changed.
