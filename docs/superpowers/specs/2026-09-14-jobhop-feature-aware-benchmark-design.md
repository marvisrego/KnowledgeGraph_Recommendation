# JobHop Feature-Aware Full-Candidate Benchmark

## Goal

Create a strictly offline benchmark that measures whether information present
for each JobHop career prefix--role tenure, time since the prior role, and
education--materially improves next-occupation ranking.  The benchmark is a
separate research result.  It does not alter the Karrierewege dataset,
production ranker, graph evidence, or promotion status.

## Alternatives considered

1. Continue tuning the Karrierewege role-only model.  The recent JobHop
   pretraining result improved held-out validation, but its remaining gap to
   60 percent Hit@10 is unlikely to close without additional per-person input.
2. Use NLSY97 as the next benchmark.  It has rich longitudinal information,
   but requires ambiguous Census-to-SOC-to-ESCO mapping and is therefore a
   poor direct full-ESCO evaluation source.
3. **Chosen: feature-aware JobHop benchmark.**  JobHop has official,
   person-disjoint partitions plus ESCO role IDs, chronological quarters, and
   education.  This permits a full-candidate measurement with inputs that are
   observed at prediction time.

## Data boundary and prefix construction

- Read only the already-cleaned `Data/JobHop_v2/processed/{train,val,test}.parquet`
  files and reject missing columns, unknown role IDs, overlapping people, or
  non-finite feature values.
- Use `train` to fit model parameters, `val` to select a single configuration,
  and leave `test` untouched until that configuration is frozen in a saved
  selection manifest.
- Sort a person's records by `order_quarter`, then use the existing conservative
  rule: skip same-quarter destinations and consecutive duplicate roles.
- For a prefix ending at record `i`, the model may use roles through `i`, the
  final role's elapsed quarters (`end_quarter - start_quarter`, clipped and
  bucketed), and the gap from the preceding role's end to its start (clipped
  and bucketed). Static résumé education is limited to a separately labelled
  ablation because JobHop provides no education-completion dates.
- It must never use the target role, its dates, later education, later roles,
  a person identifier, or a split label as a model feature.  Missing end dates
  are represented by an explicit missing-tenure bucket, not imputed from the
  target start date.

## Model and comparisons

The new offline `JobHopFeatureRanker` extends the causal role-history encoder.
It adds learned embeddings for tenure bucket, prior-gap bucket, and five-level
education to the final contextual representation before scoring the fixed live
ESCO candidate matrix.  The current role remains excluded from all candidate
rankings.

Validation compares three models under identical full-candidate evaluation:

1. role-history-only causal Transformer;
2. feature-aware Transformer using dated tenure and prior-gap features;
3. an explicitly labelled static-résumé-education ablation.

The primary JobHop metric is option 2. The education field is invariant within
each résumé and has no completion date, so it is not represented as known at a
historical prefix in the primary result.

The selected model is ordered by MRR, Hit@5, and Hit@10.  No parameter grid,
test peeking, or fusion search is permitted after test evaluation.

## Reporting and safety

Reports include MRR, Hit@1/3/5/10, NDCG@10, full 3,039-role candidate coverage,
examples, people, a person-cluster bootstrap 95 percent interval, feature
availability, split identities, and a plain statement that the result is
JobHop-specific.  Artifacts are `promoted=false` and no production flag is
changed.

## Tests

- Unit tests cover chronological prefix construction, same-quarter and
  duplicate handling, exact feature bucketing, missing-end handling, and that
  changing a valid observed feature changes logits without altering candidate
  shape or current-role exclusion.
- Tests reject target-derived feature extraction, malformed artifacts, role-ID
  mismatch, and overlapping official people.
- Focused tests, complete unittest discovery, compilation, `pip check`, and
  `git diff --check` are required before reporting an experiment.
