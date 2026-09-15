# Multi-Seed Fusion and Promotion Evidence

## Goal

Complete the remaining evidence for the fixed JobHop-pretrained
ID-residual causal ranker: five-seed validation stability, a frozen fusion
policy, a test-only fusion report with paired person bootstrap, and a
fail-closed promotion decision. Training remains local GPU work and is not
added to GitHub Actions.

## Selection and data boundary

- The fixed seed set is `17, 29, 41, 53, 71`; architecture, optimizer,
  JobHop pretraining epochs, Karrierewege fine-tuning epochs, batch size, and
  candidate matrix remain identical for every seed.
- Each run reads JobHop `train.parquet`, Karrierewege train, and Karrierewege
  validation only. No run reads Karrierewege test.
- Select exactly one seed lexicographically by validation MRR, Hit@5, and
  Hit@10. Persist a selection manifest before test-only fusion evaluation.
- Seed 17's standalone test report already exists. It is retained as an
  observed result but cannot be used for seed selection or described as a
  newly blind confirmation.

## Alternatives considered

1. Retrain the selected seed in GitHub Actions. Rejected: it makes CI costly
   and non-reproducible on ordinary runners.
2. Tune fusion on test. Rejected: that would invalidate its held-out claim.
3. **Chosen:** validation-only seed/fusion selection followed by a test-only
   evaluator that accepts the exact saved weights and has no search grid.

## Test-only fusion evaluator

- Read only the selected artifact, its immutable validation fusion manifest,
  train-derived direct/second-order counts, accepted train-derived smoother,
  and Karrierewege test prefixes.
- Score every artifact ESCO candidate, exclude the current role, and apply the
  saved single-role/multi-role RRF weights exactly.
- Emit overall metrics plus person-level paired bootstrap deltas against the
  accepted semantic smoother. The test evaluator never accepts weights from
  command-line arguments and cannot select/tune a configuration.

## Promotion decision

- Require all five validation results, documented deterministic selection,
  full candidate coverage, positive paired 95 percent interval for at least
  MRR or Hit@5, no material regression in the other gate metric, frozen fusion,
  runtime parity, CPU latency/memory checks, and API integration checks.
- A promotion manifest must explicitly disclose that seed 17's standalone test
  result was previously observed. If this prevents the review standard from
  accepting the evidence, keep the artifact unpromoted; no override exists.
- The promotion command is the sole writer of a `promoted=true` deployment
  artifact. This task does not set `SEQUENTIAL_RANKING_ENABLED=true`.

## Delivery

- Commit only the ranking/runtime sources, tests, promotion utilities, and
  relevant specifications. Do not add datasets, generated artifacts, `Writing/`,
  or unrelated existing changes.
- Run focused tests, full unittest discovery, compilation, dependency checks,
  and diff checks. Push the verified commit(s) to the existing `origin/dev`.
