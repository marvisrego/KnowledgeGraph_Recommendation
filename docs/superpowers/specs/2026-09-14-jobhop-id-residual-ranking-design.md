# JobHop-Pretrained ID-Residual Career Ranker

## Goal

Test a materially stronger, offline-only full-candidate career ranker that can
learn occupation-specific transition behaviour while retaining ESCO semantic
generalisability. The target is a validation result at or above 0.60 Hit@10;
no result is promoted without the existing frozen-test gate.

## Alternatives considered

1. Pretrain the current frozen-semantic Transformer on JobHop. This reuses
   existing code but cannot learn a destination-specific bias or a role-ID
   transition residual.
2. **Chosen: semantic-plus-ID residual causal Transformer.** It retains frozen
   ESCO vectors as a prior and adds trainable input and output role embeddings
   plus an output bias. It can learn the high-support directional structure
   missing from the fixed-vector scorer, while the semantic prior remains
   available for sparse roles.
3. Add tenure, education, employer, or location inputs. Karrierewege does not
   contain person-level versions of those fields, so this cannot be deployed
   honestly for this benchmark.

## Data and leakage boundary

- Pretrain only on `Data/JobHop_v2/processed/train.parquet`; never read its
  test partition. JobHop role IDs must be checked against the artifact ESCO
  candidate IDs before training.
- Fine-tune only on Karrierewege `train` prefixes. Validation is used for
  early stopping and model selection; test remains untouched until a model,
  seed, and fusion policy are frozen.
- The model accepts only ordered role IDs. It does not receive person IDs,
  targets, future steps, split labels, dates, education, or any persisted
  inferred transitions.
- It writes only ignored local NPZ/JSON research artifacts. It cannot change
  AuraDB, Qdrant, observed `TRANSITIONS_TO` edges, or
  `SEQUENTIAL_RANKING_ENABLED`.

## Architecture

For every occupation, retain its normalized frozen ESCO vector `e`. Learn an
input ID residual `r_in`, output ID residual `r_out`, and output bias `b`.
The causal Transformer receives `normalize(project(e) + r_in)` and scores
each candidate from its contextual state against `normalize(project(e) +
r_out) + b`. The final current role is always assigned negative infinity.

The trainer adds an explicit `causal_transformer_id_residual` model type. Its
artifact contains the frozen candidate vectors and all learned residuals. The
offline evaluator reconstructs this exact model and rejects missing or
unexpected parameters. Existing model types and artifacts remain compatible.

## Training and selection

1. Verify JobHop role-ID overlap and construct only chronological,
   deduplicated prefixes from its training partition.
2. Pretrain for the configured number of epochs, then fine-tune on
   Karrierewege train with full 3,039-candidate cross-entropy.
3. Evaluate every epoch on Karrierewege validation and retain the checkpoint
   ordered by MRR, Hit@5, then Hit@10.
4. Compare the new artifact with the semantic-only causal Transformer and the
   0.541686 validation fusion. Only if it reaches the 0.60 validation
   milestone will support-bucket fusion be designed and frozen for a test run.

## Tests and failure handling

- Unit-test output shape, current-role exclusion, and that nonzero ID
  residuals alter scores while preserving finite full-candidate logits.
- Unit-test artifact loading for the new type and rejection of malformed
  residual parameters.
- Fail before training if JobHop IDs do not align, embeddings are invalid, or
  PyTorch/read-only research resources are unavailable.
- Run focused tests, full unittest discovery, compilation, `pip check`, and
  `git diff --check`. An unsuccessful result remains an unpromoted artifact.
