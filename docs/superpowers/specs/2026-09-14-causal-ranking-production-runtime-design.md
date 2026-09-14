# Causal Ranking Production Runtime

## Goal

Add a dependency-light inference path for a promoted causal Transformer
next-occupation artifact. Training remains offline-only. CI validates source,
artifact integrity, and latency; it never retrains the model. The currently
evaluated single-seed artifact remains unpromoted and must be refused.

## Alternatives considered

1. Add PyTorch to the Flask deployment and reuse the training model. This is
   quick to implement but adds a large runtime dependency and makes serverless
   cold starts and memory less predictable.
2. Export ONNX and add an ONNX runtime. This improves portability but adds a
   new binary deployment dependency and a separate conversion contract.
3. **Chosen: deterministic NumPy implementation.** The existing sequential
   runtime already uses NumPy and runs only a ten-role, two-layer Transformer
   against 3,039 candidate roles. Extending that runtime keeps deployment
   dependencies small and makes artifact validation explicit.

## Artifact promotion boundary

- Offline training writes only `promoted=false` artifacts.
- A dedicated promotion command accepts a source artifact plus an immutable
  promotion manifest. It validates the manifest's artifact SHA-256, frozen
  test report SHA-256, full-candidate policy, 100 percent coverage, measured
  latency/memory values, and all documented gates before writing a new
  promoted deployment artifact.
- The command never edits the source artifact. It fails closed on an absent,
  malformed, or non-passing manifest.
- This implementation does not create a manifest or promote the current
  single-seed artifact. `SEQUENTIAL_RANKING_ENABLED` stays false.

## Runtime

- `SequentialRankerRuntime` adds read-only support for
  `causal_transformer` and `causal_transformer_id_residual` metadata.
- It implements the exported PyTorch evaluation graph in NumPy: normalized
  candidate vectors, input projection, positional vectors, pre-layer
  normalization, causal multi-head self-attention, GELU feed-forward blocks,
  output projection, temperature, optional ID residuals/bias, and current-role
  exclusion.
- It caps resolved history to the artifact maximum and rejects incompatible
  tensor names, shapes, non-finite values, duplicate role IDs, and unknown
  model configuration.
- Existing MLP/GRU behavior and public response contracts remain unchanged.

## CI and release workflow

1. Pull request CI: unit tests, artifact-schema tests, deterministic parity
   tests against PyTorch, and a CPU latency/memory smoke test using a fixture.
2. Offline GPU workflow: train and evaluate candidate artifacts; save reports
   and checksums outside CI.
3. Promotion workflow: verify the independently reviewed manifest, generate a
   promoted deployment artifact, and run deployment latency checks.
4. Deployment: provide the promoted artifact through
   `SEQUENTIAL_RANKING_ARTIFACT_PATH`, then explicitly set
   `SEQUENTIAL_RANKING_ENABLED=true` after the service health test passes.

## Tests

- NumPy scores must match PyTorch scores/ranking for semantic and ID-residual
  causal fixtures within a fixed numerical tolerance.
- The runtime must reject the present unpromoted artifact and all malformed,
  incomplete, or unexpected causal tensors.
- The promotion command must reject changed source/report checksums, incomplete
  gates, coverage below one, failed latency/memory limits, and a nonzero
  production flag before promotion.
- Full unittest discovery, compilation, `pip check`, and `git diff --check`
  remain required. No GitHub Action performs model training.
