# Implementation Tasks

## CPU Core

- [x] Define immutable policy configuration and exact precedence resolver in
  `particula/particles/exhaustion.py`.
- [x] Define frozen fixed-shape planning contracts for already-discovered
  capacity sidecars; no E6-F5 discovery, activation, or E6-F7 source records
  are implemented in P1.
- [x] Implement read-only all-box validation and deferred-policy planning.
- [x] Add float64 tuple-backed weighted number, species-mass, and charge
  inventories with documented later-commit accounting.
- [x] Implement deterministic stable interval-sweep resampling selection,
  conservation/moment accounting, and one all-box-preflighted commit boundary
  without resizing.
- [ ] Implement bounded same-direction representative-volume, raw-weight, and
  source-demand scaling.
- [x] Keep the P1 API concrete-only; no `particula.particles` re-export was
  added.

## Direct Warp

- [x] Add `resampling_step_gpu` and concrete-only `ResamplingBuffers` in
  `particula/gpu/kernels/exhaustion.py` for explicit release-count direct Warp
  remapping.
- [x] Implement allocation-stable read-only validation, device-resident staged
  bitonic-sort/interval-sweep planning, diagnostic gating, and a single commit.
- [x] Reject invalid schema, values, counts, bounds, devices, and aliased
  buffers before caller mutation; diagnostic planning failures skip commit.
- [x] Export only `resampling_step_gpu` through `particula.gpu.kernels`; keep
  `ResamplingBuffers` concrete-module-only.
- [ ] Add representative-volume scaling and policy/P1 resolution to a later
  direct Warp phase.

## Tooling / Tests

- [x] Add `particula/particles/tests/exhaustion_test.py` covering strict frozen
  controls and records, all-box validation/no-mutation, policy precedence,
  empty dimensions, and an independent float64 weighted-inventory oracle.
- [x] Extend the co-located CPU suite with deterministic detached P2 plans,
  fixed-capacity clearing, independent equal-stratum remap checks, diagnostic
  bound validation, and stale/later-box malformed-plan atomicity.
- [x] Add `particula/gpu/kernels/tests/exhaustion_test.py` for Warp CPU parity,
  optional CUDA, supplied-buffer ownership, diagnostics, and invalid-call
  snapshots.
- [ ] Cover capacity-sufficient no-op, sparse, full, repeated, and demand larger
  than releasable capacity cases for every policy combination.
- [ ] Run focused tests, full fast regressions, Ruff, mypy, and docs validation
  without reducing the repository coverage threshold.
