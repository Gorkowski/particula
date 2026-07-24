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
- [ ] Implement deterministic resampling selection, moment accounting, and one
  commit boundary without resizing.
- [ ] Implement bounded same-direction representative-volume, raw-weight, and
  source-demand scaling.
- [x] Keep the P1 API concrete-only; no `particula.particles` re-export was
  added.

## Direct Warp

- [ ] Mirror the CPU resolver and plan representation with same-device,
  fixed-shape `wp.int32`/`wp.float64` caller-owned sidecars.
- [ ] Add allocation-stable validation, resampling, scaling, and commit kernels
  in `particula/gpu/kernels/exhaustion.py`.
- [ ] Reject bad shape, dtype, device, values, capacity, controls, or scratch
  before clearing outputs, launching mutation, or changing volume.
- [ ] Keep concrete configuration/scratch APIs out of broad exports unless an
  existing direct-step convention requires them.

## Tooling / Tests

- [x] Add `particula/particles/tests/exhaustion_test.py` covering strict frozen
  controls and records, all-box validation/no-mutation, policy precedence,
  empty dimensions, and an independent float64 weighted-inventory oracle.
- [ ] Add `particula/gpu/kernels/tests/exhaustion_test.py` for Warp CPU parity,
  optional CUDA, supplied identity, and invalid-call snapshots.
- [ ] Cover capacity-sufficient no-op, sparse, full, repeated, and demand larger
  than releasable capacity cases for every policy combination.
- [ ] Run focused tests, full fast regressions, Ruff, mypy, and docs validation
  without reducing the repository coverage threshold.
