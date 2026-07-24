# Documentation Updates

- [x] Added `docs/Features/slot_exhaustion_policies.md` with the shipped CPU
  planning and direct-Warp primitive contracts, imports, sidecar schemas,
  diagnostics, mutation boundaries, and deferred scope.
- [x] Updated `docs/Features/Roadmap/data-oriented-gpu.md`,
  `docs/Theory/Technical/Dynamics/Nucleation_Equations.md`, and `AGENTS.md` to
  distinguish delivered primitives from deferred discovery, activation,
  source/gas handling, policy composition, and integrated loops.
- [x] Updated `.opencode/guides/architecture/architecture_outline.md` for the
  shipped P4 direct CPU/Warp scaling boundary, caller-owned sidecars,
  concrete-only exports, and its deferred policy/transfer/runnable scope.
- Updated E6/E6-F5/E6-F7/E6-F8/E6-F9 plan cross-references where final API
  names or diagnostics differed from this first pass.

Documentation states that the CPU resolver defaults to resampling on and volume
scaling off; sufficiently releasable resampling wins before enabled scaling;
no viable policy raises before a plan returns; and no demand is silently
discarded. These are planning-selection rules, not a runtime policy loop.

P5 runtime composition remains blocked on E6-F5. P7 is documentation-only and
does not establish a public policy API or add a pytest module.

Issue #1427 (P6) required no public documentation update: it adds only
conservation validation in existing CPU and Warp test modules and changes no
production or public API behavior.

P7 validation:

```bash
pytest particula/particles/tests/exhaustion_test.py \
  particula/gpu/kernels/tests/exhaustion_test.py -q -Werror
mkdocs build --strict
```
