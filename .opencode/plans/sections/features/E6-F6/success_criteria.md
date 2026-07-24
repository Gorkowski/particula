# Success Criteria

- [ ] E6-F5 remains the mandatory and only slot-classification/activation
  dependency; E6-F6 does not duplicate or weaken its predicates.
- [ ] Resampling defaults to enabled and representative-volume scaling defaults
  to disabled; controls remain independently selectable.
- [ ] When both are enabled and exhaustion occurs, resampling is planned first;
  scaling runs only if planned resampling cannot satisfy all demand.
- [ ] Capacity-sufficient calls bypass exhaustion mutation regardless of policy
  switches; both switches off with exhausted demand raises before every write.
- [ ] Invalid or unsatisfiable calls preserve particle fields, volume, request,
  diagnostics, work buffers, identities, and any caller RNG state.
- [ ] No represented demand is truncated: with scale `s`, represented demand is
  exactly `s * gas_admitted_demand`; otherwise the entire call fails closed.
- [ ] Per-box post represented number, every species mass, and charge equal
  `s * pre_state + represented_demand` at recorded float64 tolerances. Existing
  and source intensive concentrations are unchanged by representation scaling.
- [ ] The future resampling phase (P2+) satisfies the named
  radius/composition distribution-moment bounds and deterministic tie-break
  rules; read-only P1 does not claim to deliver those bounds.
- [ ] CPU and Warp CPU agree on plans, policy diagnostics, and resulting state;
  CUDA executes the same matrix when available and skips cleanly otherwise.
- [ ] Arrays remain fixed shape and preserve container/array identity, dtype,
  device, density, requests, and untouched boxes/slots.
- [ ] Focused/full tests, Ruff, mypy, and documentation validation pass without
  lowering coverage.

## Metrics

| Metric | Baseline | Target | Source |
|--------|----------|--------|--------|
| Silently dropped represented demand | Undefined | 0 | Policy diagnostics and oracle |
| Number/species-mass/charge conservation failures | No policy | 0 | Multi-box conservation matrix |
| CPU/Warp plan or state mismatches | No shared API | 0 | Parity tests |
| Invalid exhausted calls with observable writes | E6-F5 rejects capacity only | 0 | Snapshot tests |
| Dynamic resizes/hidden transfers | Out of scope | 0 | Identity tests and review |
| Distribution-moment violations | Undefined | 0 under future P2+ resampling bounds | Independent NumPy oracle |
| Changed-code coverage | Repository threshold | At least 80% | pytest-cov |

## Delivered P1 Evidence (#1422)

- [x] The concrete CPU module exposes strict frozen controls, fixed-shape
  sidecars, immutable plans, policy constants, resolver, and float64
  tuple-backed inventory only at `particula.particles.exhaustion`.
- [x] Resolver tests establish validate-all-before-resolve behavior and preserve
  every supplied sidecar on successful, invalid, and fail-closed calls.
- [x] Focused tests establish capacity activation independent of flags and
  resampling-first selection with deferred releases/scaling; later commit,
  resampling, scaling, discovery, GPU, and export criteria remain open.
