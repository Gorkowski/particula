# Phase Details

## Sequencing

E6-F5 is required. Complete P1 through P5 in order before P6 conservation
validation; P7 documents only the validated policy, precedence, and bounds.

- [x] **E6-F6-P1:** Freeze exhaustion policy, precedence, and conservation contracts with unit tests
  - Issue: #1422 | Size: S | Status: Implemented (2026-07-24)
  - Goal: Delivered strict defaults, independent controls, all-box validation
    before resolution, exact planning diagnostics, and bounded/deferred
    conservation language.
  - Files: `particula/particles/exhaustion.py`, `particula/particles/tests/exhaustion_test.py`
  - Tests: Focused unit coverage verifies config defaults and immutability,
    policy truth table and resampling-first precedence, exact activation
    prefixes/deferred sentinels, malformed sidecars, later-invalid-box
    no-mutation, empty dimensions, and a float64 multi-box/species inventory
    oracle at `rtol=1e-12`, `atol=1e-30`.
  - Boundary: P1 does not commit, choose releases, resample, assess scaling
    feasibility, discover slots, add re-exports, or add GPU behavior.

- [x] **E6-F6-P2:** Implement deterministic CPU resampling reference with unit tests
  - Issue: #1423 | Size: S | Status: Implemented (2026-07-24)
  - Goal: Delivered immutable detached CPU equal-weight remap plans and an
    all-box-preflighted commit that frees the P1-required active slots while
    validating conservation and named distribution/mixing diagnostics.
  - Files: `particula/particles/exhaustion.py`,
    `particula/particles/tests/exhaustion_test.py`
  - Tests: Co-located tests cover deterministic stable ties, independent
    interval-sweep conservation oracle, sparse/full/multi-box/zero-release
    remaps, literal released-slot clearing, strict input/bound validation, and
    stale or later-box-invalid plan no-mutation behavior.
  - Boundary: P2 is CPU-only, fixed-capacity, and plan-then-commit; it adds no
    package export, scaling, slot discovery/activation, GPU parity, or resize.

- [x] **E6-F6-P3:** Implement allocation-stable Warp resampling with parity tests
  - Issue: #1424 | Size: S | Status: Implemented (2026-07-24)
  - Goal: Delivered a direct explicit-release-count Warp remapping boundary
    with caller-owned active-device buffers, staged deterministic planning, and
    one gated commit.
  - Files: `particula/gpu/kernels/exhaustion.py`, `particula/gpu/kernels/tests/exhaustion_test.py`, `particula/gpu/kernels/__init__.py`
  - Tests: Focused tests cover Warp CPU deterministic parity, supplied-buffer
    ownership, shape/dtype/device/value/nonaliasing validation, diagnostics,
    failed-planning no-commit behavior, exported entry-point resolution, and
    optional CUDA clean skips.
  - Boundary: P3 does not add scaling, P1 policy resolution, capacity discovery
    or activation, resizing, hidden transfer/fallback, or a runnable API.

- [x] **E6-F6-P4:** Add optional CPU and Warp representative-volume scaling with tests
  - Issue: #1425 | Size: S | Status: Implemented (2026-07-24)
  - Goal: Delivered opt-in direct, all-box-preflighted scaling of representative
    volume, concentration/raw weight, and provisional source demand under
    caller-supplied bounds, preserving intensive concentrations.
  - Files: `particula/particles/exhaustion.py`,
    `particula/gpu/kernels/exhaustion.py`, corresponding exhaustion tests,
    `particula/gpu/tests/kernel_exports_test.py`, and
    `.opencode/guides/architecture/architecture_outline.md`.
  - Tests: CPU and independent-NumPy Warp CPU parity cover selected and
    unselected rows, sidecar diagnostics, protected state, identities,
    conservation, schema/value/bound/alias rejection, and atomicity; guarded
    CUDA coverage skips cleanly when unavailable. Export tests retain both APIs
    as concrete-module-only.
  - Boundary: P4 consumes neither P1 plans nor exhaustion-policy controls and
    adds no resampling, activation, source packaging, resize, transfer, fallback,
    or runnable API.

- [ ] **E6-F6-P5:** Enforce independent controls, resampling-first precedence, and fail-closed validation
  - Issue: #1426 | Size: S | Status: Blocked pending E6-F5
  - Blocker: The authoritative E6-F5 discovery-to-activation boundary is not
    present in this worktree. P5 must not duplicate discovery, free-slot
    classification, activation, or a parallel orchestration boundary.
  - Goal: Compose E6-F5 discovery with exhaustion planning so resampling is attempted first and scaling is only a configured fallback.
  - Files: CPU/GPU exhaustion and slot-management modules plus their tests
  - Tests: Four control combinations, resampling-sufficient and fallback cases, both-off pre-mutation error, unsatisfiable request, diagnostics, and no silent truncation.

- [ ] **E6-F6-P6:** Validate sparse, full, and over-capacity conservation across CPU and Warp
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Prove cross-backend behavior with an independent NumPy oracle and downstream-shaped source requests.
  - Files: `particula/particles/tests/exhaustion_test.py`, `particula/gpu/kernels/tests/exhaustion_test.py`, integration fixtures
  - Tests: Multi-box/species matrix, physical inventory before/after plus admitted demand, moment/error bounds, repeated calls, exact no-ops, and CUDA clean skips.

- [ ] **E6-F6-P7:** Update development documentation for slot exhaustion policies
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Publish defaults, precedence, equations, diagnostics, direct imports, failure boundaries, dependencies, and deferred capabilities.
  - Files: `AGENTS.md`, `docs/Features/`, `docs/Theory/Technical/Dynamics/Nucleation_Equations.md`, E6 plan sections
  - Tests: Markdown links, API snippets, shape/equation review, terminology, and focused commands.
