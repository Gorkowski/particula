# Phase Details

All retained E8-F4 phase names and design descriptions below are
non-authoritative placeholders. P1 (issue #1567) is directly blocked until
E8-F3 supplies its integrated capture-resource carrier and contract. P2 (issue
#1568) is directly blocked by absent P1, so E8-F3 is only a transitive
prerequisite of P2; complete P1 before unblocking or performing P2.

- [ ] **E8-F4-P1:** Prepared resident graph capture controller with unit tests
  - Issue: #1567 | Size: S | Status: Blocked — E8-F3 capture-resource carrier and
    contract absent
  - Current outcome: No code, tests, or user documentation changed. Do not create
    a placeholder carrier or infer its ownership, adapter, or lifecycle contract.
  - Goal: Introduce an exact concrete-only graph owner and injectable Warp
    capture adapter that bind one E8-F2 prepared plan, E8-F3 resource set, and
    E8-F1 READY signature before native capture begins.
  - Files: `particula/execution/graph_capture.py`,
    `particula/execution/tests/graph_capture_test.py`
  - Tests: Exact carrier/runtime API checks, CUDA capability decisions, state
    transitions, opaque-handle ownership, one-time cleanup, and no exports.

- [ ] **E8-F4-P2:** Complete fixed-sequence capture and CUDA smoke tests
  - Issue: #1568 | Size: S | Status: Blocked — P1 absent (E8-F3 is a transitive
    prerequisite through P1)
  - Goal: Capture exactly one complete prepared twelve-node enqueue sequence
    with no setup work inside the capture window and publish the graph only
    after `capture_end` succeeds.
  - Files: `particula/execution/graph_capture.py`, E8-F2 prepared enqueue module,
    `particula/execution/tests/{graph_capture,full_loop}_test.py`
  - Tests: Fake launch traces, forbidden allocation/readback/synchronization
    spies, capture-begin/enqueue/end failure cleanup, and CUDA
    pass-or-clean-skip capture smoke coverage.

- [ ] **E8-F4-P3:** Guarded replay and exact compatibility checks with unit tests
  - Issue: #1569 | Size: S | Status: Blocked — E8-F4-P2 native-capture owner
    unavailable
  - Current outcome: No code, tests, or user documentation changed. The required
    opaque graph handle, captured prepared-plan owner, and `capture_launch()`
    runtime adapter did not land; do not create a replacement owner or replay
    boundary.
  - Goal: Validate the current exact resident/prepared/resource/signature
    binding before opening one timestep token and launching the captured graph
    exactly once without reseeding or host process dispatch.
  - Files: `particula/execution/graph_capture.py`,
    `particula/execution/gpu_session.py`,
    `particula/execution/tests/graph_capture_test.py`
  - Tests: Accepted repeated replay, every identity-drift category, duration and
    lifecycle rejection, token completion, persistent RNG advancement, and no
    launch on failed preflight.

- [ ] **E8-F4-P4:** Lifecycle invalidation fault and recapture handling with tests
  - Issue: Unassigned | Size: S | Status: Blocked — P3 has not completed
  - Goal: Connect structural invalidation, resident fault/finalize/close,
    post-launch failure, teardown, and explicit recapture eligibility to the
    E8-F1 state machine without automatic recapture or rollback.
  - Files: `particula/execution/graph_capture.py`, lifecycle integration points,
    `particula/execution/tests/{graph_capture,gpu_session,checkpoint}_test.py`
  - Tests: Deterministic invalidation reasons, idempotent teardown, stale-handle
    rejection, writer failure faulting, read-only rejection preservation,
    terminal-state behavior, and fresh-record-only recapture.

- [ ] **E8-F4-P5:** Three-way full-loop validation and development documentation
  - Issue: Unassigned | Size: S | Status: Blocked — P4 has not completed
  - Goal: Validate multiple identical process-sequence timesteps across CPU,
    uncaptured Warp, and captured CUDA, then document the implemented internal
    contract and downstream evidence handoff.
  - Files: `particula/execution/tests/captured_full_loop_test.py`, existing CPU
    oracle/integration fixtures, `docs/Features/Roadmap/data-oriented-gpu.md`,
    `docs/Features/data-containers-and-gpu-foundations.md`, `AGENTS.md`
  - Tests: Per-field deterministic tolerances, tight conservation, aggregate
    stochastic bounds, RNG continuation/reset, no hidden operations, Warp CPU
    uncaptured baseline, CUDA pass-or-clean-skip rows, docs contracts, and
    `mkdocs build --strict`.
