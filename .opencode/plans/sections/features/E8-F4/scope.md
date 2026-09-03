# Scope

## Current Workflow Status

P1 (issue #1567) is directly blocked: the E8-F3 capture-resource carrier and
its governing contract are absent. P2 (issue #1568) is directly blocked by
absent P1, with E8-F3 only a transitive prerequisite. P3 (issue #1569) is
blocked until P2 provides the native captured-plan owner, opaque graph handle,
and `capture_launch()` runtime adapter. This workflow made no implementation,
test, or user documentation changes. The deferred P1/P2 scope remains pending
until E8-F3 enables P1 and P1 then unblocks P2.

P4 (issue #1570) is likewise unimplemented and blocked until P1/P2/P3 land,
including #1569's P3 native capture-owner/replay seam, and this branch is
rebased onto that concrete boundary. The following scope is future design only:
E8-F4 would implement a runtime owner that consumes the exact E8-F2 prepared
resident timestep and E8-F3 capture resource set, records its predetermined
device sequence into a Warp graph, and replays it only while the E8-F1
compatibility and lifecycle contracts remain valid.

## In Scope

- A planned concrete-only captured-plan/controller record retaining the exact
  prepared plan, compatibility signature, capture resource set, native device,
  opaque graph handle, and lifecycle state.
- Planned explicit `capture_begin` / prepared enqueue / `capture_end` setup on
  qualified CUDA devices, with cleanup that consumes an active capture exactly
  once.
- Planned replay preflight for exact session, registry, guard, prepared plan,
  signature, schedule, resource-set, graph-handle, device, and lifecycle
  identity.
- A planned graph launch per accepted timestep under resident token and
  writer-failure semantics; no validation, allocation, readback, or host
  scheduling inside the captured sequence.
- Planned deterministic invalidation, fault, close, teardown, and explicit
  recapture eligibility behavior for structural and resident lifecycle changes.
- Planned multi-timestep three-way validation across CPU reference, uncaptured
  Warp, and captured CUDA for identical supported process configurations.
- Planned co-located unit, lifecycle, integration, CUDA pass-or-clean-skip,
  export, and documentation-contract tests.

## Out of Scope

- Dynamic shapes, process-order selection, communication-map replacement, or
  resource replacement during replay.
- Automatic recapture, CPU fallback, retry, migration, hidden transfer,
  synchronization, checkpoint serialization of graph handles, or rollback.
- Graph capture on Warp CPU; Warp CPU remains the uncaptured contract baseline.
- New process physics, resizing, compaction, distributed execution, autodiff,
  performance targets, memory-budget publication, examples, or profiling; those
  belong to E8-F5 through E8-F8.
- Public exports from `particula.execution`, `particula.gpu.kernels`, or the
  top-level package.
