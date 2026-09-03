# Architecture Design

## Current Workflow Status

P1 (issue #1567) is directly blocked by E8-F3's absent capture-resource carrier
and contract. P2 (issue #1568) is directly blocked by absent P1; E8-F3 is its
transitive prerequisite. P3 (issue #1569) is blocked until P2 provides the
native captured-plan owner, opaque graph handle, and `capture_launch()` runtime
adapter. No E8-F4 architecture was implemented and no resource or adapter API
was inferred or fabricated.

P4 (issue #1570) remains unimplemented until P1/P2/P3 land, specifically until
#1569 provides the P3 native capture-owner/replay seam and this branch is
rebased onto it. The following describes only the future P4 architecture.

## High-Level Design

E8-F4 would add a concrete graph owner around the prepared enqueue path; it
would not create a second scheduler. Planned capture and replay would share the
same immutable prepared plan and exact E8-F3 arrays. Host validation would stay
outside capture, and planned replay would do only bounded metadata checks,
token bookkeeping, and one graph launch.

```text
Planned capture/replay sequence after the P1/P2/P3 gate is satisfied:

ACTIVE session + pinned registry + closed guard
E8-F1 READY record/signature
E8-F2 PreparedResidentTimestep
E8-F3 CaptureResourceSet
                 |
                 v
prepare_capture() -- exact identity/capability validation (host, read-only)
                 |
                 v
wp.capture_begin(cuda, force_module_load=True)
  enqueue_prepared_timestep()  # fixed twelve-node device sequence only
wp.capture_end() -> opaque graph
                 |
                 v
CapturedResidentPlan(REPLAYABLE, exact binding + graph)
                 |
        replay(duration)
  compare current signature/identities/lifecycle
  begin one resident token
  wp.capture_launch(graph)
  complete token
                 |
       +---------+----------+
       |                    |
 structural drift       launch may fail
       v                    v
  INVALIDATED          graph + session FAULTED
       |                    |
 explicit retire and fresh compatible preparation only
       v
  new READY record -> explicit recapture
```

The planned capture design would use an internal runtime adapter so
hardware-independent tests could assert call order and cleanup. Its production
resolution would require callable Warp `capture_begin`, `capture_end`, and
`capture_launch`, a qualified CUDA device, and E8-F1 capability approval. If a
future enqueue or capture-end failed, cleanup would consume the active capture
once and publish no handle. If cleanup also failed, both errors would remain
visible.

Planned replay would compare metadata before token entry and launch. Mutable
payloads and RNG words would intentionally not be compared because they would
advance in the exact pinned arrays. Any shape, device, request, schedule,
process configuration, communication map, diagnostic binding, or resource
identity drift would deterministically invalidate the record and reject launch.
A graph-launch failure would be writer-capable and fault both graph and resident
session; rollback and retry would not be promised.

## Data / API / Workflow Changes

- **Data model:** Plan to add exact immutable or narrowly stateful
  concrete-only records
  for the native capture adapter, captured resident plan, opaque graph handle,
  and teardown result. Reuse E8-F1 lifecycle/signature and invalidation reason
  types instead of creating parallel vocabulary.
- **API surface:** Plan direct-module-only setup, replay, invalidate, and
  teardown
  operations under `particula.execution.graph_capture`. Do not export them from
  `particula.execution`, `particula.gpu.kernels`, or top-level `particula`.
- **Workflow hooks:** Planned E8-F4 would consume E8-F1 lifecycle/signature,
  E8-F2 prepared
  enqueue, and E8-F3 resource publication. E8-F5 validates the accepted replay
  path; E8-F6 benchmarks its resource/graph lifetime; E8-F7 profiles the
  correctness-qualified path; and E8-F8 documents and closes the epic.
- **Lifecycle:** Planned capture would require exact ACTIVE/READY/closed
  bindings, and planned replay would require REPLAYABLE. Finalize, close, fault,
  explicit teardown, or structural drift would prevent launch. Planned recapture
  would always create a fresh record and native handle; old handles would never
  be checkpointed or resurrected.

## Planned Security & Compliance

The planned design would introduce no network, credential, persistence, or
permission boundary changes. Its key robustness control would be fail-closed
native-handle use: opaque handles would never cross devices, checkpoints,
sessions, or restarts, and stale or terminal records could not launch. Planned
validation would expose identities and metadata only, never pointers, payloads,
or RNG words. Exceptions would not trigger fallback, migration, retry, or
automatic recapture, and post-launch failures would retain the no-rollback
contract.
