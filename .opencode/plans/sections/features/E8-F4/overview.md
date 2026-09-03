# Overview

## Current Status

**P4 (issue #1570) remains blocked on dependency #1569.** The P3 native
capture-owner/replay seam is unlanded in this worktree, so no P4 implementation
diff exists. Existing branch commits record the blocker; this status does not
claim shipped lifecycle invalidation, teardown, fault handling, or recapture.

**P3 (issue #1569) is blocked by the unavailable E8-F4-P2 native-capture
owner.** The required opaque graph handle, captured prepared-plan owner, and
`capture_launch()` runtime adapter did not land. No E8-F4 code, tests, or user
documentation was implemented or updated in this workflow. Resume P3 only after
the P2 owner is available; P3 must consume that owner by identity rather than
introducing a replacement capture boundary.

## Problem Statement

The resident scheduler currently validates and dispatches every timestep from
Python. Existing CUDA graph tests stop at a single condensation call and prove
that public wrappers contain host readbacks that cannot be captured. E8-F1,
E8-F2, and E8-F3 define the compatibility lifecycle, prepared enqueue plan, and
identity-stable resource set, but no owner yet captures that complete prepared
sequence, guards replay, or invalidates the opaque graph when its binding drifts.

## Planned Value Proposition

After P1, P2, and P3 land, including #1569's P3 native
capture-owner/replay seam, and this branch is rebased onto them, E8-F4 is
intended to turn the prepared resident plan into an executable, concrete-only
Warp graph boundary. Its planned setup would capture the authoritative fixed
twelve-node sequence once on a qualified CUDA device, and planned replay would
check exact lifecycle and compatibility metadata before launching the same
opaque graph handle. The planned design would invalidate or fault the graph for
structural drift, terminal resident state, and post-launch failures without
hidden recapture, CPU fallback, transfer, synchronization, or retry. It would
also establish multi-timestep CPU, uncaptured Warp, and captured CUDA evidence
for the same process configuration.

## User Stories

- As a GPU simulation operator, I would want a prepared resident timestep
  captured once and replayed safely so repeated fixed-shape timesteps could
  avoid Python launch orchestration without changing physics.
- As a library maintainer, I would want exact pre-launch compatibility checks
  and deterministic invalidation reasons so stale graphs could not launch
  against replaced arrays, schedules, maps, or devices.
- As a scientific reviewer, I would want planned captured results compared with
  CPU and uncaptured GPU references so graph execution could be supported by
  numerical, conservation, RNG-lifecycle, and failure-state evidence.
