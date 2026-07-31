# ADR-016: Resident Communication-Map Validation Boundary

**Status:** Accepted
**Date:** 2026-07-31
**Decision Makers:** ADW Development Team
**Technical Story:** [#1507](https://github.com/Gorkowski/particula/issues/1507)

## Context

Future resident communication requires fixed-capacity map declarations that can
be validated before a transport writer is introduced. The resident architecture
must retain caller-owned Warp arrays and preserve the frozen 26-name
`particula.execution` and top-level public surfaces.

### Problem Statement

Provide a deterministic, read-only P1 boundary for one-dimensional and
arbitrary directed-pair resident communication maps without prematurely adding
transport, mutable state, resource ownership, or public exports.

### Forces

**Driving Forces:**
- Future writers need validated fixed-shape topology and outbound bounds.
- Resident callers need identity preservation and retry-safe rejection.
- Communication schemas must coexist with the frozen execution API.

**Restraining Forces:**
- P1 must not transfer, write, transport, synchronize unrelated work, or fall
  back.
- Transport-resource allocation, registration, resizing, and volume evolution
  belong to later phases.

## Decision

Use `particula.execution.communication` as a concrete direct-import-only E7-F7
P1 declaration and read-only validation boundary. Keep it absent from
`particula.execution` and top-level exports.

### Chosen Option

**Option 2: Fixed-capacity declarations with complete read-only preflight**

The module will:

1. Retain immutable declaration records and all caller-owned Warp arrays by
   identity.
2. Validate metadata, schemas, storage nonaliasing, domains, enabled topology,
   strict outbound totals, and representation in a documented deterministic
   order.
3. Permit only bounded private device validation status/total scratch; defer all
   writer, transport, and volume-evolution behavior to P2+.

## Alternatives Considered

### Option 1: Add validation to a future transport writer

**Description:** Defer declarations and all validation until communication can
mutate resident state.

**Pros:**
- Fewer modules in P1.

**Cons:**
- Couples stable schema validation to an unshipped writer.
- Prevents independently retryable, no-mutation preflight.

**Reason for Rejection:** Fixed-capacity map semantics and rejection guarantees
need an explicit owner before transport exists.

---

### Option 2: Fixed-capacity declarations with complete read-only preflight
(chosen)

**Description:** Define direct-import-only records and validate all required
payload semantics before future writer phases.

**Pros:**
- Preserves caller identity and rejection recovery.
- Establishes deterministic topology and outbound-bound authority.
- Keeps the frozen public surface unchanged.

**Cons:**
- Callers use a concrete module import.
- P1 validation does not itself move data.

**Reason for Selection:** It creates a stable, bounded contract without
committing to transport mechanics.

---

### Option 3: Export map declarations from `particula.execution`

**Description:** Promote communication records as public execution APIs.

**Pros:**
- Shorter import path.

**Cons:**
- Breaks the frozen 26-name execution surface.
- Promotes phase-specific transport declarations before transport exists.

**Reason for Rejection:** The module is a concrete resident seam, not a new
general execution-selection API.

## Rationale

The P1 boundary separates immutable communication-map authority from future
execution. Complete preflight prevents disabled placeholder lanes from bypassing
schema or domain checks, allows reverse directed edges while rejecting duplicate
directed edges, and makes per-source outbound limits explicit. Read-only device
validation preserves caller resources and supports correction and retry.

### Trade-offs Accepted

1. **Concrete import path:** Records and validation are not package exports.
2. **No transport in P1:** A valid declaration has no writer or transfer effect.

## Consequences

### Positive

- Valid maps have deterministic schemas, topology, and outbound limits.
- Rejected preflight leaves supplied records and arrays unchanged.
- Empty and all-disabled maps have explicit no-op declaration semantics.

### Negative

- P1 performs no useful data movement by itself.
- A future writer must define its own post-launch fault and rollback policy.

### Neutral

- Bounded private validation scratch is permitted; caller-owned arrays remain
  unmodified and are never materialized as host payloads.

## Implementation

### Required Changes

1. **Communication declarations and preflight**
   (`particula/execution/communication.py`)
   - Add immutable map, volume, and shape declarations.
   - Add deterministic Warp-side validation with strict outbound bounds.
2. **Contract coverage** (`particula/execution/tests/communication_test.py`)
   - Cover valid maps, invalid schemas/domains/topologies, identity retention,
     and no-mutation rejection.
3. **Architecture documentation**
   - Record direct-import and frozen-export boundaries.

### Testing Strategy

Use same-device Warp CPU fixtures for valid one-dimensional and pair maps,
empty/all-disabled declarations, schema/domain/topology/outbound failures, and
unchanged caller state after rejection. Verify package and top-level export
surfaces remain frozen.

### Rollback Plan

Remove the concrete module and its tests; no package or top-level export needs
to be removed, and no transport state exists to migrate.

## Validation

### Success Criteria

- [x] Communication declarations retain caller-owned Warp arrays by identity.
- [x] Validation is deterministic and read-only for all required P1 checks.
- [x] The frozen 26-name execution and top-level public surfaces are unchanged.
- [x] Writer/transport and volume evolution remain deferred.

## References

- [ADR-004: Concrete GPU-Resident Session Boundary](ADR-004-concrete-gpu-resident-session-boundary.md)
- [ADR-015: Execution Public Surface and Experimental GPU Policy](ADR-015-execution-public-surface-and-experimental-gpu-policy.md)
- [Architecture Guide](../architecture_guide.md)
- [Architecture Outline](../architecture_outline.md)
- [Issue #1507](https://github.com/Gorkowski/particula/issues/1507)

## Notes

No prior ADR is superseded.
