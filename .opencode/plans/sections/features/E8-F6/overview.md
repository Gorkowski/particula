# Overview

## Problem Statement

Epic H needs reproducible evidence for its primary performance axis: many
independent boxes executed through the GPU-resident loop. Existing opt-in CUDA
benchmarks cover direct condensation and coagulation, but they do not compare
captured and uncaptured complete resident timesteps or account consistently for
the memory required by primary state, inactive fixed slots, registry-owned
temporaries, diagnostics, communication, checkpoints, and future autodiff tape.

## Value Proposition

E8-F6 extends the existing `--benchmark` surface with a box-first resident
matrix, raw captured-versus-uncaptured timing samples, explicit unavailable
rows, and a reproducible memory-budget report. The report consumes E8-F3's
canonical logical-byte inventory, separates analytical logical bytes from
allocator-observed peak memory, and records enough hardware and command context
for maintainers to reproduce or qualify every claim.

## Delivered P1 Foundation

Issue #1581 delivered the concrete, host-only evidence foundation at
`particula/execution/tests/resident_benchmark_support.py`, with default-
collection tests in `particula/execution/tests/resident_benchmark_support_test.py`.
It validates frozen benchmark cases, results, timing summaries, and complete
caller-provided provenance before any CUDA-facing work; deterministically
round-trips schema-versioned JSON; and atomically writes generic JSON only
beneath a verified `.artifacts` root. It does not execute benchmarks, import or
probe Warp/CUDA, modify resident execution, add exports, or publish evidence.

## User Stories

- As a performance engineer, I want captured and uncaptured resident loops run
  against identical fixtures so that launch-overhead savings are measurable.
- As a simulation operator, I want box/particle/species cases checked against a
  memory budget before allocation so that oversized rows skip explicitly.
- As an Epic I planner, I want current and projected tape-memory components
  reported separately so that differentiable workloads can be scoped honestly.
