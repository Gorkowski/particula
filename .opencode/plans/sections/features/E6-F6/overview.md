# Overview

## Problem Statement

Fixed-shape CPU and Warp particle arrays cannot admit every particle-creation
request once a box runs out of free slots. E6-F5 deliberately reports capacity
and fails atomically; without a separate exhaustion policy, downstream
nucleation would either stop, resize arrays, or silently discard demand.

## Value Proposition

E6-F6 provides a shared, conservation-checked policy layer. Deterministic
resampling is enabled by default and creates capacity without allocation;
representative-volume scaling is an independently selectable fallback that is
off by default. A complete read-only plan precedes every commit, so unsupported
or unsatisfiable requests fail without particle, volume, sidecar, or RNG writes.

## User Stories

- As a particle-resolved simulation user, I want full boxes handled by a
  documented default so new-particle demand is never silently lost.
- As a scientific developer, I want per-box number, species-mass, and charge
  accounting so resolution changes cannot create or destroy inventory.
- As a GPU developer, I want fixed-shape, caller-owned work buffers and CPU/Warp
  parity so exhaustion handling remains device-resident and allocation-stable.

This is feature track T6 under E6 and depends on E6-F5.

## Delivered P1 Baseline

Issue #1422 delivered the CPU-only, concrete
`particula.particles.exhaustion` planning boundary and its focused unit suite.
It validates fixed-shape `int32` capacity sidecars for every box before
resolving, returns frozen tuple-backed activation or deferred-policy plans, and
uses resampling-first policy selection. It also supplies float64 tuple-backed
weighted number, species-mass, and charge inventories. This baseline makes no
commit, release selection, resampling, scaling-feasibility, discovery,
re-export, or GPU change; those remain later phases.

## Delivered P2 Reference

Issue #1423 delivered the CPU-only deterministic fixed-capacity equal-weight
resampling reference in `particula/particles/exhaustion.py`. It builds frozen,
detached plans from cached validated state, stable-sorts active sources by
radius, composition, charge, and slot index, then uses a linear interval sweep
to form equal-weight retained records and release the requested trailing active
slots. Diagnostics validate represented number, species mass, signed charge,
radius-cubed, mean-radius, surface, and diversity/mixing bounds before an
all-box-preflighted atomic apply clears released slots. Focused co-located tests
cover deterministic, conservation, validation, and later-box atomicity paths.
P2 does not add a package re-export, scaling, GPU parity, discovery, or resize
behavior.

## Delivered P3 Warp Resampling

Issue #1424 delivered the direct fixed-capacity Warp boundary
`resampling_step_gpu` in `particula.gpu.kernels.exhaustion`. It accepts
explicit per-box release counts and concrete-only caller-owned
`ResamplingBuffers`, performs read-only schema/physical-value/count/buffer
validation, then plans entirely on the active device with staged bitonic sort
and interval-sweep remapping. Planning records diagnostic status and gates one
all-box commit, so failed planning leaves particles unchanged while successful
calls remap retained original slots and clear released slots without resizing
or hidden transfer.

Only `resampling_step_gpu` is exported through `particula.gpu.kernels`;
`ResamplingBuffers` remains concrete-module-only. Focused tests cover the
direct export, validation and ownership boundary, deterministic planning and
commit behavior, diagnostics, and Warp CPU parity with optional CUDA coverage.
Scaling, policy/P1 resolution, discovery/activation, and a high-level runnable
remain deferred.
