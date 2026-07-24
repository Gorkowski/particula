# Architecture Outline

## Particle Package

`particula/particles/` contains particle data, representations, distribution
strategies, and concrete particle-domain helpers.

### particula/particles/

**Key Components:**
- `exhaustion.py` - Concrete, deliberately unexported CPU P1 boundary for
  read-only fixed-shape capacity exhaustion planning and float64 weighted
  inventory accounting. It validates every box before resolution, applies
  resampling-first deferred-policy selection, and returns immutable plans; it
  owns neither state mutation nor commit, GPU work, or container schema.
- `distribution_strategies/` - Particle distribution implementations
- `properties/` - Particle property calculations
- `tests/` - Test coverage

## GPU Package

`particula/gpu/` contains Warp-backed data containers, explicit CPU↔GPU
transfer helpers, device-side physics helpers, and kernel entry points.

### particula/gpu/

**Key Components:**
- `__init__.py` - Public GPU exports
- `conversion.py` - Explicit CPU↔GPU transfer helpers only
- `warp_types.py` - Warp container schemas only
- `dynamics/` - GPU physics helper functions
- `properties/` - GPU property helper functions
- `kernels/` - GPU kernel entry points and private kernel support helpers
- `tests/` - Test coverage

### particula/gpu/kernels/

GPU kernel entry points own launch-time orchestration and may depend on shared
private helpers for cross-kernel setup.

**Key Components:**
- `condensation.py` - Condensation GPU entry points and kernels
- `coagulation.py` - Coagulation GPU entry points and kernels
- `dilution.py` - Concrete P1 GPU dilution input boundary; validation scans may
  allocate or launch, but rejected calls have no update-kernel launch or caller
  mutation
- `exhaustion.py` - Direct Warp fixed-capacity equal-weight resampling boundary.
  It consumes explicit per-box release counts, uses caller-owned planning and
  diagnostic buffers, and atomically commits only after all boxes pass
  diagnostics. Only `resampling_step_gpu` is exported; `ResamplingBuffers`,
  status codes, and kernels remain concrete-module-only. It provides no
  runnable, policy resolution, CPU fallback or transfer, or resizing.
- `wall_loss.py` - Concrete fixed-slot neutral/charged GPU wall-loss boundary;
  owns immutable host configuration, frozen preflight, bounded fixed-slot
  removal, and the external caller-owned per-box RNG sidecar lifecycle. Charged
  mode composes private image-charge and field-drift helpers from
  `particula.gpu.dynamics.wall_loss_funcs` only for nonzero-charge slots;
  zero-charge slots retain the neutral path. The sidecar is not added to Warp
  particle schemas or package exports, and sequential per-box ownership
  advances it only for eligible slots.
- `environment.py` - Shared private normalization and validation for kernel
  environment inputs
- `tests/` - Test coverage
