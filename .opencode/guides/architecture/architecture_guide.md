# Architecture Guide

## Particle Capacity-Planning Boundary

- `particula.particles.exhaustion` is a deliberately unexported concrete CPU
  P1 boundary. Callers that need it import the concrete module explicitly; do
  not add it to package re-exports.
- It accepts fixed-shape `int32` capacity sidecars, validates all boxes before
  resolving any plan, and returns immutable plan records. It applies
  resampling-first deferred-policy selection, but does not choose releases or
  scaling feasibility.
- P1 policy resolution is read-only: it owns no particle, RNG, diagnostic,
  work-buffer, or container state mutation. P2 creates a detached CPU remap
  plan and then performs a separately validated, all-box atomic commit; neither
  phase provides GPU work. A caller may safely retry P1 with corrected sidecars
  after validation or policy failure.
- Its weighted inventory helper provides float64 number, mass, and charge
  accounting only. Commit conservation and any moment-preservation guarantee
  remain responsibilities of later commit phases.

## GPU Module Boundaries

The GPU package keeps a strict separation between transfer, schema, and
kernel-entry responsibilities.

### Transfer boundary

- `particula/gpu/conversion.py` owns explicit CPU↔GPU transfer helpers only.
- It should not absorb launch-time kernel validation or normalization logic.

### Schema boundary

- `particula/gpu/warp_types.py` defines Warp-backed container schemas only.
- It should remain a passive data-shape layer rather than a behavior layer.

### Kernel normalization boundary

- `particula/gpu/kernels/environment.py` owns shared private normalization and
  validation for GPU kernel entry points.
- This module is the common boundary for accepting legacy scalars, direct
  `(n_boxes,)` Warp arrays, or `WarpEnvironmentData` inputs before launch-time
  work.
- Condensation and coagulation should reuse this boundary rather than
  re-implementing environment validation independently.

### GPU package export boundary

- `particula.gpu` remains the public home for Warp availability, context, and
  explicit CPU↔GPU transfer helpers.
- Direct GPU step entry points should be imported from
  `particula.gpu.kernels`, not re-exported from top-level `particula.gpu`.
- Lower-level kernel helpers should stay module-local to
  `particula.gpu.kernels.condensation` and
  `particula.gpu.kernels.coagulation` unless a broader public contract is
  intentionally documented.
- Import the supported low-level dilution entry point with
  `from particula.gpu.kernels import dilution_step_gpu`.
- `dilution_step_gpu` completes deterministic, read-only validation before
  allocating private storage, launching a kernel, or mutating caller-owned
  state. Successful calls update particle and gas concentrations in place as
  `c_new = c * exp(-alpha * time_step)` and return the identical containers.
- The preflight guarantee ends at launch: post-launch rollback is not
  provided. This direct entry point does not imply CPU fallback or runnable
  support.
- Import the supported fixed-slot wall-loss boundary with
  `from particula.gpu.kernels import wall_loss_step_gpu`. Its
  `NeutralWallLossConfig` is deliberately concrete-module-only at
  `particula.gpu.kernels.wall_loss`; do not re-export it through
  `particula.gpu.kernels` or `particula.gpu`.
- `wall_loss_step_gpu` owns immutable host configuration and frozen preflight
  for particle-resolved neutral and charged inputs. It dispatches the
  unchanged neutral kernel for neutral mode; charged kernels compose the private
  image-charge and electric-field-drift helpers in
  `particula.gpu.dynamics.wall_loss_funcs` only for nonzero-charge slots.
  Image-charge enhancement remains active for nonzero charge at zero wall
  potential, while charged zero-charge slots retain the exact neutral
  coefficient and RNG path. Spherical charged execution preserves a signed
  scalar field before adding the signed potential-derived contribution.
  Rectangular execution reduces caller-owned `(3,)` `float64` Warp field
  storage to its Euclidean magnitude before adding the signed
  potential-derived contribution; component signs do not individually select
  drift direction. The rectangular field is passed only to the charged
  rectangular kernel.
- After successful preflight, a nonzero call stochastically clears eligible
  fixed slots in place and returns the identical particle object. Removed slots
  have every mass lane, concentration, and caller-owned `charge` cleared;
  capacity and unselected storage are preserved. Zero time is write-free;
  pre-launch failures are atomic; rollback after a mutation launch is not
  promised.
- Its caller-owned `WarpParticleData.charge` field and optional `(n_boxes,)`
  `uint32` RNG sidecar remain external state rather than hidden transfer
  results. Successful positive-time calls advance each box sequentially only
  for eligible slots, while omitted RNG state is private to the call. Explicit
  `initialize_rng=True` is the only supplied-state reset path. Zero-time,
  preflight failures, and positive-time inputs with no usable slots leave a
  supplied sidecar unchanged. This serial per-box ownership is a correctness
  constraint, not a throughput claim. Runnables, hidden transfers/fallbacks,
  and CPU/Warp stochastic parity remain deferred. See
  [ADR-001](decisions/ADR-001-neutral-gpu-wall-loss-boundary.md).

## Design Intent

- Keep CPU↔GPU transfers explicit.
- Keep Warp container definitions stable and behavior-free.
- Keep cross-entry-point normalization private to `particula/gpu/kernels/`.
- Share validation at kernel boundaries when multiple GPU entry points consume
  the same environment contract.
- Keep GPU exports deliberate: top-level helpers in `particula.gpu`, direct
  step entry points in `particula.gpu.kernels`.
