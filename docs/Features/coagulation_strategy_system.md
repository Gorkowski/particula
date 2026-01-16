# Coagulation Strategy System

> Strategy-based coagulation that unifies Brownian, charged, turbulent, and
> sedimentation kernels under a single particula dynamics workflow.

## Overview

The coagulation strategy system models particle collisions using the same
object-oriented patterns as condensation and wall loss. Instead of calling
standalone kernels, you configure strategy objects that operate on
`ParticleRepresentation`, support multiple distribution types, and expose a
consistent `rate` / `step` interface. Strategies plug into the
`Coagulation` runnable so you can compose coagulation with other dynamics in
one pipeline.

This feature is built around user-facing APIs exposed via `particula.dynamics`:

- `CoagulationStrategyABC` – abstract base for coagulation models.
- `BrownianCoagulationStrategy`, `ChargedCoagulationStrategy`,
  `TurbulentShearCoagulationStrategy`, `TurbulentDNSCoagulationStrategy`,
  `SedimentationCoagulationStrategy`, and `CombineCoagulationStrategy` –
  concrete kernels for diffusion, electrostatic effects, turbulence, settling,
  and kernel summation.
- `BrownianCoagulationBuilder`, `ChargedCoagulationBuilder`,
  `TurbulentShearCoagulationBuilder`, `TurbulentDNSCoagulationBuilder`,
  `SedimentationCoagulationBuilder`, and `CombineCoagulationStrategyBuilder`
  – validated builders for strategy configuration.
- `CoagulationFactory` – factory for selecting a strategy by name with builder
  defaults.
- `Coagulation` runnable – delegates to a coagulation strategy on `Aerosol`,
  splits `time_step` across `sub_steps`, and intentionally does **not** clamp
  concentrations.

## Key Benefits

- **Consistent dynamics workflow**: Use the same strategy-based API (`rate`,
  `step`, `distribution_type`) as condensation and wall loss.
- **Builder/factory parity with validation**: Configure coagulation using
  builders and the factory with required parameters, unit conversion, and
  distribution-type checks.
- **Unified API surface**: Access Brownian, charged, turbulent, sedimentation,
  and combined kernels directly from `particula.dynamics`.
- **Runnable integration**: Drop strategies into `Coagulation` and compose
  with other runnables (e.g., `Coagulation | WallLoss`) without glue code.
- **Particle-resolved support**: Run stochastic collision stepping on
  particle-resolved ensembles with configurable kernel binning.

## Who It's For

This feature is designed for:

- **Chamber and ambient modelers**: Simulating time-dependent coagulation with
  Brownian, charged, or turbulent enhancements.
- **Process engineers**: Assessing collision-driven growth in industrial
  aerosol systems under shear or DNS-calibrated turbulence.
- **Research and teaching users**: Comparing coagulation mechanisms with a
  unified API and reproducible examples.
- **Method developers**: Extending particula with new kernels while reusing
  the validated strategy, builder, and runnable patterns.

## Capabilities

### Unified coagulation API in `particula.dynamics`

Coagulation is exposed alongside other dynamics components:

```python
import particula as par

# Abstract interface (not instantiated directly)
par.dynamics.CoagulationStrategyABC

# Concrete strategies
par.dynamics.BrownianCoagulationStrategy
par.dynamics.ChargedCoagulationStrategy
par.dynamics.TurbulentShearCoagulationStrategy
par.dynamics.TurbulentDNSCoagulationStrategy
par.dynamics.SedimentationCoagulationStrategy
par.dynamics.CombineCoagulationStrategy

# Builders and factory
par.dynamics.BrownianCoagulationBuilder
par.dynamics.ChargedCoagulationBuilder
par.dynamics.TurbulentShearCoagulationBuilder
par.dynamics.TurbulentDNSCoagulationBuilder
par.dynamics.SedimentationCoagulationBuilder
par.dynamics.CombineCoagulationStrategyBuilder
par.dynamics.CoagulationFactory
```

All coagulation strategies share a common shape:

- Initialize with distribution type and strategy-specific parameters.
- Call `rate(...)` or `net_rate(...)` to inspect collision tendencies.
- Call `step(...)` to update the particle distribution.

### Runnable entry point: `Coagulation`

`Coagulation` is a `RunnableABC` implementation exported as
`par.dynamics.Coagulation`. It operates on an `Aerosol`, delegates `rate` and
`step` to the provided coagulation strategy, splits `time_step` across any
`sub_steps`, and **does not clamp** concentrations after each sub-step. Use
smaller `time_step` or higher `sub_steps` to improve stability, especially
with large kernels or charged/turbulent strategies.

```python
import particula as par

aerosol = ...  # your Aerosol with particles + atmosphere
coagulation = par.dynamics.Coagulation(
    coagulation_strategy=par.dynamics.BrownianCoagulationStrategy(
        distribution_type="discrete",
    ),
)

aerosol = coagulation.execute(
    aerosol,
    time_step=30.0,
    sub_steps=3,
)

# Chain with wall loss in one pipeline
wall_loss = par.dynamics.WallLoss(
    wall_loss_strategy=par.dynamics.SphericalWallLossStrategy(
        wall_eddy_diffusivity=1e-3,
        chamber_radius=0.5,
        distribution_type="discrete",
    )
)
combined = coagulation | wall_loss

aerosol = combined.execute(aerosol, time_step=30.0)
```

### Brownian coagulation strategy

`BrownianCoagulationStrategy` models diffusion-driven collisions. It works for
`"discrete"`, `"continuous_pdf"`, and `"particle_resolved"` distributions.
Its `dimensionless_kernel` is not implemented; use `kernel(...)` directly.

```python
import particula as par

brownian = par.dynamics.BrownianCoagulationStrategy(
    distribution_type="discrete",
)
T = 298.15  # K
P = 101325.0  # Pa
kernel = brownian.kernel(particle=particle, temperature=T, pressure=P)
particle = brownian.step(
    particle=particle,
    temperature=T,
    pressure=P,
    time_step=10.0,
)
```

### Charged coagulation strategy

`ChargedCoagulationStrategy` augments Brownian coagulation with Coulomb
interactions. It requires a charged kernel strategy, e.g.,
`HardSphereKernelStrategy` or `CoulombGopalakrishnan2012KernelStrategy`.

```python
import particula as par

kernel_strategy = par.dynamics.HardSphereKernelStrategy()
charged = par.dynamics.ChargedCoagulationStrategy(
    distribution_type="discrete",
    kernel_strategy=kernel_strategy,
)
particle = charged.step(
    particle=particle,
    temperature=298.15,
    pressure=101325.0,
    time_step=5.0,
)
```

### Turbulent shear coagulation strategy

`TurbulentShearCoagulationStrategy` implements the Saffman–Turner (1956)
turbulent shear kernel. Provide `turbulent_dissipation` and `fluid_density`.

```python
import particula as par

turbulent_dissipation = 0.01  # m^2/s^3
fluid_density = 1.225         # kg/m^3
shear = par.dynamics.TurbulentShearCoagulationStrategy(
    distribution_type="continuous_pdf",
    turbulent_dissipation=turbulent_dissipation,
    fluid_density=fluid_density,
)
particle = shear.step(
    particle=particle,
    temperature=298.15,
    pressure=101325.0,
    time_step=2.0,
)
```

### Turbulent DNS coagulation strategy

`TurbulentDNSCoagulationStrategy` follows Ayala et al. (2008) fits for larger
particles. Provide `turbulent_dissipation`, `fluid_density`,
`reynolds_lambda`, and `relative_velocity`.

```python
import particula as par

dns = par.dynamics.TurbulentDNSCoagulationStrategy(
    distribution_type="discrete",
    turbulent_dissipation=0.01,
    fluid_density=1.225,
    reynolds_lambda=23.0,
    relative_velocity=0.5,
)
particle = dns.step(
    particle=particle,
    temperature=298.15,
    pressure=101325.0,
    time_step=1.0,
)
```

### Sedimentation coagulation strategy

`SedimentationCoagulationStrategy` models gravitational-settling collisions via
Seinfeld & Pandis (2016). It is available via its builder or direct
construction but is **not** registered in `CoagulationFactory` mappings.

```python
import particula as par

sedimentation = par.dynamics.SedimentationCoagulationStrategy(
    distribution_type="discrete",
)
particle = sedimentation.step(
    particle=particle,
    temperature=298.15,
    pressure=101325.0,
    time_step=20.0,
)
```

### Combine coagulation strategy

`CombineCoagulationStrategy` sums kernels from multiple strategies. All
child strategies must share the same `distribution_type`. Its
`dimensionless_kernel` is not implemented.

```python
import particula as par

brownian = par.dynamics.BrownianCoagulationStrategy(
    distribution_type="discrete",
)
shear = par.dynamics.TurbulentShearCoagulationStrategy(
    distribution_type="discrete",
    turbulent_dissipation=0.01,
    fluid_density=1.225,
)
combined = par.dynamics.CombineCoagulationStrategy(
    strategies=[brownian, shear],
)
particle = combined.step(
    particle=particle,
    temperature=298.15,
    pressure=101325.0,
    time_step=5.0,
)
```

### Builder and factory workflow

Builders and the factory provide validated construction with required
parameters and distribution-type checks. Sedimentation is available via its
builder or direct strategy but is **not** mapped in the factory.

```python
import particula as par

# Brownian via builder
brownian = (
    par.dynamics.BrownianCoagulationBuilder()
    .set_distribution_type("discrete")
    .build()
)

# Charged via builder
charged = (
    par.dynamics.ChargedCoagulationBuilder()
    .set_distribution_type("discrete")
    .set_charged_kernel_strategy(par.dynamics.HardSphereKernelStrategy())
    .build()
)

# Factory selection (brownian, charged, turbulent_shear, turbulent_dns, combine)
factory = par.dynamics.CoagulationFactory()
shear = factory.get_strategy(
    strategy_type="turbulent_shear",
    parameters={
        "distribution_type": "continuous_pdf",
        "turbulent_dissipation": 0.02,
        "fluid_density": 1.1,
    },
)
```

### Support for multiple distribution types

The strategy system operates on the same distribution types used elsewhere:

- `"discrete"` – radius-binned distributions.
- `"continuous_pdf"` – continuous probability-density representations.
- `"particle_resolved"` – ensembles of individual particles with stochastic
  collision stepping.

Set `distribution_type` at initialization or via builders/factory parameters.
For particle-resolved runs, configure kernel binning inputs to avoid zero
radii and ensure stable `collide_pairs` selection.

### Particle-resolved coagulation notes

`CoagulationStrategyABC.step` handles particle-resolved mode by deriving a
kernel radius grid, converting to a speciated mass representation, and calling
`collide_pairs`. Provide optional kernel radii or bin counts to control the
binning used for kernel evaluation.

```python
import numpy as np
import particula as par

# Configure kernel radii or bins for particle-resolved runs
brownian_pr = par.dynamics.BrownianCoagulationStrategy(
    distribution_type="particle_resolved",
    particle_resolved_kernel_radius=np.geomspace(1e-8, 1e-5, 25),
)

particle = brownian_pr.step(
    particle=particle,
    temperature=298.15,
    pressure=101325.0,
    time_step=0.5,
)
```

## Getting Started

### Quick start: Brownian coagulation

```python
import particula as par

# 1. Build a radius-binned particle distribution
particle = par.particles.PresetParticleRadiusBuilder().build()

# 2. Configure Brownian coagulation
brownian = par.dynamics.BrownianCoagulationStrategy(
    distribution_type="discrete",
)

# 3. Run inside the Coagulation runnable
coagulation = par.dynamics.Coagulation(coagulation_strategy=brownian)
aerosol = par.aerosol.Aerosol(particles=particle)

aerosol = coagulation.execute(
    aerosol,
    time_step=30.0,
    sub_steps=2,
)
```

### Prerequisites

- `particula` version 0.2.6 or later installed.
- A `ParticleRepresentation` instance (e.g., from preset particle builders).
- Basic familiarity with particula dynamics and examples.

## Typical Workflows

### 1. Choose a distribution and strategy

Select a distribution type and configure a strategy with validated parameters.

```python
import particula as par

particle = par.particles.PresetParticleRadiusBuilder().build()
charged_kernel = par.dynamics.HardSphereKernelStrategy()
charged = par.dynamics.ChargedCoagulationStrategy(
    distribution_type="discrete",
    kernel_strategy=charged_kernel,
)
```

### 2. Build via factory or builder

Use the factory for quick selection or builders for explicit control.

```python
factory = par.dynamics.CoagulationFactory()
turbulent_dns = factory.get_strategy(
    strategy_type="turbulent_dns",
    parameters={
        "distribution_type": "discrete",
        "turbulent_dissipation": 0.01,
        "fluid_density": 1.225,
        "reynolds_lambda": 23.0,
        "relative_velocity": 0.5,
    },
)
```

### 3. Run with `Coagulation` and compose with other processes

```python
import particula as par

coagulation = par.dynamics.Coagulation(coagulation_strategy=charged)
wall_loss = par.dynamics.WallLoss(
    wall_loss_strategy=par.dynamics.SphericalWallLossStrategy(
        wall_eddy_diffusivity=1e-3,
        chamber_radius=0.5,
        distribution_type="discrete",
    )
)
combined = coagulation | wall_loss

aerosol = combined.execute(aerosol, time_step=20.0, sub_steps=4)
```

## Use Cases

### Use case 1: Chamber Brownian coagulation with stability control

**Scenario:** You need Brownian coagulation on a discrete distribution and want
stable time stepping without negative concentrations.

**Solution:** Configure `BrownianCoagulationStrategy` for `"discrete"` and run
it inside `Coagulation` with moderate `sub_steps` for stability. Inspect
`net_rate` before choosing `time_step`.

### Use case 2: Charged coagulation sensitivity study

**Scenario:** You want to study how Coulomb attraction/repulsion changes the
collision rate.

**Solution:** Use `ChargedCoagulationStrategy` with a chosen charged kernel
strategy, sweep electric potentials in the particle data, and compare `net_rate`
outcomes.

### Use case 3: Turbulent enhancement with DNS fits

**Scenario:** You need turbulence-enhanced collisions for super-micron
particles using DNS-calibrated kernels.

**Solution:** Use `TurbulentShearCoagulationStrategy` or
`TurbulentDNSCoagulationStrategy` with validated dissipation, fluid density,
Reynolds lambda, and relative velocity; run via the factory to ensure required
fields are set.

## Configuration

| Option | Description | Default |
|--------|-------------|---------|
| `distribution_type` | `"discrete"`, `"continuous_pdf"`, or `"particle_resolved"`. | `"discrete"` |
| `turbulent_dissipation` | Turbulent kinetic energy dissipation [m^2/s^3] (shear/DNS). | Required (shear/DNS) |
| `fluid_density` | Fluid density [kg/m^3] (shear/DNS). | Required (shear/DNS) |
| `reynolds_lambda` | Reynolds lambda (DNS). | Required (DNS) |
| `relative_velocity` | Relative air velocity [m/s] (DNS). | Required (DNS) |
| `charged_kernel_strategy` | Charged kernel strategy instance for `ChargedCoagulationStrategy`. | Required (charged) |
| `strategies` | List of strategies for `CombineCoagulationStrategy`; all must share distribution type. | Required (combine) |
| `particle_resolved_kernel_radius` | Optional radius grid for particle-resolved kernel evaluation. | `None` |
| `particle_resolved_kernel_bins_number` | Number of kernel bins for particle-resolved mode. | `None` |
| `particle_resolved_kernel_bins_per_decade` | Bins per decade for particle-resolved kernel grid. | `10` |
| `sub_steps` | Time-step subdivisions in `Coagulation.execute`; improves stability; no clamping. | `1` |

Notes:

- The dimensionless kernel is not implemented in `BrownianCoagulationStrategy`
  or `CombineCoagulationStrategy`.
- `CombineCoagulationStrategy` requires all child strategies to share the same
  `distribution_type`.
- `SedimentationCoagulationStrategy` is available via its builder or direct
  constructor; it is **not** registered in `CoagulationFactory` mappings.

## Best Practices

1. **Match distribution types**: Align `distribution_type` with how your
   `ParticleRepresentation` was constructed.
2. **Use sub-steps for stability**: Increase `sub_steps` or reduce `time_step`
   for charged or turbulence-enhanced kernels since `Coagulation` does not
   clamp concentrations.
3. **Validate via builders/factory**: Prefer builders or the factory to ensure
   required fields (dissipation, fluid density, kernel strategy) are set.
4. **Keep kernel units consistent**: Use SI units for dissipation, density,
   velocity, and pressure to avoid scale errors in kernel magnitude.
5. **Combine thoughtfully**: When using `CombineCoagulationStrategy`, ensure
   kernels are physically compatible and share the same distribution type.

## Limitations

- No clamping in `Coagulation`; negative concentrations can appear if time
  steps are too large relative to kernel magnitude.
- Dimensionless kernel functions are not implemented for Brownian or combined
  strategies.
- Factory mappings cover Brownian, charged, turbulent shear, turbulent DNS,
  and combine; sedimentation is available via direct builder/strategy only.
- No high-level orchestrator is provided; you manage the time loop and
  runnable composition.

## Related Documentation

- **Dynamics overview**: [Coagulation examples](../Examples/Dynamics/index.md#coagulation)
- **Notebooks**: [Coagulation 4: Methods Compared](../Examples/Dynamics/Coagulation/Coagulation_4_Compared.ipynb)
- **Functional charged example**: [Coagulation with charge (objects)](../Examples/Dynamics/Coagulation/Charge/Coagulation_with_Charge_objects.ipynb)
- **Particle-resolved pattern**: [Coagulation 3: Particle Resolved](../Examples/Dynamics/Coagulation/Coagulation_3_Particle_Resolved_Pattern.ipynb)
- **Theory reference**: [Droplet Coagulation Kernel Ayala 2008](../Theory/Technical/Dynamics/Cloud_Droplet_Coagulation/Droplet_Coagulation_Kernel_Ayala2008.md)

## FAQ

### How do I stabilize charged or turbulent coagulation runs?

Use smaller `time_step` or larger `sub_steps` in `Coagulation.execute`, and
inspect `net_rate` to size steps. Because `Coagulation` does not clamp, overly
large steps can produce negative concentrations.

### Can I mix distribution types inside `CombineCoagulationStrategy`?

No. All child strategies must share the same `distribution_type` to ensure
kernels align with the same particle representation.

### How do I include sedimentation via the factory?

Currently, sedimentation is not mapped in `CoagulationFactory`. Instantiate it
via `SedimentationCoagulationBuilder` or construct
`SedimentationCoagulationStrategy` directly.

## See Also

- [Wall loss strategy system](./wall_loss_strategy_system.md)
- [Simulation examples](../Examples/Simulations/index.md)
- [Particle phase examples](../Examples/Particle_Phase/index.md)
