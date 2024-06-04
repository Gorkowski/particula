# Representation Builders

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Particles](./index.md#particles) / Representation Builders

> Auto-generated documentation for [particula.next.particles.representation_builders](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_builders.py) module.

## LimitedRadiusParticleBuilder

[Show source in representation_builders.py:172](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_builders.py#L172)

General ParticleRepresentation objects with radius-based bins.

#### Methods

- `set_distribution_strategy(strategy)` - Set the DistributionStrategy.
- `set_activity_strategy(strategy)` - Set the ActivityStrategy.
- `set_surface_strategy(strategy)` - Set the SurfaceStrategy.
- `set_modes(radius_limits)` - Set the limits for the mode.

#### Signature

```python
class LimitedRadiusParticleBuilder(
    BuilderABC,
    BuilderDistributionStrategyMixin,
    BuilderActivityStrategyMixin,
    BuilderSurfaceStrategyMixin,
    BuilderRadiusMixin,
    BuilderDensityMixin,
    BuilderConcentrationMixin,
    BuilderChargeMixin,
):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)
- [BuilderActivityStrategyMixin](../abc_builder.md#builderactivitystrategymixin)
- [BuilderChargeMixin](../abc_builder.md#builderchargemixin)
- [BuilderConcentrationMixin](../abc_builder.md#builderconcentrationmixin)
- [BuilderDensityMixin](../abc_builder.md#builderdensitymixin)
- [BuilderDistributionStrategyMixin](../abc_builder.md#builderdistributionstrategymixin)
- [BuilderRadiusMixin](../abc_builder.md#builderradiusmixin)
- [BuilderSurfaceStrategyMixin](../abc_builder.md#buildersurfacestrategymixin)

### LimitedRadiusParticleBuilder().build

[Show source in representation_builders.py:300](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_builders.py#L300)

Validate and return the ParticleRepresentation object.

This will build a distribution of particles with a lognormal size
distribution, before returning the ParticleRepresentation object.

#### Returns

The validated ParticleRepresentation object.

#### Signature

```python
def build(self) -> ParticleRepresentation: ...
```

#### See also

- [ParticleRepresentation](./representation.md#particlerepresentation)

### LimitedRadiusParticleBuilder().set_geometric_standard_deviation

[Show source in representation_builders.py:242](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_builders.py#L242)

Set the geometric standard deviation for the distribution

#### Arguments

- `geometric_standard_deviation` - The geometric standard deviation for
the radius.

#### Signature

```python
def set_geometric_standard_deviation(
    self,
    geometric_standard_deviation: NDArray[np.float_],
    geometric_standard_deviation_units: Optional[str] = None,
): ...
```

### LimitedRadiusParticleBuilder().set_mode

[Show source in representation_builders.py:223](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_builders.py#L223)

Set the modes for distribution

#### Arguments

- `modes` - The modes for the radius.
- `modes_units` - The units for the modes.

#### Signature

```python
def set_mode(self, mode: NDArray[np.float_], mode_units): ...
```

### LimitedRadiusParticleBuilder().set_number_concentration

[Show source in representation_builders.py:263](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_builders.py#L263)

Set the number concentration for the distribution

#### Arguments

- `number_concentration` - The number concentration for the radius.

#### Signature

```python
def set_number_concentration(
    self,
    number_concentration: NDArray[np.float_],
    number_concentration_units: Optional[str] = None,
): ...
```

### LimitedRadiusParticleBuilder().set_radius_bins

[Show source in representation_builders.py:282](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_builders.py#L282)

Set the radius bins for the distribution

#### Arguments

- `radius_bins` - The radius bins for the distribution.

#### Signature

```python
def set_radius_bins(
    self, radius_bins: NDArray[np.float_], radius_bins_units: Optional[str] = None
): ...
```



## MassParticleRepresentationBuilder

[Show source in representation_builders.py:48](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_builders.py#L48)

General ParticleRepresentation objects with mass-based bins.

#### Methods

- `set_distribution_strategy(strategy)` - Set the DistributionStrategy.
- `set_activity_strategy(strategy)` - Set the ActivityStrategy.
- `set_surface_strategy(strategy)` - Set the SurfaceStrategy.
- `set_mass(mass,` *mass_units)* - Set the mass of the particles. Default
    units are 'kg'.
- `set_density(density,` *density_units)* - Set the density of the particles.
    Default units are 'kg/m**3'.
- `set_concentration(concentration,` *concentration_units)* - Set the
    concentration of the particles. Default units are '/m**3'.
- `set_charge(charge,` *charge_units)* - Set the number of charges.

#### Signature

```python
class MassParticleRepresentationBuilder(
    BuilderABC,
    BuilderDistributionStrategyMixin,
    BuilderActivityStrategyMixin,
    BuilderSurfaceStrategyMixin,
    BuilderMassMixin,
    BuilderDensityMixin,
    BuilderConcentrationMixin,
    BuilderChargeMixin,
):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)
- [BuilderActivityStrategyMixin](../abc_builder.md#builderactivitystrategymixin)
- [BuilderChargeMixin](../abc_builder.md#builderchargemixin)
- [BuilderConcentrationMixin](../abc_builder.md#builderconcentrationmixin)
- [BuilderDensityMixin](../abc_builder.md#builderdensitymixin)
- [BuilderDistributionStrategyMixin](../abc_builder.md#builderdistributionstrategymixin)
- [BuilderMassMixin](../abc_builder.md#buildermassmixin)
- [BuilderSurfaceStrategyMixin](../abc_builder.md#buildersurfacestrategymixin)

### MassParticleRepresentationBuilder().build

[Show source in representation_builders.py:92](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_builders.py#L92)

Validate and return the ParticleRepresentation object.

#### Returns

The validated ParticleRepresentation object.

#### Signature

```python
def build(self) -> ParticleRepresentation: ...
```

#### See also

- [ParticleRepresentation](./representation.md#particlerepresentation)



## RadiusParticleRepresentationBuilder

[Show source in representation_builders.py:110](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_builders.py#L110)

General ParticleRepresentation objects with radius-based bins.

#### Methods

- `set_distribution_strategy(strategy)` - Set the DistributionStrategy.
- `set_activity_strategy(strategy)` - Set the ActivityStrategy.
- `set_surface_strategy(strategy)` - Set the SurfaceStrategy.
- `set_radius(radius,` *radius_units)* - Set the radius of the particles.
    Default units are 'm'.
- `set_density(density,` *density_units)* - Set the density of the particles.
    Default units are 'kg/m**3'.
- `set_concentration(concentration,` *concentration_units)* - Set the
    concentration of the particles. Default units are '/m**3'.
- `set_charge(charge,` *charge_units)* - Set the number of charges.

#### Signature

```python
class RadiusParticleRepresentationBuilder(
    BuilderABC,
    BuilderDistributionStrategyMixin,
    BuilderActivityStrategyMixin,
    BuilderSurfaceStrategyMixin,
    BuilderRadiusMixin,
    BuilderDensityMixin,
    BuilderConcentrationMixin,
    BuilderChargeMixin,
):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)
- [BuilderActivityStrategyMixin](../abc_builder.md#builderactivitystrategymixin)
- [BuilderChargeMixin](../abc_builder.md#builderchargemixin)
- [BuilderConcentrationMixin](../abc_builder.md#builderconcentrationmixin)
- [BuilderDensityMixin](../abc_builder.md#builderdensitymixin)
- [BuilderDistributionStrategyMixin](../abc_builder.md#builderdistributionstrategymixin)
- [BuilderRadiusMixin](../abc_builder.md#builderradiusmixin)
- [BuilderSurfaceStrategyMixin](../abc_builder.md#buildersurfacestrategymixin)

### RadiusParticleRepresentationBuilder().build

[Show source in representation_builders.py:154](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_builders.py#L154)

Validate and return the ParticleRepresentation object.

#### Returns

The validated ParticleRepresentation object.

#### Signature

```python
def build(self) -> ParticleRepresentation: ...
```

#### See also

- [ParticleRepresentation](./representation.md#particlerepresentation)
