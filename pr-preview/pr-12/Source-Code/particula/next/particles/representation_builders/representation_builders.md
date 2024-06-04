# Representation Builders

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Particles](./index.md#particles) / Representation Builders

> Auto-generated documentation for [particula.next.particles.representation_builders](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_builders.py) module.

## LogNormalParticleRepresentationBuilder

[Show source in representation_builders.py:160](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_builders.py#L160)

Builder class for ParticleRepresentation objects with log-normal
distribution.

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
class LogNormalParticleRepresentationBuilder(
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

### LogNormalParticleRepresentationBuilder().build

[Show source in representation_builders.py:217](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_builders.py#L217)

Validate and return the ParticleRepresentation object.

#### Returns

The validated ParticleRepresentation object.

#### Signature

```python
def build(self) -> ParticleRepresentation: ...
```

#### See also

- [ParticleRepresentation](./representation.md#particlerepresentation)

### LogNormalParticleRepresentationBuilder().set_mean_radius

[Show source in representation_builders.py:204](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_builders.py#L204)

_summary_

#### Arguments

- `mean_radius` - Modes of the distribution.
- `mean_radius_units` - _description_. Defaults to "m".

#### Signature

```python
def set_mean_radius(
    self,
    mean_radius: Union[float, NDArrya[np.float_]],
    mean_radius_units: Optional[str] = "m",
): ...
```



## MassParticleRepresentationBuilder

[Show source in representation_builders.py:36](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_builders.py#L36)

Builder class for ParticleRepresentation objects with mass-based bins.

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

[Show source in representation_builders.py:80](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_builders.py#L80)

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

[Show source in representation_builders.py:98](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_builders.py#L98)

Builder class for ParticleRepresentation objects with radius-based bins.

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

[Show source in representation_builders.py:142](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_builders.py#L142)

Validate and return the ParticleRepresentation object.

#### Returns

The validated ParticleRepresentation object.

#### Signature

```python
def build(self) -> ParticleRepresentation: ...
```

#### See also

- [ParticleRepresentation](./representation.md#particlerepresentation)
