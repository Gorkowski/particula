# Representation Builders

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Particles](./index.md#particles) / Representation Builders

> Auto-generated documentation for [particula.next.particles.representation_builders](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_builders.py) module.

## LimitedRadiusParticleBuilder

[Show source in representation_builders.py:173](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_builders.py#L173)

General ParticleRepresentation objects with radius-based bins.

#### Methods

- `set_mode(mode,mode_units)` - Set the mode(s) of the distribution.
    Default is np.array([100e-9, 1e-6]) meters.
set_geometric_standard_deviation(
    geometric_standard_deviation,geometric_standard_deviation_units):
        Set the geometric standard deviation(s) of the distribution.
        Default is np.array([1.2, 1.4]).
set_number_concentration(
    - `number_concentration,number_concentration_units)` - Set the
        number concentration of the distribution. Default is
        np.array([1e4*1e6, 1e3*1e6]) particles/m**3.
- `set_radius_bins(radius_bins,radius_bins_units)` - Set the radius bins
    of the distribution. Default is np.logspace(-9, -4, 250), meters.

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

[Show source in representation_builders.py:322](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_builders.py#L322)

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

### LimitedRadiusParticleBuilder().set_distribution_type

[Show source in representation_builders.py:303](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_builders.py#L303)

Set the distribution type for the particle representation.

#### Arguments

- `distribution_type` - The type of distribution to use.

#### Signature

```python
def set_distribution_type(
    self, distribution_type: str, distribution_type_units: Optional[str] = None
): ...
```

### LimitedRadiusParticleBuilder().set_geometric_standard_deviation

[Show source in representation_builders.py:247](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_builders.py#L247)

Set the geometric standard deviation for the distribution

#### Arguments

- `geometric_standard_deviation` - The geometric standard deviation for
the radius.

#### Signature

```python
def set_geometric_standard_deviation(
    self,
    geometric_standard_deviation: NDArray[np.float64],
    geometric_standard_deviation_units: Optional[str] = None,
): ...
```

### LimitedRadiusParticleBuilder().set_mode

[Show source in representation_builders.py:229](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_builders.py#L229)

Set the modes for distribution

#### Arguments

- `modes` - The modes for the radius.
- `modes_units` - The units for the modes.

#### Signature

```python
def set_mode(self, mode: NDArray[np.float64], mode_units: str = "m"): ...
```

### LimitedRadiusParticleBuilder().set_number_concentration

[Show source in representation_builders.py:267](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_builders.py#L267)

Set the number concentration for the distribution

#### Arguments

- `number_concentration` - The number concentration for the radius.

#### Signature

```python
def set_number_concentration(
    self,
    number_concentration: NDArray[np.float64],
    number_concentration_units: str = "1/m^3",
): ...
```

### LimitedRadiusParticleBuilder().set_radius_bins

[Show source in representation_builders.py:286](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_builders.py#L286)

Set the radius bins for the distribution

#### Arguments

- `radius_bins` - The radius bins for the distribution.

#### Signature

```python
def set_radius_bins(
    self, radius_bins: NDArray[np.float64], radius_bins_units: str = "m"
): ...
```



## MassParticleRepresentationBuilder

[Show source in representation_builders.py:49](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_builders.py#L49)

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

[Show source in representation_builders.py:93](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_builders.py#L93)

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

[Show source in representation_builders.py:111](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_builders.py#L111)

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
    concentration of the particles. Default units are '1/m^3'.
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

[Show source in representation_builders.py:155](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_builders.py#L155)

Validate and return the ParticleRepresentation object.

#### Returns

The validated ParticleRepresentation object.

#### Signature

```python
def build(self) -> ParticleRepresentation: ...
```

#### See also

- [ParticleRepresentation](./representation.md#particlerepresentation)