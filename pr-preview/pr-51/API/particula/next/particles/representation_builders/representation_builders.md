# Representation Builders

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Particles](./index.md#particles) / Representation Builders

> Auto-generated documentation for [particula.next.particles.representation_builders](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_builders.py) module.

## ParticleMassRepresentationBuilder

[Show source in representation_builders.py:54](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_builders.py#L54)

General ParticleRepresentation objects with mass-based bins.

#### Attributes

- `distribution_strategy` - Set the DistributionStrategy.
- `activity_strategy` - Set the ActivityStrategy.
- `surface_strategy` - Set the SurfaceStrategy.
- `mass` - Set the mass of the particles. Default units are 'kg'.
- `density` - Set the density of the particles. Default units are 'kg/m^3'.
- `concentration` - Set the concentration of the particles.
    Default units are '1/m^3'.
- `charge` - Set the number of charges.

#### Signature

```python
class ParticleMassRepresentationBuilder(
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

### ParticleMassRepresentationBuilder().build

[Show source in representation_builders.py:96](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_builders.py#L96)

Validate and return the ParticleRepresentation object.

#### Returns

The validated ParticleRepresentation object.

#### Signature

```python
def build(self) -> ParticleRepresentation: ...
```

#### See also

- [ParticleRepresentation](./representation.md#particlerepresentation)



## ParticleRadiusRepresentationBuilder

[Show source in representation_builders.py:114](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_builders.py#L114)

General ParticleRepresentation objects with radius-based bins.

#### Attributes

- `distribution_strategy` - Set the DistributionStrategy.
- `activity_strategy` - Set the ActivityStrategy.
- `surface_strategy` - Set the SurfaceStrategy.
- `radius` - Set the radius of the particles. Default units are 'm'.
- `density` - Set the density of the particles. Default units are 'kg/m**3'.
- `concentration` - Set the concentration of the particles. Default units
    are '1/m^3'.
- `charge` - Set the number of charges.

#### Signature

```python
class ParticleRadiusRepresentationBuilder(
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

### ParticleRadiusRepresentationBuilder().build

[Show source in representation_builders.py:156](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_builders.py#L156)

Validate and return the ParticleRepresentation object.

#### Returns

The validated ParticleRepresentation object.

#### Signature

```python
def build(self) -> ParticleRepresentation: ...
```

#### See also

- [ParticleRepresentation](./representation.md#particlerepresentation)



## PresetParticleRadiusBuilder

[Show source in representation_builders.py:174](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_builders.py#L174)

General ParticleRepresentation objects with radius-based bins.

#### Attributes

- `mode` - Set the mode(s) of the distribution.
    Default is np.array([100e-9, 1e-6]) meters.
- `geometric_standard_deviation` - Set the geometric standard deviation(s)
    of the distribution. Default is np.array([1.2, 1.4]).
- `number_concentration` - Set the number concentration of the distribution.
    Default is np.array([1e4x1e6, 1e3x1e6]) particles/m^3.
- `radius_bins` - Set the radius bins of the distribution. Default is
    np.logspace(-9, -4, 250), meters.

#### Signature

```python
class PresetParticleRadiusBuilder(
    BuilderABC,
    BuilderDistributionStrategyMixin,
    BuilderActivityStrategyMixin,
    BuilderSurfaceStrategyMixin,
    BuilderRadiusMixin,
    BuilderDensityMixin,
    BuilderConcentrationMixin,
    BuilderChargeMixin,
    BuilderLognormalMixin,
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
- [BuilderLognormalMixin](../abc_builder.md#builderlognormalmixin)
- [BuilderRadiusMixin](../abc_builder.md#builderradiusmixin)
- [BuilderSurfaceStrategyMixin](../abc_builder.md#buildersurfacestrategymixin)

### PresetParticleRadiusBuilder().build

[Show source in representation_builders.py:264](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_builders.py#L264)

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

### PresetParticleRadiusBuilder().set_distribution_type

[Show source in representation_builders.py:245](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_builders.py#L245)

Set the distribution type for the particle representation.

#### Arguments

- `distribution_type` - The type of distribution to use.

#### Signature

```python
def set_distribution_type(
    self, distribution_type: str, distribution_type_units: Optional[str] = None
): ...
```

### PresetParticleRadiusBuilder().set_radius_bins

[Show source in representation_builders.py:228](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_builders.py#L228)

Set the radius bins for the distribution

#### Arguments

- `radius_bins` - The radius bins for the distribution.

#### Signature

```python
def set_radius_bins(
    self, radius_bins: NDArray[np.float64], radius_bins_units: str = "m"
): ...
```



## PresetResolvedParticleMassBuilder

[Show source in representation_builders.py:378](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_builders.py#L378)

General ParticleRepresentation objects with particle resolved masses.

This class has preset values for all the attributes, and allows you to
override them as needed. This is useful when you want to quickly
particle representation object with resolved masses.

#### Attributes

- `distribution_strategy` - Set the DistributionStrategy.
- `activity_strategy` - Set the ActivityStrategy.
- `surface_strategy` - Set the SurfaceStrategy.
- `mass` - Set the mass of the particles Default
    units are 'kg'.
- `density` - Set the density of the particles.
    Default units are 'kg/m^3'.
- `charge` - Set the number of charges.
- `mode` - Set the mode(s) of the distribution.
    Default is np.array([100e-9, 1e-6]) meters.
- `geometric_standard_deviation` - Set the geometric standard
    deviation(s) of the distribution. Default is np.array([1.2, 1.4]).
- `number_concentration` - Set the number concentration of the
    distribution. Default is np.array([1e4 1e6, 1e3 1e6])
    particles/m^3.
- `particle_resolved_count` - Set the number of resolved particles.

#### Signature

```python
class PresetResolvedParticleMassBuilder(
    BuilderABC,
    BuilderDistributionStrategyMixin,
    BuilderActivityStrategyMixin,
    BuilderSurfaceStrategyMixin,
    BuilderDensityMixin,
    BuilderChargeMixin,
    BuilderLognormalMixin,
    BuilderVolumeMixin,
    BuilderParticleResolvedCountMixin,
):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)
- [BuilderActivityStrategyMixin](../abc_builder.md#builderactivitystrategymixin)
- [BuilderChargeMixin](../abc_builder.md#builderchargemixin)
- [BuilderDensityMixin](../abc_builder.md#builderdensitymixin)
- [BuilderDistributionStrategyMixin](../abc_builder.md#builderdistributionstrategymixin)
- [BuilderLognormalMixin](../abc_builder.md#builderlognormalmixin)
- [BuilderParticleResolvedCountMixin](../abc_builder.md#builderparticleresolvedcountmixin)
- [BuilderSurfaceStrategyMixin](../abc_builder.md#buildersurfacestrategymixin)
- [BuilderVolumeMixin](../abc_builder.md#buildervolumemixin)

### PresetResolvedParticleMassBuilder().build

[Show source in representation_builders.py:446](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_builders.py#L446)

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



## ResolvedParticleMassRepresentationBuilder

[Show source in representation_builders.py:304](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_builders.py#L304)

Builder class for constructing ParticleRepresentation objects with
resolved masses.

This class allows you to set various attributes for a particle
representation, such as distribution strategy, mass, density, charge,
volume, and more. These attributes are validated and there a no presets.

#### Attributes

- `distribution_strategy` - Set the distribution strategy for particles.
- `activity_strategy` - Set the activity strategy for the particles.
- `surface_strategy` - Set the surface strategy for the particles.
- `mass` - Set the particle mass. Defaults to 'kg'.
- `density` - Set the particle density. Defaults to 'kg/m^3'.
- `charge` - Set the particle charge.
- `volume` - Set the particle volume. Defaults to 'm^3'.

#### Signature

```python
class ResolvedParticleMassRepresentationBuilder(
    BuilderABC,
    BuilderDistributionStrategyMixin,
    BuilderActivityStrategyMixin,
    BuilderSurfaceStrategyMixin,
    BuilderDensityMixin,
    BuilderChargeMixin,
    BuilderVolumeMixin,
    BuilderMassMixin,
):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)
- [BuilderActivityStrategyMixin](../abc_builder.md#builderactivitystrategymixin)
- [BuilderChargeMixin](../abc_builder.md#builderchargemixin)
- [BuilderDensityMixin](../abc_builder.md#builderdensitymixin)
- [BuilderDistributionStrategyMixin](../abc_builder.md#builderdistributionstrategymixin)
- [BuilderMassMixin](../abc_builder.md#buildermassmixin)
- [BuilderSurfaceStrategyMixin](../abc_builder.md#buildersurfacestrategymixin)
- [BuilderVolumeMixin](../abc_builder.md#buildervolumemixin)

### ResolvedParticleMassRepresentationBuilder().build

[Show source in representation_builders.py:351](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation_builders.py#L351)

Validate and return a ParticleRepresentation object.

This method validates all the required attributes and builds a particle
representation with a lognormal size distribution.

#### Returns

- `ParticleRepresentation` - A validated particle representation object.

#### Signature

```python
def build(self) -> ParticleRepresentation: ...
```

#### See also

- [ParticleRepresentation](./representation.md#particlerepresentation)
