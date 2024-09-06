# Representation

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Particles](./index.md#particles) / Representation

> Auto-generated documentation for [particula.next.particles.representation](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation.py) module.

## ParticleRepresentation

[Show source in representation.py:14](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation.py#L14)

Everything needed to represent a particle or a collection of particles.

Represents a particle or a collection of particles, encapsulating the
strategy for calculating mass, radius, and total mass based on a
specified particle distribution, density, and concentration. This class
allows for flexibility in representing particles.

#### Attributes

- `strategy` - The computation strategy for particle representations.
- `activity` - The activity strategy for the partial pressure calculations.
- `surface` - The surface strategy for surface tension and Kelvin effect.
- `distribution` - The distribution data for the particles, which could
    represent sizes, masses, or another relevant metric.
- `density` - The density of the material from which the particles are made.
- `concentration` - The concentration of particles within the distribution.
- `charge` - The charge on each particle.
- `volume` - The air volume for simulation of particles in the air,
    default is 1 m^3. This is only used in ParticleResolved Strategies.

#### Signature

```python
class ParticleRepresentation:
    def __init__(
        self,
        strategy: DistributionStrategy,
        activity: ActivityStrategy,
        surface: SurfaceStrategy,
        distribution: NDArray[np.float64],
        density: NDArray[np.float64],
        concentration: NDArray[np.float64],
        charge: NDArray[np.float64],
        volume: float = 1,
    ): ...
```

#### See also

- [ActivityStrategy](./activity_strategies.md#activitystrategy)
- [DistributionStrategy](./distribution_strategies.md#distributionstrategy)
- [SurfaceStrategy](./surface_strategies.md#surfacestrategy)

### ParticleRepresentation().add_concentration

[Show source in representation.py:103](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation.py#L103)

Adds concentration to the particle distribution.

#### Arguments

- `added_concentration` - The concentration to be
    added per distribution bin.

#### Signature

```python
def add_concentration(self, added_concentration: NDArray[np.float64]) -> None: ...
```

### ParticleRepresentation().add_mass

[Show source in representation.py:92](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation.py#L92)

Adds mass to the particle distribution, and updates parameters.

#### Arguments

- `added_mass` - The mass to be added per
    distribution bin.

#### Signature

```python
def add_mass(self, added_mass: NDArray[np.float64]) -> None: ...
```

### ParticleRepresentation().collide_pairs

[Show source in representation.py:114](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation.py#L114)

Collide pairs of indices, used for ParticleResolved Strategies.

#### Arguments

- `indices` - The indices to collide.

#### Signature

```python
def collide_pairs(self, indices: NDArray[np.int64]) -> None: ...
```

### ParticleRepresentation().get_charge

[Show source in representation.py:71](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation.py#L71)

Returns the charge per particle.

#### Returns

The charge of the particles.

#### Signature

```python
def get_charge(self) -> NDArray[np.float64]: ...
```

### ParticleRepresentation().get_mass

[Show source in representation.py:55](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation.py#L55)

Returns the mass of the particles as calculated by the strategy.

#### Returns

The mass of the particles.

#### Signature

```python
def get_mass(self) -> NDArray[np.float64]: ...
```

### ParticleRepresentation().get_radius

[Show source in representation.py:63](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation.py#L63)

Returns the radius of the particles as calculated by the strategy.

#### Returns

The radius of the particles.

#### Signature

```python
def get_radius(self) -> NDArray[np.float64]: ...
```

### ParticleRepresentation().get_total_mass

[Show source in representation.py:79](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation.py#L79)

Returns the total mass of the particles.

The total mass is as calculated by the strategy, taking into account
the distribution and concentration.

#### Returns

- `np.float64` - The total mass of the particles.

#### Signature

```python
def get_total_mass(self) -> np.float64: ...
```
