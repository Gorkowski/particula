# Representation

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Particles](./index.md#particles) / Representation

> Auto-generated documentation for [particula.next.particles.representation](../../../../../particula/next/particles/representation.py) module.

## ParticleRepresentation

[Show source in representation.py:14](../../../../../particula/next/particles/representation.py#L14)

Represents a particle or a collection of particles, encapsulating the
strategy for calculating mass, radius, and total mass based on a
specified particle distribution, density, and concentration. This class
allows for flexibility in representing particles.

#### Attributes

- strategy (ParticleStrategy): The computation strategy for particle
representations.
- activity (ParticleActivityStrategy): The activity strategy for the
partial pressure calculations.
- surface (SurfaceStrategy): The surface strategy for surface tension and
Kelvin effect.
- distribution (NDArray[np.float_]): The distribution data for the
particles, which could represent sizes, masses, or another relevant metric.
- density (np.float_): The density of the material from which the
particles are made.
- concentration (NDArray[np.float_]): The concentration of particles
within the distribution.

#### Signature

```python
class ParticleRepresentation:
    def __init__(
        self,
        strategy: DistributionStrategy,
        activity: ActivityStrategy,
        surface: SurfaceStrategy,
        distribution: NDArray[np.float_],
        density: NDArray[np.float_],
        concentration: NDArray[np.float_],
        charge: NDArray[np.float_],
    ): ...
```

#### See also

- [ActivityStrategy](./activity_strategies.md#activitystrategy)
- [DistributionStrategy](./distribution_strategies.md#distributionstrategy)
- [SurfaceStrategy](./surface_strategies.md#surfacestrategy)

### ParticleRepresentation().add_concentration

[Show source in representation.py:117](../../../../../particula/next/particles/representation.py#L117)

Adds concentration to the particle distribution, updating the
concentration array.

#### Arguments

- added_concentration (NDArray[np.float_]): The concentration to be
    added per distribution bin.

#### Signature

```python
def add_concentration(self, added_concentration: NDArray[np.float_]) -> None: ...
```

### ParticleRepresentation().add_mass

[Show source in representation.py:105](../../../../../particula/next/particles/representation.py#L105)

Adds mass to the particle distribution, updating the concentration
and distribution arrays.

#### Arguments

- added_mass (NDArray[np.float_]): The mass to be added per
    distribution bin.

#### Signature

```python
def add_mass(self, added_mass: NDArray[np.float_]) -> None: ...
```

### ParticleRepresentation().get_charge

[Show source in representation.py:85](../../../../../particula/next/particles/representation.py#L85)

Returns the charge per particle.

#### Returns

- `-` *NDArray[np.float_]* - The charge of the particles.

#### Signature

```python
def get_charge(self) -> NDArray[np.float_]: ...
```

### ParticleRepresentation().get_mass

[Show source in representation.py:67](../../../../../particula/next/particles/representation.py#L67)

Returns the mass of the particles as calculated by the strategy.

#### Returns

- `-` *NDArray[np.float_]* - The mass of the particles.

#### Signature

```python
def get_mass(self) -> NDArray[np.float_]: ...
```

### ParticleRepresentation().get_radius

[Show source in representation.py:76](../../../../../particula/next/particles/representation.py#L76)

Returns the radius of the particles as calculated by the strategy.

#### Returns

- `-` *NDArray[np.float_]* - The radius of the particles.

#### Signature

```python
def get_radius(self) -> NDArray[np.float_]: ...
```

### ParticleRepresentation().get_total_mass

[Show source in representation.py:94](../../../../../particula/next/particles/representation.py#L94)

Returns the total mass of the particles as calculated by the strategy,
taking into account the distribution and concentration.

#### Returns

- `np.float_` - The total mass of the particles.

#### Signature

```python
def get_total_mass(self) -> np.float_: ...
```
