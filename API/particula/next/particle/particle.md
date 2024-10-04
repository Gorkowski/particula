# Particle

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Next](./index.md#next) / Particle

> Auto-generated documentation for [particula.next.particle](https://github.com/Gorkowski/particula/blob/main/particula/next/particle.py) module.

## MassBasedStrategy

[Show source in particle.py:76](https://github.com/Gorkowski/particula/blob/main/particula/next/particle.py#L76)

A strategy for particles represented by their mass distribution, and
particle number concentration. This class provides the implementation
of the methods for ParticleStrategy.

#### Signature

```python
class MassBasedStrategy(ParticleStrategy): ...
```

#### See also

- [ParticleStrategy](#particlestrategy)

### MassBasedStrategy().get_mass

[Show source in particle.py:83](https://github.com/Gorkowski/particula/blob/main/particula/next/particle.py#L83)

#### Signature

```python
def get_mass(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```

### MassBasedStrategy().get_radius

[Show source in particle.py:91](https://github.com/Gorkowski/particula/blob/main/particula/next/particle.py#L91)

#### Signature

```python
def get_radius(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```

### MassBasedStrategy().get_total_mass

[Show source in particle.py:101](https://github.com/Gorkowski/particula/blob/main/particula/next/particle.py#L101)

#### Signature

```python
def get_total_mass(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    density: NDArray[np.float64],
) -> np.float64: ...
```



## Particle

[Show source in particle.py:251](https://github.com/Gorkowski/particula/blob/main/particula/next/particle.py#L251)

Represents a particle or a collection of particles, encapsulating the
strategy for calculating mass, radius, and total mass based on a
specified particle distribution, density, and concentration. This class
allows for flexibility in representing particles by delegating computation
to a strategy pattern.

#### Attributes

- strategy (ParticleStrategy): The computation strategy for particle
properties.
- distribution (NDArray[np.float64]): The distribution data for the
particles, which could represent sizes, masses, or another relevant metric.
- density (np.float64): The density of the material from which the
particles are made.
- concentration (NDArray[np.float64]): The concentration of particles
within the distribution.
- charge (Optional[NDArray[np.float64]]): The charge distribution of the
particles.
- shape_factor (Optional[NDArray[np.float64]]): The shape factor
distribution of the particles.

#### Signature

```python
class Particle:
    def __init__(
        self,
        strategy: ParticleStrategy,
        distribution: NDArray[np.float64],
        density: NDArray[np.float64],
        concentration: NDArray[np.float64],
    ): ...
```

#### See also

- [ParticleStrategy](#particlestrategy)

### Particle().get_mass

[Show source in particle.py:321](https://github.com/Gorkowski/particula/blob/main/particula/next/particle.py#L321)

Returns the mass of the particles as calculated by the strategy.

#### Returns

- `-` *NDArray[np.float64]* - The mass of the particles.

#### Signature

```python
def get_mass(self) -> NDArray[np.float64]: ...
```

### Particle().get_radius

[Show source in particle.py:330](https://github.com/Gorkowski/particula/blob/main/particula/next/particle.py#L330)

Returns the radius of the particles as calculated by the strategy.

#### Returns

- `-` *NDArray[np.float64]* - The radius of the particles.

#### Signature

```python
def get_radius(self) -> NDArray[np.float64]: ...
```

### Particle().get_total_mass

[Show source in particle.py:339](https://github.com/Gorkowski/particula/blob/main/particula/next/particle.py#L339)

Returns the total mass of the particles as calculated by the strategy,
taking into account the distribution and concentration.

#### Returns

- `np.float64` - The total mass of the particles.

#### Signature

```python
def get_total_mass(self) -> np.float64: ...
```

### Particle().set_charge

[Show source in particle.py:301](https://github.com/Gorkowski/particula/blob/main/particula/next/particle.py#L301)

Sets the charge distribution for the particles.

#### Arguments

- charge (NDArray[np.float64]): The charge distribution across the
particles.

#### Signature

```python
def set_charge(self, charge: NDArray[np.float64]): ...
```

### Particle().set_shape_factor

[Show source in particle.py:311](https://github.com/Gorkowski/particula/blob/main/particula/next/particle.py#L311)

Sets the shape factor distribution for the particles.

#### Arguments

- shape_factor (NDArray[np.float64]): The shape factor distribution
across the particles.

#### Signature

```python
def set_shape_factor(self, shape_factor: NDArray[np.float64]): ...
```



## ParticleStrategy

[Show source in particle.py:8](https://github.com/Gorkowski/particula/blob/main/particula/next/particle.py#L8)

Abstract base class for particle strategy, defining the common
interface for mass, radius, and total mass calculations for different
particle representations.

#### Signature

```python
class ParticleStrategy(ABC): ...
```

### ParticleStrategy().get_mass

[Show source in particle.py:15](https://github.com/Gorkowski/particula/blob/main/particula/next/particle.py#L15)

Calculates the mass of particles based on their distribution and
density.

#### Arguments

- distribution (NDArray[np.float64]): The distribution of particle
    sizes or masses.
- density (NDArray[np.float64]): The density of the particles.

#### Returns

- `-` *NDArray[np.float64]* - The mass of the particles.

#### Signature

```python
@abstractmethod
def get_mass(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```

### ParticleStrategy().get_radius

[Show source in particle.py:34](https://github.com/Gorkowski/particula/blob/main/particula/next/particle.py#L34)

Calculates the radius of particles based on their distribution and
density.

#### Arguments

- distribution (NDArray[np.float64]): The distribution of particle
    sizes or masses.
- density (NDArray[np.float64]): The density of the particles.

#### Returns

- `-` *NDArray[np.float64]* - The radius of the particles.

#### Signature

```python
@abstractmethod
def get_radius(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```

### ParticleStrategy().get_total_mass

[Show source in particle.py:53](https://github.com/Gorkowski/particula/blob/main/particula/next/particle.py#L53)

Calculates the total mass of particles based on their distribution,
concentration, and density.

#### Arguments

- distribution (NDArray[np.float64]): The distribution of particle
    sizes or masses.
- concentration (NDArray[np.float64]): The concentration of each
    particle size or mass in the distribution.
- density (NDArray[np.float64]): The density of the particles.

#### Returns

- `-` *np.float64* - The total mass of the particles.

#### Signature

```python
@abstractmethod
def get_total_mass(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    density: NDArray[np.float64],
) -> np.float64: ...
```



## RadiiBasedStrategy

[Show source in particle.py:112](https://github.com/Gorkowski/particula/blob/main/particula/next/particle.py#L112)

A strategy for particles represented by their radius (distribution),
and particle conentraiton. Implementing the ParticleStrategy interface.
This strategy calculates particle mass, radius, and total mass based on
the particle's radius, number concentraiton, and density.

#### Signature

```python
class RadiiBasedStrategy(ParticleStrategy): ...
```

#### See also

- [ParticleStrategy](#particlestrategy)

### RadiiBasedStrategy().get_mass

[Show source in particle.py:120](https://github.com/Gorkowski/particula/blob/main/particula/next/particle.py#L120)

#### Signature

```python
def get_mass(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```

### RadiiBasedStrategy().get_radius

[Show source in particle.py:129](https://github.com/Gorkowski/particula/blob/main/particula/next/particle.py#L129)

#### Signature

```python
def get_radius(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```

### RadiiBasedStrategy().get_total_mass

[Show source in particle.py:136](https://github.com/Gorkowski/particula/blob/main/particula/next/particle.py#L136)

#### Signature

```python
def get_total_mass(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    density: NDArray[np.float64],
) -> np.float64: ...
```



## SpeciatedMassStrategy

[Show source in particle.py:148](https://github.com/Gorkowski/particula/blob/main/particula/next/particle.py#L148)

Strategy for particles with speciated mass distribution.
Some particles may have different densities and their mass is
distributed across different species. This strategy calculates mass,
radius, and total mass based on the species at each mass, density,
the particle concentration.

#### Signature

```python
class SpeciatedMassStrategy(ParticleStrategy): ...
```

#### See also

- [ParticleStrategy](#particlestrategy)

### SpeciatedMassStrategy().get_mass

[Show source in particle.py:155](https://github.com/Gorkowski/particula/blob/main/particula/next/particle.py#L155)

Calculates the mass for each mass and species, leveraging densities
for adjustment.

#### Arguments

- distribution (NDArray[np.float64]): A 2D array with rows
    representing mass bins and columns representing species.
- densities (NDArray[np.float64]): An array of densities for each
    species.

#### Returns

- `-` *NDArray[np.float64]* - A 1D array of calculated masses for each mass
    bin. The sum of each column (species) in the distribution matrix.

#### Signature

```python
def get_mass(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```

### SpeciatedMassStrategy().get_radius

[Show source in particle.py:177](https://github.com/Gorkowski/particula/blob/main/particula/next/particle.py#L177)

Calculates the radius for each mass bin and species, based on the
volume derived from mass and density.

#### Arguments

- distribution (NDArray[np.float64]): A 2D array with rows representing
    mass bins and columns representing species.
- dinsity (NDArray[np.float64]): An array of densities for each
    species.

#### Returns

- `-` *NDArray[np.float64]* - A 1D array of calculated radii for each mass
    bin.

#### Signature

```python
def get_radius(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```

### SpeciatedMassStrategy().get_total_mass

[Show source in particle.py:200](https://github.com/Gorkowski/particula/blob/main/particula/next/particle.py#L200)

Calculates the total mass of all species, incorporating the
concentration of particles per species.

#### Arguments

- distribution (NDArray[np.float64]): The mass distribution matrix.
- counts (NDArray[np.float64]): A 1D array with elements representing
    the count of particles for each species.

#### Returns

- `-` *np.float64* - The total mass of all particles.

#### Signature

```python
def get_total_mass(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    density: NDArray[np.float64],
) -> np.float64: ...
```



## create_particle_strategy

[Show source in particle.py:223](https://github.com/Gorkowski/particula/blob/main/particula/next/particle.py#L223)

Factory function for creating instances of particle strategies based on
the specified representation type.

#### Arguments

- particle_type (str): The type of particle representation, determining
which strategy instance to create.
[mass_based, radii_based, speciated_mass]

#### Returns

- An instance of ParticleStrategy corresponding to the specified
    particle type.

#### Raises

- `-` *ValueError* - If an unknown particle type is specified.

#### Signature

```python
def create_particle_strategy(particle_representation: str) -> ParticleStrategy: ...
```

#### See also

- [ParticleStrategy](#particlestrategy)
