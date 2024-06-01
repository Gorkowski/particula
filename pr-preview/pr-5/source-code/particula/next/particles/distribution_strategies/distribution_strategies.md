# Distribution Strategies

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Particles](./index.md#particles) / Distribution Strategies

> Auto-generated documentation for [particula.next.particles.distribution_strategies](../../../../../particula/next/particles/distribution_strategies.py) module.

## DistributionStrategy

[Show source in distribution_strategies.py:11](../../../../../particula/next/particles/distribution_strategies.py#L11)

Abstract base class for particle strategy, defining the common
interface for mass, radius, and total mass calculations for different
particle representations.

#### Signature

```python
class DistributionStrategy(ABC): ...
```

### DistributionStrategy().add_mass

[Show source in distribution_strategies.py:78](../../../../../particula/next/particles/distribution_strategies.py#L78)

Adds mass to the distribution of particles based on their distribution,
concentration, and density.

#### Arguments

- distribution (NDArray[np.float_]): The distribution representation
of particles
- concentration (NDArray[np.float_]): The concentration of each
particle in the distribution.
- density (NDArray[np.float_]): The density of the particles.
- added_mass (NDArray[np.float_]): The mass to be added per
distribution bin.

#### Returns

- `-` *NDArray[np.float_]* - The new concentration array.
- `-` *NDArray[np.float_]* - The new distribution array.

#### Signature

```python
@abstractmethod
def add_mass(
    self,
    distribution: NDArray[np.float_],
    concentration: NDArray[np.float_],
    density: NDArray[np.float_],
    added_mass: NDArray[np.float_],
) -> tuple[NDArray[np.float_], NDArray[np.float_]]: ...
```

### DistributionStrategy().get_mass

[Show source in distribution_strategies.py:18](../../../../../particula/next/particles/distribution_strategies.py#L18)

Calculates the mass of particles based on their distribution and
density.

#### Arguments

- distribution (NDArray[np.float_]): The distribution of particle
    sizes or masses.
- density (NDArray[np.float_]): The density of the particles.

#### Returns

- `-` *NDArray[np.float_]* - The mass of the particles.

#### Signature

```python
@abstractmethod
def get_mass(
    self, distribution: NDArray[np.float_], density: NDArray[np.float_]
) -> NDArray[np.float_]: ...
```

### DistributionStrategy().get_radius

[Show source in distribution_strategies.py:37](../../../../../particula/next/particles/distribution_strategies.py#L37)

Calculates the radius of particles based on their distribution and
density.

#### Arguments

- distribution (NDArray[np.float_]): The distribution of particle
    sizes or masses.
- density (NDArray[np.float_]): The density of the particles.

#### Returns

- `-` *NDArray[np.float_]* - The radius of the particles.

#### Signature

```python
@abstractmethod
def get_radius(
    self, distribution: NDArray[np.float_], density: NDArray[np.float_]
) -> NDArray[np.float_]: ...
```

### DistributionStrategy().get_total_mass

[Show source in distribution_strategies.py:56](../../../../../particula/next/particles/distribution_strategies.py#L56)

Calculates the total mass of particles based on their distribution,
concentration, and density.

#### Arguments

- distribution (NDArray[np.float_]): The distribution of particle
    sizes or masses.
- concentration (NDArray[np.float_]): The concentration of each
    particle size or mass in the distribution.
- density (NDArray[np.float_]): The density of the particles.

#### Returns

- `-` *np.float_* - The total mass of the particles.

#### Signature

```python
@abstractmethod
def get_total_mass(
    self,
    distribution: NDArray[np.float_],
    concentration: NDArray[np.float_],
    density: NDArray[np.float_],
) -> np.float_: ...
```



## MassBasedMovingBin

[Show source in distribution_strategies.py:105](../../../../../particula/next/particles/distribution_strategies.py#L105)

A strategy for particles represented by their mass distribution, and
particle number concentration. Moving the bins when adding mass.

#### Signature

```python
class MassBasedMovingBin(DistributionStrategy): ...
```

#### See also

- [DistributionStrategy](#distributionstrategy)

### MassBasedMovingBin().add_mass

[Show source in distribution_strategies.py:139](../../../../../particula/next/particles/distribution_strategies.py#L139)

#### Signature

```python
def add_mass(
    self,
    distribution: NDArray[np.float_],
    concentration: NDArray[np.float_],
    density: NDArray[np.float_],
    added_mass: NDArray[np.float_],
) -> tuple[NDArray[np.float_], NDArray[np.float_]]: ...
```

### MassBasedMovingBin().get_mass

[Show source in distribution_strategies.py:111](../../../../../particula/next/particles/distribution_strategies.py#L111)

#### Signature

```python
def get_mass(
    self, distribution: NDArray[np.float_], density: NDArray[np.float_]
) -> NDArray[np.float_]: ...
```

### MassBasedMovingBin().get_radius

[Show source in distribution_strategies.py:119](../../../../../particula/next/particles/distribution_strategies.py#L119)

#### Signature

```python
def get_radius(
    self, distribution: NDArray[np.float_], density: NDArray[np.float_]
) -> NDArray[np.float_]: ...
```

### MassBasedMovingBin().get_total_mass

[Show source in distribution_strategies.py:129](../../../../../particula/next/particles/distribution_strategies.py#L129)

#### Signature

```python
def get_total_mass(
    self,
    distribution: NDArray[np.float_],
    concentration: NDArray[np.float_],
    density: NDArray[np.float_],
) -> np.float_: ...
```



## RadiiBasedMovingBin

[Show source in distribution_strategies.py:150](../../../../../particula/next/particles/distribution_strategies.py#L150)

A strategy for particles represented by their radius (distribution),
and particle concentration. Implementing the DistributionStrategy
interface.
This strategy calculates particle mass, radius, and total mass based on
the particle's radius, number concentration, and density.

#### Signature

```python
class RadiiBasedMovingBin(DistributionStrategy): ...
```

#### See also

- [DistributionStrategy](#distributionstrategy)

### RadiiBasedMovingBin().add_mass

[Show source in distribution_strategies.py:186](../../../../../particula/next/particles/distribution_strategies.py#L186)

#### Signature

```python
def add_mass(
    self,
    distribution: NDArray[np.float_],
    concentration: NDArray[np.float_],
    density: NDArray[np.float_],
    added_mass: NDArray[np.float_],
) -> tuple[NDArray[np.float_], NDArray[np.float_]]: ...
```

### RadiiBasedMovingBin().get_mass

[Show source in distribution_strategies.py:159](../../../../../particula/next/particles/distribution_strategies.py#L159)

#### Signature

```python
def get_mass(
    self, distribution: NDArray[np.float_], density: NDArray[np.float_]
) -> NDArray[np.float_]: ...
```

### RadiiBasedMovingBin().get_radius

[Show source in distribution_strategies.py:168](../../../../../particula/next/particles/distribution_strategies.py#L168)

#### Signature

```python
def get_radius(
    self, distribution: NDArray[np.float_], density: NDArray[np.float_]
) -> NDArray[np.float_]: ...
```

### RadiiBasedMovingBin().get_total_mass

[Show source in distribution_strategies.py:175](../../../../../particula/next/particles/distribution_strategies.py#L175)

#### Signature

```python
def get_total_mass(
    self,
    distribution: NDArray[np.float_],
    concentration: NDArray[np.float_],
    density: NDArray[np.float_],
) -> np.float_: ...
```



## SpeciatedMassMovingBin

[Show source in distribution_strategies.py:198](../../../../../particula/next/particles/distribution_strategies.py#L198)

Strategy for particles with speciated mass distribution.
Some particles may have different densities and their mass is
distributed across different species. This strategy calculates mass,
radius, and total mass based on the species at each mass, density,
the particle concentration.

#### Signature

```python
class SpeciatedMassMovingBin(DistributionStrategy): ...
```

#### See also

- [DistributionStrategy](#distributionstrategy)

### SpeciatedMassMovingBin().add_mass

[Show source in distribution_strategies.py:274](../../../../../particula/next/particles/distribution_strategies.py#L274)

#### Signature

```python
def add_mass(
    self,
    distribution: NDArray[np.float_],
    concentration: NDArray[np.float_],
    density: NDArray[np.float_],
    added_mass: NDArray[np.float_],
) -> tuple[NDArray[np.float_], NDArray[np.float_]]: ...
```

### SpeciatedMassMovingBin().get_mass

[Show source in distribution_strategies.py:205](../../../../../particula/next/particles/distribution_strategies.py#L205)

Calculates the mass for each mass and species, leveraging densities
for adjustment.

#### Arguments

- distribution (NDArray[np.float_]): A 2D array with rows
    representing mass bins and columns representing species.
- densities (NDArray[np.float_]): An array of densities for each
    species.

#### Returns

- `-` *NDArray[np.float_]* - A 1D array of calculated masses for each mass
    bin. The sum of each column (species) in the distribution matrix.

#### Signature

```python
def get_mass(
    self, distribution: NDArray[np.float_], density: NDArray[np.float_]
) -> NDArray[np.float_]: ...
```

### SpeciatedMassMovingBin().get_radius

[Show source in distribution_strategies.py:229](../../../../../particula/next/particles/distribution_strategies.py#L229)

Calculates the radius for each mass bin and species, based on the
volume derived from mass and density.

#### Arguments

- distribution (NDArray[np.float_]): A 2D array with rows representing
    mass bins and columns representing species.
- density (NDArray[np.float_]): An array of densities for each
    species.

#### Returns

- `-` *NDArray[np.float_]* - A 1D array of calculated radii for each mass
    bin.

#### Signature

```python
def get_radius(
    self, distribution: NDArray[np.float_], density: NDArray[np.float_]
) -> NDArray[np.float_]: ...
```

### SpeciatedMassMovingBin().get_total_mass

[Show source in distribution_strategies.py:252](../../../../../particula/next/particles/distribution_strategies.py#L252)

Calculates the total mass of all species, incorporating the
concentration of particles per species.

#### Arguments

- distribution (NDArray[np.float_]): The mass distribution matrix.
- counts (NDArray[np.float_]): A 1D array with elements representing
    the count of particles for each species.

#### Returns

- `-` *np.float_* - The total mass of all particles.

#### Signature

```python
def get_total_mass(
    self,
    distribution: NDArray[np.float_],
    concentration: NDArray[np.float_],
    density: NDArray[np.float_],
) -> np.float_: ...
```
