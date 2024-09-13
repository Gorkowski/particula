# Distribution Strategies

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Particles](./index.md#particles) / Distribution Strategies

> Auto-generated documentation for [particula.next.particles.distribution_strategies](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py) module.

## DistributionStrategy

[Show source in distribution_strategies.py:13](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L13)

Abstract base class for particle strategy, defining the common
interface for mass, radius, and total mass calculations for different
particle representations.

#### Methods

- `get_name` - Returns the type of the distribution strategy.
- `get_mass` - Calculates the mass of particles.
- `get_radius` - Calculates the radius of particles.
- `get_total_mass` - Calculates the total mass of particles.
- `add_mass` - Adds mass to the distribution of particles.

#### Signature

```python
class DistributionStrategy(ABC): ...
```

### DistributionStrategy().add_mass

[Show source in distribution_strategies.py:98](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L98)

Adds mass to the distribution of particles.

#### Arguments

- `distribution` - The distribution of particle sizes or masses.
- `concentration` - The concentration of each particle size or mass in
    the distribution.
- `density` - The density of the particles.
- `added_mass` - The mass to be added per distribution bin.

#### Returns

- `NDArray[np.float64]` - The new concentration array.
- `NDArray[np.float64]` - The new distribution array.

#### Signature

```python
@abstractmethod
def add_mass(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    density: NDArray[np.float64],
    added_mass: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```

### DistributionStrategy().collide_pairs

[Show source in distribution_strategies.py:120](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L120)

Collides index pairs.

#### Arguments

- `distribution` - The distribution of particle sizes or masses.
- `concentration` - The concentration of each particle size or mass in
    the distribution.
- `density` - The density of the particles.
- `indices` - The indices of the particles to collide.

#### Returns

- `NDArray[np.float64]` - The new concentration array.
- `NDArray[np.float64]` - The new distribution array.

#### Signature

```python
@abstractmethod
def collide_pairs(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    density: NDArray[np.float64],
    indices: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```

### DistributionStrategy().get_mass

[Show source in distribution_strategies.py:45](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L45)

Calculates the mass of the particles (or bin).

#### Arguments

- `distribution` - The distribution of particle sizes or masses.
- `density` - The density of the particles.

#### Returns

- `NDArray[np.float64]` - The mass of the particles.

#### Signature

```python
def get_mass(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```

### DistributionStrategy().get_name

[Show source in distribution_strategies.py:27](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L27)

Return the type of the distribution strategy.

#### Signature

```python
def get_name(self) -> str: ...
```

### DistributionStrategy().get_radius

[Show source in distribution_strategies.py:84](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L84)

Calculates the radius of the particles.

#### Arguments

- `distribution` - The distribution of particle sizes or masses.
- `density` - The density of the particles.

#### Returns

- `NDArray[np.float64]` - The radius of the particles.

#### Signature

```python
@abstractmethod
def get_radius(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```

### DistributionStrategy().get_species_mass

[Show source in distribution_strategies.py:31](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L31)

The mass per species in the particles (or bin).

#### Arguments

- `distribution` - The distribution of particle sizes or masses.
- `density` - The density of the particles.

#### Returns

- `NDArray[np.float64]` - The mass of the particles

#### Signature

```python
@abstractmethod
def get_species_mass(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```

### DistributionStrategy().get_total_mass

[Show source in distribution_strategies.py:63](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L63)

Calculates the total mass of all particles (or bin).

#### Arguments

- `distribution` - The distribution of particle sizes or masses.
- `concentration` - The concentration of each particle size or mass in
the distribution.
- `density` - The density of the particles.

#### Returns

- `np.float64` - The total mass of the particles.

#### Signature

```python
def get_total_mass(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    density: NDArray[np.float64],
) -> np.float64: ...
```



## MassBasedMovingBin

[Show source in distribution_strategies.py:143](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L143)

A strategy for particles represented by their mass distribution.

This strategy calculates particle mass, radius, and total mass based on
the particle's mass, number concentration, and density. It also moves the
bins when adding mass to the distribution.

#### Signature

```python
class MassBasedMovingBin(DistributionStrategy): ...
```

#### See also

- [DistributionStrategy](#distributionstrategy)

### MassBasedMovingBin().add_mass

[Show source in distribution_strategies.py:165](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L165)

#### Signature

```python
def add_mass(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    density: NDArray[np.float64],
    added_mass: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```

### MassBasedMovingBin().collide_pairs

[Show source in distribution_strategies.py:175](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L175)

#### Signature

```python
def collide_pairs(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    density: NDArray[np.float64],
    indices: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```

### MassBasedMovingBin().get_radius

[Show source in distribution_strategies.py:157](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L157)

#### Signature

```python
def get_radius(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```

### MassBasedMovingBin().get_species_mass

[Show source in distribution_strategies.py:151](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L151)

#### Signature

```python
def get_species_mass(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```



## ParticleResolvedSpeciatedMass

[Show source in distribution_strategies.py:303](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L303)

Strategy for resolved particles via speciated mass.

Strategy for resolved particles with speciated mass.
Particles may have different densities and their mass is
distributed across different species. This strategy calculates mass,
radius, and total mass based on the species at each mass, density,
the particle concentration.

#### Signature

```python
class ParticleResolvedSpeciatedMass(DistributionStrategy): ...
```

#### See also

- [DistributionStrategy](#distributionstrategy)

### ParticleResolvedSpeciatedMass().add_mass

[Show source in distribution_strategies.py:328](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L328)

#### Signature

```python
def add_mass(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    density: NDArray[np.float64],
    added_mass: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```

### ParticleResolvedSpeciatedMass().collide_pairs

[Show source in distribution_strategies.py:347](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L347)

#### Signature

```python
def collide_pairs(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    density: NDArray[np.float64],
    indices: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```

### ParticleResolvedSpeciatedMass().get_radius

[Show source in distribution_strategies.py:318](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L318)

#### Signature

```python
def get_radius(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```

### ParticleResolvedSpeciatedMass().get_species_mass

[Show source in distribution_strategies.py:313](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L313)

#### Signature

```python
def get_species_mass(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```



## RadiiBasedMovingBin

[Show source in distribution_strategies.py:190](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L190)

A strategy for particles represented by their radius.

This strategy calculates particle mass, radius, and total mass based on
the particle's radius, number concentration, and density.

#### Signature

```python
class RadiiBasedMovingBin(DistributionStrategy): ...
```

#### See also

- [DistributionStrategy](#distributionstrategy)

### RadiiBasedMovingBin().add_mass

[Show source in distribution_strategies.py:211](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L211)

#### Signature

```python
def add_mass(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    density: NDArray[np.float64],
    added_mass: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```

### RadiiBasedMovingBin().collide_pairs

[Show source in distribution_strategies.py:229](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L229)

#### Signature

```python
def collide_pairs(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    density: NDArray[np.float64],
    indices: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```

### RadiiBasedMovingBin().get_radius

[Show source in distribution_strategies.py:204](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L204)

#### Signature

```python
def get_radius(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```

### RadiiBasedMovingBin().get_species_mass

[Show source in distribution_strategies.py:197](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L197)

#### Signature

```python
def get_species_mass(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```



## SpeciatedMassMovingBin

[Show source in distribution_strategies.py:244](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L244)

Strategy for particles with speciated mass distribution.

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

[Show source in distribution_strategies.py:266](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L266)

#### Signature

```python
def add_mass(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    density: NDArray[np.float64],
    added_mass: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```

### SpeciatedMassMovingBin().collide_pairs

[Show source in distribution_strategies.py:288](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L288)

#### Signature

```python
def collide_pairs(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    density: NDArray[np.float64],
    indices: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```

### SpeciatedMassMovingBin().get_radius

[Show source in distribution_strategies.py:259](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L259)

#### Signature

```python
def get_radius(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```

### SpeciatedMassMovingBin().get_species_mass

[Show source in distribution_strategies.py:254](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L254)

#### Signature

```python
def get_species_mass(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```
