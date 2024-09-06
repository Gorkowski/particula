# Distribution Strategies

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Particles](./index.md#particles) / Distribution Strategies

> Auto-generated documentation for [particula.next.particles.distribution_strategies](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py) module.

## DistributionStrategy

[Show source in distribution_strategies.py:13](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L13)

Abstract base class for particle strategy, defining the common
interface for mass, radius, and total mass calculations for different
particle representations.

#### Methods

- `get_mass` - Calculates the mass of particles.
- `get_radius` - Calculates the radius of particles.
- `get_total_mass` - Calculates the total mass of particles.
- `add_mass` - Adds mass to the distribution of particles.

#### Signature

```python
class DistributionStrategy(ABC): ...
```

### DistributionStrategy().add_mass

[Show source in distribution_strategies.py:73](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L73)

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

[Show source in distribution_strategies.py:95](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L95)

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

[Show source in distribution_strategies.py:26](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L26)

Calculates the mass of the particles.

#### Arguments

- `distribution` - The distribution of particle sizes or masses.
- `density` - The density of the particles.

#### Returns

- `NDArray[np.float64]` - The mass of the particles.

#### Signature

```python
@abstractmethod
def get_mass(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```

### DistributionStrategy().get_radius

[Show source in distribution_strategies.py:40](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L40)

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

### DistributionStrategy().get_total_mass

[Show source in distribution_strategies.py:54](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L54)

Calculates the total mass of particles.

#### Arguments

- `distribution` - The distribution of particle sizes or masses.
- `concentration` - The concentration of each particle size or mass in
the distribution.
- `density` - The density of the particles.

#### Returns

- `np.float64` - The total mass of the particles.

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



## MassBasedMovingBin

[Show source in distribution_strategies.py:118](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L118)

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

[Show source in distribution_strategies.py:150](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L150)

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

[Show source in distribution_strategies.py:160](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L160)

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

### MassBasedMovingBin().get_mass

[Show source in distribution_strategies.py:126](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L126)

#### Signature

```python
def get_mass(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```

### MassBasedMovingBin().get_radius

[Show source in distribution_strategies.py:132](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L132)

#### Signature

```python
def get_radius(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```

### MassBasedMovingBin().get_total_mass

[Show source in distribution_strategies.py:140](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L140)

#### Signature

```python
def get_total_mass(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    density: NDArray[np.float64],
) -> np.float64: ...
```



## ParticleResolvedSpeciatedMass

[Show source in distribution_strategies.py:316](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L316)

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

[Show source in distribution_strategies.py:352](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L352)

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

[Show source in distribution_strategies.py:371](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L371)

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

### ParticleResolvedSpeciatedMass().get_mass

[Show source in distribution_strategies.py:326](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L326)

#### Signature

```python
def get_mass(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```

### ParticleResolvedSpeciatedMass().get_radius

[Show source in distribution_strategies.py:334](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L334)

#### Signature

```python
def get_radius(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```

### ParticleResolvedSpeciatedMass().get_total_mass

[Show source in distribution_strategies.py:341](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L341)

#### Signature

```python
def get_total_mass(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    density: NDArray[np.float64],
) -> np.float64: ...
```



## RadiiBasedMovingBin

[Show source in distribution_strategies.py:175](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L175)

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

[Show source in distribution_strategies.py:207](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L207)

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

[Show source in distribution_strategies.py:227](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L227)

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

### RadiiBasedMovingBin().get_mass

[Show source in distribution_strategies.py:182](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L182)

#### Signature

```python
def get_mass(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```

### RadiiBasedMovingBin().get_radius

[Show source in distribution_strategies.py:189](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L189)

#### Signature

```python
def get_radius(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```

### RadiiBasedMovingBin().get_total_mass

[Show source in distribution_strategies.py:196](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L196)

#### Signature

```python
def get_total_mass(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    density: NDArray[np.float64],
) -> np.float64: ...
```



## SpeciatedMassMovingBin

[Show source in distribution_strategies.py:242](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L242)

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

[Show source in distribution_strategies.py:277](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L277)

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

[Show source in distribution_strategies.py:301](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L301)

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

### SpeciatedMassMovingBin().get_mass

[Show source in distribution_strategies.py:252](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L252)

#### Signature

```python
def get_mass(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```

### SpeciatedMassMovingBin().get_radius

[Show source in distribution_strategies.py:260](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L260)

#### Signature

```python
def get_radius(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```

### SpeciatedMassMovingBin().get_total_mass

[Show source in distribution_strategies.py:267](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L267)

#### Signature

```python
def get_total_mass(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    density: NDArray[np.float64],
) -> np.float64: ...
```
