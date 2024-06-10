# Distribution Strategies

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Particles](./index.md#particles) / Distribution Strategies

> Auto-generated documentation for [particula.next.particles.distribution_strategies](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py) module.

## DistributionStrategy

[Show source in distribution_strategies.py:10](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L10)

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

[Show source in distribution_strategies.py:70](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L70)

Adds mass to the distribution of particles.

#### Arguments

- `distribution` - The distribution of particle sizes or masses.
- `concentration` - The concentration of each particle size or mass in
the distribution.
- `density` - The density of the particles.
- `added_mass` - The mass to be added per distribution bin.

#### Returns

- `NDArray[np.float_]` - The new concentration array.
- `NDArray[np.float_]` - The new distribution array.

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

[Show source in distribution_strategies.py:23](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L23)

Calculates the mass of the particles.

#### Arguments

- `distribution` - The distribution of particle sizes or masses.
- `density` - The density of the particles.

#### Returns

- `NDArray[np.float_]` - The mass of the particles.

#### Signature

```python
@abstractmethod
def get_mass(
    self, distribution: NDArray[np.float_], density: NDArray[np.float_]
) -> NDArray[np.float_]: ...
```

### DistributionStrategy().get_radius

[Show source in distribution_strategies.py:37](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L37)

Calculates the radius of the particles.

#### Arguments

- `distribution` - The distribution of particle sizes or masses.
- `density` - The density of the particles.

#### Returns

- `NDArray[np.float_]` - The radius of the particles.

#### Signature

```python
@abstractmethod
def get_radius(
    self, distribution: NDArray[np.float_], density: NDArray[np.float_]
) -> NDArray[np.float_]: ...
```

### DistributionStrategy().get_total_mass

[Show source in distribution_strategies.py:51](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L51)

Calculates the total mass of particles.

#### Arguments

- `distribution` - The distribution of particle sizes or masses.
- `concentration` - The concentration of each particle size or mass in
the distribution.
- `density` - The density of the particles.

#### Returns

- `np.float_` - The total mass of the particles.

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

[Show source in distribution_strategies.py:93](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L93)

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

[Show source in distribution_strategies.py:125](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L125)

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

[Show source in distribution_strategies.py:101](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L101)

#### Signature

```python
def get_mass(
    self, distribution: NDArray[np.float_], density: NDArray[np.float_]
) -> NDArray[np.float_]: ...
```

### MassBasedMovingBin().get_radius

[Show source in distribution_strategies.py:107](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L107)

#### Signature

```python
def get_radius(
    self, distribution: NDArray[np.float_], density: NDArray[np.float_]
) -> NDArray[np.float_]: ...
```

### MassBasedMovingBin().get_total_mass

[Show source in distribution_strategies.py:115](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L115)

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

[Show source in distribution_strategies.py:136](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L136)

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

[Show source in distribution_strategies.py:168](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L168)

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

[Show source in distribution_strategies.py:143](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L143)

#### Signature

```python
def get_mass(
    self, distribution: NDArray[np.float_], density: NDArray[np.float_]
) -> NDArray[np.float_]: ...
```

### RadiiBasedMovingBin().get_radius

[Show source in distribution_strategies.py:150](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L150)

#### Signature

```python
def get_radius(
    self, distribution: NDArray[np.float_], density: NDArray[np.float_]
) -> NDArray[np.float_]: ...
```

### RadiiBasedMovingBin().get_total_mass

[Show source in distribution_strategies.py:157](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L157)

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

[Show source in distribution_strategies.py:180](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L180)

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

[Show source in distribution_strategies.py:216](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L216)

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

[Show source in distribution_strategies.py:190](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L190)

#### Signature

```python
def get_mass(
    self, distribution: NDArray[np.float_], density: NDArray[np.float_]
) -> NDArray[np.float_]: ...
```

### SpeciatedMassMovingBin().get_radius

[Show source in distribution_strategies.py:198](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L198)

#### Signature

```python
def get_radius(
    self, distribution: NDArray[np.float_], density: NDArray[np.float_]
) -> NDArray[np.float_]: ...
```

### SpeciatedMassMovingBin().get_total_mass

[Show source in distribution_strategies.py:205](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L205)

#### Signature

```python
def get_total_mass(
    self,
    distribution: NDArray[np.float_],
    concentration: NDArray[np.float_],
    density: NDArray[np.float_],
) -> np.float_: ...
```
