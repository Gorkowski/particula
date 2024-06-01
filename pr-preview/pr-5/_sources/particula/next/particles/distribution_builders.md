# Distribution Builders

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Particles](./index.md#particles) / Distribution Builders

> Auto-generated documentation for [particula.next.particles.distribution_builders](../../../../particula/next/particles/distribution_builders.py) module.

- [Distribution Builders](#distribution-builders)
  - [MassBasedMovingBinBuilder](#massbasedmovingbinbuilder)
    - [MassBasedMovingBinBuilder().build](#massbasedmovingbinbuilder()build)
  - [RadiiBasedMovingBinBuilder](#radiibasedmovingbinbuilder)
    - [RadiiBasedMovingBinBuilder().build](#radiibasedmovingbinbuilder()build)
  - [SpeciatedMassMovingBinBuilder](#speciatedmassmovingbinbuilder)
    - [SpeciatedMassMovingBinBuilder().build](#speciatedmassmovingbinbuilder()build)

## MassBasedMovingBinBuilder

[Show source in distribution_builders.py:14](../../../../particula/next/particles/distribution_builders.py#L14)

Builds a MassBasedMovingBin instance.

#### Signature

```python
class MassBasedMovingBinBuilder(BuilderABC):
    def __init__(self) -> None: ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)

### MassBasedMovingBinBuilder().build

[Show source in distribution_builders.py:21](../../../../particula/next/particles/distribution_builders.py#L21)

Builds a MassBasedMovingBin instance.

#### Signature

```python
def build(self) -> MassBasedMovingBin: ...
```

#### See also

- [MassBasedMovingBin](./distribution_strategies.md#massbasedmovingbin)



## RadiiBasedMovingBinBuilder

[Show source in distribution_builders.py:26](../../../../particula/next/particles/distribution_builders.py#L26)

Builds a RadiiBasedMovingBin instance.

#### Signature

```python
class RadiiBasedMovingBinBuilder(BuilderABC):
    def __init__(self) -> None: ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)

### RadiiBasedMovingBinBuilder().build

[Show source in distribution_builders.py:33](../../../../particula/next/particles/distribution_builders.py#L33)

Builds a RadiiBasedMovingBin instance.

#### Signature

```python
def build(self) -> RadiiBasedMovingBin: ...
```

#### See also

- [RadiiBasedMovingBin](./distribution_strategies.md#radiibasedmovingbin)



## SpeciatedMassMovingBinBuilder

[Show source in distribution_builders.py:38](../../../../particula/next/particles/distribution_builders.py#L38)

Builds a SpeciatedMassMovingBin instance.

#### Signature

```python
class SpeciatedMassMovingBinBuilder(BuilderABC):
    def __init__(self) -> None: ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)

### SpeciatedMassMovingBinBuilder().build

[Show source in distribution_builders.py:45](../../../../particula/next/particles/distribution_builders.py#L45)

Builds a SpeciatedMassMovingBin instance.

#### Signature

```python
def build(self) -> SpeciatedMassMovingBin: ...
```

#### See also

- [SpeciatedMassMovingBin](./distribution_strategies.md#speciatedmassmovingbin)