# Particle Process

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Dynamics](./index.md#dynamics) / Particle Process

> Auto-generated documentation for [particula.next.dynamics.particle_process](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/particle_process.py) module.

## Coagulation

[Show source in particle_process.py:113](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/particle_process.py#L113)

A class for running a coagulation strategy.

#### Arguments

-----
- coagulation_strategy (CoagulationStrategy): The coagulation strategy to
use.

#### Methods

--------
- `-` *execute* - Execute the coagulation process.
- `-` *rate* - Calculate the rate of coagulation for each particle.

#### Signature

```python
class Coagulation(Runnable):
    def __init__(self, coagulation_strategy: CoagulationStrategy): ...
```

#### See also

- [CoagulationStrategy](coagulation/strategy.md#coagulationstrategy)
- [Runnable](../runnable.md#runnable)

### Coagulation().execute

[Show source in particle_process.py:131](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/particle_process.py#L131)

Execute the coagulation process.

#### Arguments

-----
- aerosol (Aerosol): The aerosol instance to modify.

#### Signature

```python
def execute(self, aerosol: Aerosol, time_step: float) -> Aerosol: ...
```

#### See also

- [Aerosol](../aerosol.md#aerosol)

### Coagulation().rate

[Show source in particle_process.py:151](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/particle_process.py#L151)

Calculate the rate of coagulation for each particle.

#### Arguments

-----
- aerosol (Aerosol): The aerosol instance to modify.

#### Returns

--------
- `-` *np.ndarray* - An array of coagulation rates for each particle.

#### Signature

```python
def rate(self, aerosol: Aerosol) -> Any: ...
```

#### See also

- [Aerosol](../aerosol.md#aerosol)



## MassCondensation

[Show source in particle_process.py:15](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/particle_process.py#L15)

A class for running a mass condensation process.

#### Arguments

-----
- condensation_strategy (CondensationStrategy): The condensation strategy
to use.

#### Methods

--------
- `-` *execute* - Execute the mass condensation process.
- `-` *rate* - Calculate the rate of mass condensation for each particle due to
each condensable gas species.

#### Signature

```python
class MassCondensation(Runnable):
    def __init__(self, condensation_strategy: CondensationStrategy): ...
```

#### See also

- [CondensationStrategy](./condensation.md#condensationstrategy)
- [Runnable](../runnable.md#runnable)

### MassCondensation().execute

[Show source in particle_process.py:34](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/particle_process.py#L34)

Execute the mass condensation process.

#### Arguments

-----
- aerosol (Aerosol): The aerosol instance to modify.

#### Signature

```python
def execute(self, aerosol: Aerosol, time_step: float) -> Aerosol: ...
```

#### See also

- [Aerosol](../aerosol.md#aerosol)

### MassCondensation().rate

[Show source in particle_process.py:76](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/particle_process.py#L76)

Calculate the rate of mass condensation for each particle due to
each condensable gas species.

#### Arguments

-----
- aerosol (Aerosol): The aerosol instance to modify.

#### Returns

--------
- `-` *np.ndarray* - An array of condensation rates for each particle.

#### Signature

```python
def rate(self, aerosol: Aerosol) -> Any: ...
```

#### See also

- [Aerosol](../aerosol.md#aerosol)