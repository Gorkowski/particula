# Strategy

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Next](../../index.md#next) / [Dynamics](../index.md#dynamics) / [Coagulation](./index.md#coagulation) / Strategy

> Auto-generated documentation for [particula.next.dynamics.coagulation.strategy](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py) module.

## CoagulationStrategy

[Show source in strategy.py:20](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L20)

Abstract class for defining a coagulation strategy. This class defines the
methods that must be implemented by any coagulation strategy.

#### Methods

--------
- kernel (abstractmethod): Calculate the coagulation kernel.
- `-` *loss_rate* - Calculate the coagulation loss rate.
- `-` *gain_rate* - Calculate the coagulation gain rate.
- `-` *net_rate* - Calculate the net coagulation rate.
- `-` *diffusive_knudsen* - Calculate the diffusive Knudsen number.
- `-` *coulomb_potential_ratio* - Calculate the Coulomb potential ratio.

#### Signature

```python
class CoagulationStrategy(ABC): ...
```

### CoagulationStrategy().coulomb_potential_ratio

[Show source in strategy.py:198](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L198)

Calculate the Coulomb potential ratio based on the particle properties
and temperature.

#### Arguments

-----
- particle (Particle class): The particles for which the Coulomb
potential ratio is to be calculated.
- temperature (float): The temperature of the gas phase [K].

#### Returns

--------
- `-` *NDArray[np.float64]* - The Coulomb potential ratio for the particle
[dimensionless].

#### Signature

```python
def coulomb_potential_ratio(
    self, particle: ParticleRepresentation, temperature: float
) -> NDArray[np.float64]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### CoagulationStrategy().diffusive_knudsen

[Show source in strategy.py:160](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L160)

Calculate the diffusive Knudsen number based on the particle
properties, temperature, and pressure.

#### Arguments

-----
- particle (Particle class): The particle for which the diffusive
Knudsen number is to be calculated.
- temperature (float): The temperature of the gas phase [K].
- pressure (float): The pressure of the gas phase [Pa].

#### Returns

--------
- `-` *NDArray[np.float64]* - The diffusive Knudsen number for the particle
[dimensionless].

#### Signature

```python
def diffusive_knudsen(
    self, particle: ParticleRepresentation, temperature: float, pressure: float
) -> NDArray[np.float64]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### CoagulationStrategy().dimensionless_kernel

[Show source in strategy.py:35](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L35)

Calculate the dimensionless coagulation kernel based on the particle
properties interactions,
diffusive Knudsen number and Coulomb potential

#### Arguments

-----
- diffusive_knudsen (NDArray[np.float64]): The diffusive Knudsen number
for the particle [dimensionless].
- coulomb_potential_ratio (NDArray[np.float64]): The Coulomb potential
ratio for the particle [dimensionless].

#### Returns

--------
- `-` *NDArray[np.float64]* - The dimensionless coagulation kernel for the
particle [dimensionless].

#### Signature

```python
@abstractmethod
def dimensionless_kernel(
    self,
    diffusive_knudsen: NDArray[np.float64],
    coulomb_potential_ratio: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```

### CoagulationStrategy().friction_factor

[Show source in strategy.py:222](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L222)

Calculate the friction factor based on the particle properties,
temperature, and pressure.

#### Arguments

-----
- particle (Particle class): The particle for which the friction factor
is to be calculated.
- temperature (float): The temperature of the gas phase [K].
- pressure (float): The pressure of the gas phase [Pa].

#### Returns

--------
- `-` *NDArray[np.float64]* - The friction factor for the particle
[dimensionless].

#### Signature

```python
def friction_factor(
    self, particle: ParticleRepresentation, temperature: float, pressure: float
) -> NDArray[np.float64]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### CoagulationStrategy().gain_rate

[Show source in strategy.py:104](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L104)

Calculate the coagulation gain rate based on the particle radius,
distribution, and the coagulation kernel.

#### Arguments

-----
- particle (Particle class): The particle for which the coagulation
gain rate is to be calculated.
- kernel (NDArray[np.float64]): The coagulation kernel.

#### Returns

--------
- Union[float, NDArray[np.float64]]: The coagulation gain rate for the
particle [kg/s].

#### Notes

------
May be abstracted to a separate module when different coagulation
strategies are implemented (super droplet).

#### Signature

```python
@abstractmethod
def gain_rate(
    self, particle: ParticleRepresentation, kernel: NDArray[np.float64]
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### CoagulationStrategy().kernel

[Show source in strategy.py:59](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L59)

Calculate the coagulation kernel based on the particle properties,
temperature, and pressure.

#### Arguments

-----
- particle (Particle class): The particle for which the coagulation
kernel is to be calculated.
- temperature (float): The temperature of the gas phase [K].
- pressure (float): The pressure of the gas phase [Pa].

#### Returns

--------
- `-` *NDArray[np.float64]* - The coagulation kernel for the particle [m^3/s].

#### Signature

```python
@abstractmethod
def kernel(
    self, particle: ParticleRepresentation, temperature: float, pressure: float
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### CoagulationStrategy().loss_rate

[Show source in strategy.py:82](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L82)

Calculate the coagulation loss rate based on the particle radius,
distribution, and the coagulation kernel.

#### Arguments

-----
- particle (Particle class): The particle for which the coagulation
loss rate is to be calculated.
- kernel (NDArray[np.float64]): The coagulation kernel.

#### Returns

--------
- Union[float, NDArray[np.float64]]: The coagulation loss rate for the
particle [kg/s].

#### Signature

```python
@abstractmethod
def loss_rate(
    self, particle: ParticleRepresentation, kernel: NDArray[np.float64]
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### CoagulationStrategy().net_rate

[Show source in strategy.py:131](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L131)

Calculate the net coagulation rate based on the particle radius,
distribution, and the coagulation kernel.

#### Arguments

-----
- particle (Particle class): The particle class for which the
coagulation net rate is to be calculated.
- temperature (float): The temperature of the gas phase [K].
- pressure (float): The pressure of the gas phase [Pa].

#### Returns

--------
- Union[float, NDArray[np.float64]]: The net coagulation rate for the
particle [kg/s].

#### Signature

```python
def net_rate(
    self, particle: ParticleRepresentation, temperature: float, pressure: float
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)



## ContinuousGeneralPDF

[Show source in strategy.py:421](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L421)

Continuous PDF coagulation strategy class. This class implements the
methods defined in the CoagulationStrategy abstract class. The kernel
strategy is passed as an argument to the class, should use a dimensionless
kernel representation.

#### Methods

--------
- `-` *kernel* - Calculate the coagulation kernel.
- `-` *loss_rate* - Calculate the coagulation loss rate.
- `-` *gain_rate* - Calculate the coagulation gain rate.
- `-` *net_rate* - Calculate the net coagulation rate.

#### Signature

```python
class ContinuousGeneralPDF(CoagulationStrategy):
    def __init__(self, kernel_strategy: KernelStrategy): ...
```

#### See also

- [CoagulationStrategy](#coagulationstrategy)
- [KernelStrategy](./kernel.md#kernelstrategy)

### ContinuousGeneralPDF().dimensionless_kernel

[Show source in strategy.py:439](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L439)

#### Signature

```python
def dimensionless_kernel(
    self,
    diffusive_knudsen: NDArray[np.float64],
    coulomb_potential_ratio: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```

### ContinuousGeneralPDF().gain_rate

[Show source in strategy.py:496](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L496)

#### Signature

```python
def gain_rate(
    self, particle: ParticleRepresentation, kernel: NDArray[np.float64]
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### ContinuousGeneralPDF().kernel

[Show source in strategy.py:449](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L449)

#### Signature

```python
def kernel(
    self, particle: ParticleRepresentation, temperature: float, pressure: float
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### ContinuousGeneralPDF().loss_rate

[Show source in strategy.py:484](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L484)

#### Signature

```python
def loss_rate(
    self, particle: ParticleRepresentation, kernel: NDArray[np.float64]
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)



## DiscreteGeneral

[Show source in strategy.py:329](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L329)

Discrete general coagulation strategy class. This class implements the
methods defined in the CoagulationStrategy abstract class. The kernel
strategy is passed as an argument to the class, to use a dimensionless
kernel representation.

#### Attributes

-----------
- `-` *kernel_strategy* - The kernel strategy to be used for the coagulation, from
the KernelStrategy class.

#### Methods

--------
- `-` *kernel* - Calculate the coagulation kernel.
- `-` *loss_rate* - Calculate the coagulation loss rate.
- `-` *gain_rate* - Calculate the coagulation gain rate.
- `-` *net_rate* - Calculate the net coagulation rate.

#### Signature

```python
class DiscreteGeneral(CoagulationStrategy):
    def __init__(self, kernel_strategy: KernelStrategy): ...
```

#### See also

- [CoagulationStrategy](#coagulationstrategy)
- [KernelStrategy](./kernel.md#kernelstrategy)

### DiscreteGeneral().dimensionless_kernel

[Show source in strategy.py:352](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L352)

#### Signature

```python
def dimensionless_kernel(
    self,
    diffusive_knudsen: NDArray[np.float64],
    coulomb_potential_ratio: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```

### DiscreteGeneral().gain_rate

[Show source in strategy.py:408](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L408)

#### Signature

```python
def gain_rate(
    self, particle: ParticleRepresentation, kernel: NDArray[np.float64]
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### DiscreteGeneral().kernel

[Show source in strategy.py:362](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L362)

#### Signature

```python
def kernel(
    self, particle: ParticleRepresentation, temperature: float, pressure: float
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### DiscreteGeneral().loss_rate

[Show source in strategy.py:397](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L397)

#### Signature

```python
def loss_rate(
    self, particle: ParticleRepresentation, kernel: NDArray[np.float64]
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)



## DiscreteSimple

[Show source in strategy.py:267](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L267)

Discrete Brownian coagulation strategy class. This class implements the
methods defined in the CoagulationStrategy abstract class.

#### Methods

--------
- `-` *kernel* - Calculate the coagulation kernel.
- `-` *loss_rate* - Calculate the coagulation loss rate.
- `-` *gain_rate* - Calculate the coagulation gain rate.
- `-` *net_rate* - Calculate the net coagulation rate.

#### Signature

```python
class DiscreteSimple(CoagulationStrategy): ...
```

#### See also

- [CoagulationStrategy](#coagulationstrategy)

### DiscreteSimple().dimensionless_kernel

[Show source in strategy.py:280](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L280)

#### Signature

```python
def dimensionless_kernel(
    self,
    diffusive_knudsen: NDArray[np.float64],
    coulomb_potential_ratio: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```

### DiscreteSimple().gain_rate

[Show source in strategy.py:316](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L316)

#### Signature

```python
def gain_rate(
    self, particle: ParticleRepresentation, kernel: NDArray[np.float64]
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### DiscreteSimple().kernel

[Show source in strategy.py:291](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L291)

#### Signature

```python
def kernel(
    self, particle: ParticleRepresentation, temperature: float, pressure: float
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### DiscreteSimple().loss_rate

[Show source in strategy.py:305](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L305)

#### Signature

```python
def loss_rate(
    self, particle: ParticleRepresentation, kernel: NDArray[np.float64]
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)
