# Strategy

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Next](../../index.md#next) / [Dynamics](../index.md#dynamics) / [Coagulation](./index.md#coagulation) / Strategy

> Auto-generated documentation for [particula.next.dynamics.coagulation.strategy](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py) module.

## CoagulationStrategy

[Show source in strategy.py:26](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L26)

Abstract class for defining a coagulation strategy. This class defines the
methods that must be implemented by any coagulation strategy.

#### Methods

- `kernel` - Calculate the coagulation kernel.
- `loss_rate` - Calculate the coagulation loss rate.
- `gain_rate` - Calculate the coagulation gain rate.
- `net_rate` - Calculate the net coagulation rate.
- `diffusive_knudsen` - Calculate the diffusive Knudsen number.
- `coulomb_potential_ratio` - Calculate the Coulomb potential ratio.

#### Signature

```python
class CoagulationStrategy(ABC): ...
```

### CoagulationStrategy().coulomb_potential_ratio

[Show source in strategy.py:205](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L205)

Calculate the Coulomb potential ratio based on the particle properties
and temperature.

#### Arguments

- `particle` - The particles for which the Coulomb
    potential ratio is to be calculated.
- `temperature` - The temperature of the gas phase [K].

#### Returns

The Coulomb potential ratio for the particle
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

[Show source in strategy.py:169](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L169)

Calculate the diffusive Knudsen number based on the particle
properties, temperature, and pressure.

#### Arguments

- `particle` - The particle for which the diffusive
    Knudsen number is to be calculated.
- `temperature` - The temperature of the gas phase [K].
- `pressure` - The pressure of the gas phase [Pa].

#### Returns

- `NDArray[np.float64]` - The diffusive Knudsen number for the particle
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

[Show source in strategy.py:40](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L40)

Calculate the dimensionless coagulation kernel based on the particle
properties interactions,
diffusive Knudsen number and Coulomb potential

#### Arguments

- [CoagulationStrategy().diffusive_knudsen](#coagulationstrategydiffusive_knudsen) - The diffusive Knudsen number
    for the particle [dimensionless].
- [CoagulationStrategy().coulomb_potential_ratio](#coagulationstrategycoulomb_potential_ratio) - The Coulomb potential
    ratio for the particle [dimensionless].

#### Returns

The dimensionless coagulation kernel for the particle
    [dimensionless].

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

[Show source in strategy.py:227](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L227)

Calculate the friction factor based on the particle properties,
temperature, and pressure.

#### Arguments

- `particle` - The particle for which the friction factor
    is to be calculated.
- `temperature` - The temperature of the gas phase [K].
- `pressure` - The pressure of the gas phase [Pa].

#### Returns

The friction factor for the particle [dimensionless].

#### Signature

```python
def friction_factor(
    self, particle: ParticleRepresentation, temperature: float, pressure: float
) -> NDArray[np.float64]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### CoagulationStrategy().gain_rate

[Show source in strategy.py:102](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L102)

Calculate the coagulation gain rate based on the particle radius,
distribution, and the coagulation kernel.

#### Arguments

- `particle` - The particle for which the coagulation
    gain rate is to be calculated.
- [CoagulationStrategy().kernel](#coagulationstrategykernel) - The coagulation kernel.

#### Returns

The coagulation gain rate for the particle [kg/s].

#### Notes

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

[Show source in strategy.py:62](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L62)

Calculate the coagulation kernel based on the particle properties,
temperature, and pressure.

#### Arguments

- `particle` - The particle for which the coagulation
    kernel is to be calculated.
- `temperature` - The temperature of the gas phase [K].
- `pressure` - The pressure of the gas phase [Pa].

#### Returns

The coagulation kernel for the particle [m^3/s].

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

[Show source in strategy.py:83](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L83)

Calculate the coagulation loss rate based on the particle radius,
distribution, and the coagulation kernel.

#### Arguments

- `particle` - The particle for which the coagulation
    loss rate is to be calculated.
- [CoagulationStrategy().kernel](#coagulationstrategykernel) - The coagulation kernel.

#### Returns

The coagulation loss rate for the particle [kg/s].

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

[Show source in strategy.py:125](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L125)

Calculate the net coagulation rate based on the particle radius,
distribution, and the coagulation kernel.

#### Arguments

- `particle` - The particle class for which the
    coagulation net rate is to be calculated.
- `temperature` - The temperature of the gas phase [K].
- `pressure` - The pressure of the gas phase [Pa].

#### Returns

- `Union[float,` *NDArray[np.float64]]* - The net coagulation rate for the
    particle [kg/s].

#### Signature

```python
@abstractmethod
def net_rate(
    self, particle: ParticleRepresentation, temperature: float, pressure: float
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### CoagulationStrategy().step

[Show source in strategy.py:147](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L147)

Perform a single step of the coagulation process.

#### Arguments

- `particle` - The particle for which the coagulation step
    is to be performed.
- `temperature` - The temperature of the gas phase [K].
- `pressure` - The pressure of the gas phase [Pa].
- `time_step` - The time step for the coagulation process [s].

#### Returns

- `ParticleRepresentation` - The particle after the coagulation step.

#### Signature

```python
@abstractmethod
def step(
    self,
    particle: ParticleRepresentation,
    temperature: float,
    pressure: float,
    time_step: float,
) -> ParticleRepresentation: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)



## ContinuousGeneralPDF

[Show source in strategy.py:480](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L480)

Continuous PDF coagulation strategy class. This class implements the
methods defined in the CoagulationStrategy abstract class. The kernel
strategy is passed as an argument to the class, should use a dimensionless
kernel representation.

#### Methods

- `kernel` - Calculate the coagulation kernel.
- `loss_rate` - Calculate the coagulation loss rate.
- `gain_rate` - Calculate the coagulation gain rate.
- `net_rate` - Calculate the net coagulation rate.

#### Signature

```python
class ContinuousGeneralPDF(CoagulationStrategy):
    def __init__(self, kernel_strategy: KernelStrategy): ...
```

#### See also

- [CoagulationStrategy](#coagulationstrategy)
- [KernelStrategy](./kernel.md#kernelstrategy)

### ContinuousGeneralPDF().dimensionless_kernel

[Show source in strategy.py:497](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L497)

#### Signature

```python
def dimensionless_kernel(
    self,
    diffusive_knudsen: NDArray[np.float64],
    coulomb_potential_ratio: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```

### ContinuousGeneralPDF().gain_rate

[Show source in strategy.py:554](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L554)

#### Signature

```python
def gain_rate(
    self, particle: ParticleRepresentation, kernel: NDArray[np.float64]
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### ContinuousGeneralPDF().kernel

[Show source in strategy.py:507](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L507)

#### Signature

```python
def kernel(
    self, particle: ParticleRepresentation, temperature: float, pressure: float
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### ContinuousGeneralPDF().loss_rate

[Show source in strategy.py:542](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L542)

#### Signature

```python
def loss_rate(
    self, particle: ParticleRepresentation, kernel: NDArray[np.float64]
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### ContinuousGeneralPDF().net_rate

[Show source in strategy.py:566](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L566)

#### Signature

```python
def net_rate(
    self, particle: ParticleRepresentation, temperature: float, pressure: float
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### ContinuousGeneralPDF().step

[Show source in strategy.py:580](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L580)

#### Signature

```python
def step(
    self,
    particle: ParticleRepresentation,
    temperature: float,
    pressure: float,
    time_step: float,
) -> ParticleRepresentation: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)



## DiscreteGeneral

[Show source in strategy.py:360](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L360)

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

[Show source in strategy.py:383](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L383)

#### Signature

```python
def dimensionless_kernel(
    self,
    diffusive_knudsen: NDArray[np.float64],
    coulomb_potential_ratio: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```

### DiscreteGeneral().gain_rate

[Show source in strategy.py:439](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L439)

#### Signature

```python
def gain_rate(
    self, particle: ParticleRepresentation, kernel: NDArray[np.float64]
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### DiscreteGeneral().kernel

[Show source in strategy.py:393](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L393)

#### Signature

```python
def kernel(
    self, particle: ParticleRepresentation, temperature: float, pressure: float
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### DiscreteGeneral().loss_rate

[Show source in strategy.py:428](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L428)

#### Signature

```python
def loss_rate(
    self, particle: ParticleRepresentation, kernel: NDArray[np.float64]
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### DiscreteGeneral().net_rate

[Show source in strategy.py:451](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L451)

#### Signature

```python
def net_rate(
    self, particle: ParticleRepresentation, temperature: float, pressure: float
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### DiscreteGeneral().step

[Show source in strategy.py:465](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L465)

#### Signature

```python
def step(
    self,
    particle: ParticleRepresentation,
    temperature: float,
    pressure: float,
    time_step: float,
) -> ParticleRepresentation: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)



## DiscreteSimple

[Show source in strategy.py:269](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L269)

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

[Show source in strategy.py:282](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L282)

#### Signature

```python
def dimensionless_kernel(
    self,
    diffusive_knudsen: NDArray[np.float64],
    coulomb_potential_ratio: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```

### DiscreteSimple().gain_rate

[Show source in strategy.py:319](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L319)

#### Signature

```python
def gain_rate(
    self, particle: ParticleRepresentation, kernel: NDArray[np.float64]
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### DiscreteSimple().kernel

[Show source in strategy.py:294](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L294)

#### Signature

```python
def kernel(
    self, particle: ParticleRepresentation, temperature: float, pressure: float
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### DiscreteSimple().loss_rate

[Show source in strategy.py:308](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L308)

#### Signature

```python
def loss_rate(
    self, particle: ParticleRepresentation, kernel: NDArray[np.float64]
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### DiscreteSimple().net_rate

[Show source in strategy.py:331](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L331)

#### Signature

```python
def net_rate(
    self, particle: ParticleRepresentation, temperature: float, pressure: float
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### DiscreteSimple().step

[Show source in strategy.py:345](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L345)

#### Signature

```python
def step(
    self,
    particle: ParticleRepresentation,
    temperature: float,
    pressure: float,
    time_step: float,
) -> ParticleRepresentation: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)



## ParticleResolved

[Show source in strategy.py:595](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L595)

Particle-resolved coagulation strategy class. This class implements the
methods defined in the CoagulationStrategy abstract class. The kernel
strategy is passed as an argument to the class, should use a dimensionless
kernel representation.

#### Methods

- `kernel` - Calculate the coagulation kernel.
- `loss_rate` - Not implemented.
- `gain_rate` - Not implemented.
- `net_rate` - Not implemented.
- `step` - Perform a single step of the coagulation process.

#### Signature

```python
class ParticleResolved(CoagulationStrategy):
    def __init__(
        self,
        kernel_strategy: KernelStrategy,
        kernel_radius: Optional[NDArray[np.float64]] = None,
        kernel_bins_number: int = 100,
    ): ...
```

#### See also

- [CoagulationStrategy](#coagulationstrategy)
- [KernelStrategy](./kernel.md#kernelstrategy)

### ParticleResolved().dimensionless_kernel

[Show source in strategy.py:648](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L648)

#### Signature

```python
def dimensionless_kernel(
    self,
    diffusive_knudsen: NDArray[np.float64],
    coulomb_potential_ratio: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```

### ParticleResolved().gain_rate

[Show source in strategy.py:688](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L688)

#### Signature

```python
def gain_rate(
    self, particle: ParticleRepresentation, kernel: NDArray[np.float64]
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### ParticleResolved().get_kernel_radius

[Show source in strategy.py:620](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L620)

Get the binning for the kernel radius.

If the kernel radius is not set, it will be calculated based on the
particle radius.

#### Arguments

- `particle` - The particle for which the kernel radius is to be
    calculated.

#### Returns

The kernel radius for the particle [m].

#### Signature

```python
def get_kernel_radius(
    self, particle: ParticleRepresentation
) -> Optional[NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### ParticleResolved().kernel

[Show source in strategy.py:661](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L661)

#### Signature

```python
def kernel(
    self, particle: ParticleRepresentation, temperature: float, pressure: float
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### ParticleResolved().loss_rate

[Show source in strategy.py:678](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L678)

#### Signature

```python
def loss_rate(
    self, particle: ParticleRepresentation, kernel: NDArray[np.float64]
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### ParticleResolved().net_rate

[Show source in strategy.py:698](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L698)

#### Signature

```python
def net_rate(
    self, particle: ParticleRepresentation, temperature: float, pressure: float
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)
