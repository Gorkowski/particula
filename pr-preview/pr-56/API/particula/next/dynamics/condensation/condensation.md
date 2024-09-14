# Condensation

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Dynamics](./index.md#dynamics) / Condensation

> Auto-generated documentation for [particula.next.dynamics.condensation](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/condensation.py) module.

## CondensationIsothermal

[Show source in condensation.py:508](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/condensation.py#L508)

Condensation strategy for isothermal conditions.

Condensation strategy for isothermal conditions, where the temperature
remains constant. This class implements the mass transfer rate calculation
for condensation of particles based on partial pressures. No Latent heat
of vaporization effect is considered.

#### Signature

```python
class CondensationIsothermal(CondensationStrategy):
    def __init__(
        self,
        molar_mass: Union[float, NDArray[np.float64]],
        diffusion_coefficient: Union[float, NDArray[np.float64]] = 2 * 1e-09,
        accommodation_coefficient: Union[float, NDArray[np.float64]] = 1.0,
        update_gases: bool = True,
    ): ...
```

#### See also

- [CondensationStrategy](#condensationstrategy)

### CondensationIsothermal().mass_transfer_rate

[Show source in condensation.py:531](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/condensation.py#L531)

#### Signature

```python
def mass_transfer_rate(
    self,
    particle: ParticleRepresentation,
    gas_species: GasSpecies,
    temperature: float,
    pressure: float,
    dynamic_viscosity: Optional[float] = None,
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [GasSpecies](../gas/species.md#gasspecies)
- [ParticleRepresentation](../particles/representation.md#particlerepresentation)

### CondensationIsothermal().rate

[Show source in condensation.py:577](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/condensation.py#L577)

#### Signature

```python
def rate(
    self,
    particle: ParticleRepresentation,
    gas_species: GasSpecies,
    temperature: float,
    pressure: float,
) -> NDArray[np.float64]: ...
```

#### See also

- [GasSpecies](../gas/species.md#gasspecies)
- [ParticleRepresentation](../particles/representation.md#particlerepresentation)

### CondensationIsothermal().step

[Show source in condensation.py:606](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/condensation.py#L606)

#### Signature

```python
def step(
    self,
    particle: ParticleRepresentation,
    gas_species: GasSpecies,
    temperature: float,
    pressure: float,
    time_step: float,
) -> Tuple[ParticleRepresentation, GasSpecies]: ...
```

#### See also

- [GasSpecies](../gas/species.md#gasspecies)
- [ParticleRepresentation](../particles/representation.md#particlerepresentation)



## CondensationStrategy

[Show source in condensation.py:278](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/condensation.py#L278)

Condensation strategy abstract class.

Abstract class for mass transfer strategies, for condensation or
evaporation of particles. This class should be subclassed to implement
specific mass transfer strategies.

#### Arguments

- `molar_mass` - The molar mass of the species [kg/mol]. If a single value
is provided, it will be used for all species.
- `diffusion_coefficient` - The diffusion coefficient of the species
[m^2/s]. If a single value is provided, it will be used for all
species. Default is 2*1e-9 m^2/s for air.
- `accommodation_coefficient` - The mass accommodation coefficient of the
species. If a single value is provided, it will be used for all
species. Default is 1.0.

#### Signature

```python
class CondensationStrategy(ABC):
    def __init__(
        self,
        molar_mass: Union[float, NDArray[np.float64]],
        diffusion_coefficient: Union[float, NDArray[np.float64]] = 2 * 1e-09,
        accommodation_coefficient: Union[float, NDArray[np.float64]] = 1.0,
        update_gases: bool = True,
    ): ...
```

### CondensationStrategy().first_order_mass_transport

[Show source in condensation.py:374](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/condensation.py#L374)

First-order mass transport coefficient per particle.

Calculate the first-order mass transport coefficient, K, for a given
particle based on the diffusion coefficient, radius, and vapor
transition correction factor.

#### Arguments

- `radius` - The radius of the particle [m].
- `temperature` - The temperature at which the first-order mass
transport coefficient is to be calculated.
- `pressure` - The pressure of the gas phase.
- `dynamic_viscosity` - The dynamic viscosity of the gas [Pa*s]. If not
provided, it will be calculated based on the temperature

#### Returns

- `Union[float,` *NDArray[np.float64]]* - The first-order mass transport
coefficient per particle (m^3/s).

#### References

Aerosol Modeling, Chapter 2, Equation 2.49 (excluding particle
number)

#### Signature

```python
def first_order_mass_transport(
    self,
    radius: Union[float, NDArray[np.float64]],
    temperature: float,
    pressure: float,
    dynamic_viscosity: Optional[float] = None,
) -> Union[float, NDArray[np.float64]]: ...
```

### CondensationStrategy().knudsen_number

[Show source in condensation.py:339](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/condensation.py#L339)

The Knudsen number for a particle.

Calculate the Knudsen number based on the mean free path of the gas
molecules and the radius of the particle.

#### Arguments

- `radius` - The radius of the particle [m].
- `temperature` - The temperature of the gas [K].
- `pressure` - The pressure of the gas [Pa].
- `dynamic_viscosity` - The dynamic viscosity of the gas [Pa*s]. If
not provided, it will be calculated based on the temperature

#### Returns

- `Union[float,` *NDArray[np.float64]]* - The Knudsen number, which is the
ratio of the mean free path to the particle radius.

#### References

[Knudsen Number](https://en.wikipedia.org/wiki/Knudsen_number)

#### Signature

```python
def knudsen_number(
    self,
    radius: Union[float, NDArray[np.float64]],
    temperature: float,
    pressure: float,
    dynamic_viscosity: Optional[float] = None,
) -> Union[float, NDArray[np.float64]]: ...
```

### CondensationStrategy().mass_transfer_rate

[Show source in condensation.py:418](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/condensation.py#L418)

Mass transfer rate for a particle.

Calculate the mass transfer rate based on the difference in partial
pressure and the first-order mass transport coefficient.

#### Arguments

- `particle` - The particle for which the mass transfer rate is to be
calculated.
- `gas_species` - The gas species with which the particle is in contact.
- `temperature` - The temperature at which the mass transfer rate
is to be calculated.
- `pressure` - The pressure of the gas phase.
- `dynamic_viscosity` - The dynamic viscosity of the gas [Pa*s]. If not
provided, it will be calculated based on the temperature

#### Returns

- `Union[float,` *NDArray[np.float64]]* - The mass transfer rate for the
particle [kg/s].

#### Signature

```python
@abstractmethod
def mass_transfer_rate(
    self,
    particle: ParticleRepresentation,
    gas_species: GasSpecies,
    temperature: float,
    pressure: float,
    dynamic_viscosity: Optional[float] = None,
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [GasSpecies](../gas/species.md#gasspecies)
- [ParticleRepresentation](../particles/representation.md#particlerepresentation)

### CondensationStrategy().mean_free_path

[Show source in condensation.py:308](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/condensation.py#L308)

Calculate the mean free path of the gas molecules based on the
temperature, pressure, and dynamic viscosity of the gas.

#### Arguments

- `temperature` - The temperature of the gas [K].
- `pressure` - The pressure of the gas [Pa].
- `dynamic_viscosity` - The dynamic viscosity of the gas [Pa*s]. If not
provided, it will be calculated based on the temperature

#### Returns

- `Union[float,` *NDArray[np.float64]]* - The mean free path of the gas
molecules in meters (m).

#### References

Mean Free Path:
[Wikipedia](https://en.wikipedia.org/wiki/Mean_free_path)

#### Signature

```python
def mean_free_path(
    self, temperature: float, pressure: float, dynamic_viscosity: Optional[float] = None
) -> Union[float, NDArray[np.float64]]: ...
```

### CondensationStrategy().rate

[Show source in condensation.py:448](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/condensation.py#L448)

Calculate the rate of mass condensation for each particle due to
each condensable gas species.

The rate of condensation is determined based on the mass transfer rate,
which is a function of particle properties, gas species properties,
temperature, and pressure. This rate is then scaled by the
concentration of particles in the system to get the overall
condensation rate for each particle or bin.

#### Arguments

- `particle` *ParticleRepresentation* - Representation of the particles,
    including properties such as size, concentration, and mass.
- `gas_species` *GasSpecies* - The species of gas condensing onto the
    particles.
- `temperature` *float* - The temperature of the system in Kelvin.
- `pressure` *float* - The pressure of the system in Pascals.

#### Returns

An array of condensation rates for each particle,
scaled by
particle concentration.

#### Signature

```python
@abstractmethod
def rate(
    self,
    particle: ParticleRepresentation,
    gas_species: GasSpecies,
    temperature: float,
    pressure: float,
) -> NDArray[np.float64]: ...
```

#### See also

- [GasSpecies](../gas/species.md#gasspecies)
- [ParticleRepresentation](../particles/representation.md#particlerepresentation)

### CondensationStrategy().step

[Show source in condensation.py:481](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/condensation.py#L481)

Execute the condensation process for a given time step.

#### Arguments

- `particle` *ParticleRepresentation* - The particle to modify.
- `gas_species` *GasSpecies* - The gas species to condense onto the
    particle.
- `temperature` *float* - The temperature of the system in Kelvin.
- `pressure` *float* - The pressure of the system in Pascals.
- `time_step` *float* - The time step for the process in seconds.

#### Returns

- `ParticleRepresentation` - The modified particle instance.
- `GasSpecies` - The modified gas species instance.

#### Signature

```python
@abstractmethod
def step(
    self,
    particle: ParticleRepresentation,
    gas_species: GasSpecies,
    temperature: float,
    pressure: float,
    time_step: float,
) -> Tuple[ParticleRepresentation, GasSpecies]: ...
```

#### See also

- [GasSpecies](../gas/species.md#gasspecies)
- [ParticleRepresentation](../particles/representation.md#particlerepresentation)



## calculate_mass_transfer

[Show source in condensation.py:129](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/condensation.py#L129)

Helper function that routes the mass transfer calculation to either the
single-species or multi-species calculation functions based on the input
dimensions of gas_mass.

#### Arguments

- `mass_rate` - The rate of mass transfer per particle (kg/s).
- `time_step` - The time step for the mass transfer calculation (seconds).
- `gas_mass` - The available mass of gas species (kg).
- `particle_mass` - The mass of each particle (kg).
- `particle_concentration` - The concentration of particles (number/m^3).

#### Returns

The amount of mass transferred, accounting for gas and particle
    limitations.

#### Signature

```python
def calculate_mass_transfer(
    mass_rate: NDArray[np.float64],
    time_step: float,
    gas_mass: NDArray[np.float64],
    particle_mass: NDArray[np.float64],
    particle_concentration: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```



## calculate_mass_transfer_multiple_species

[Show source in condensation.py:214](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/condensation.py#L214)

Calculate mass transfer for multiple gas species.

#### Arguments

- `mass_rate` - The rate of mass transfer per particle for each gas species
    (kg/s).
- `time_step` - The time step for the mass transfer calculation (seconds).
- `gas_mass` - The available mass of each gas species (kg).
- `particle_mass` - The mass of each particle for each gas species (kg).
- `particle_concentration` - The concentration of particles for each gas
    species (number/m^3).

#### Returns

The amount of mass transferred for multiple gas species.

#### Signature

```python
def calculate_mass_transfer_multiple_species(
    mass_rate: NDArray[np.float64],
    time_step: float,
    gas_mass: NDArray[np.float64],
    particle_mass: NDArray[np.float64],
    particle_concentration: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```



## calculate_mass_transfer_single_species

[Show source in condensation.py:170](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/condensation.py#L170)

Calculate mass transfer for a single gas species (m=1).

#### Arguments

- `mass_rate` - The rate of mass transfer per particle (number*kg/s).
- `time_step` - The time step for the mass transfer calculation (seconds).
- `gas_mass` - The available mass of gas species (kg).
- `particle_mass` - The mass of each particle (kg).
- `particle_concentration` - The concentration of particles (number/m^3).

#### Returns

The amount of mass transferred for a single gas species.

#### Signature

```python
def calculate_mass_transfer_single_species(
    mass_rate: NDArray[np.float64],
    time_step: float,
    gas_mass: NDArray[np.float64],
    particle_mass: NDArray[np.float64],
    particle_concentration: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```



## first_order_mass_transport_k

[Show source in condensation.py:55](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/condensation.py#L55)

First-order mass transport coefficient per particle.

Calculate the first-order mass transport coefficient, K, for a given radius
diffusion coefficient, and vapor transition correction factor. For a
single particle.

#### Arguments

- `radius` - The radius of the particle [m].
- `diffusion_coefficient` - The diffusion coefficient of the vapor [m^2/s],
default to air.
- `vapor_transition` - The vapor transition correction factor. [unitless]

#### Returns

- `Union[float,` *NDArray[np.float64]]* - The first-order mass transport
coefficient per particle (m^3/s).

#### References

- Aerosol Modeling: Chapter 2, Equation 2.49 (excluding number)
- Mass Diffusivity:
    [Wikipedia](https://en.wikipedia.org/wiki/Mass_diffusivity)

#### Signature

```python
def first_order_mass_transport_k(
    radius: Union[float, NDArray[np.float64]],
    vapor_transition: Union[float, NDArray[np.float64]],
    diffusion_coefficient: Union[float, NDArray[np.float64]] = 2 * 1e-09,
) -> Union[float, NDArray[np.float64]]: ...
```



## mass_transfer_rate

[Show source in condensation.py:93](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/condensation.py#L93)

Calculate the mass transfer rate for a particle.

Calculate the mass transfer rate based on the difference in partial
pressure and the first-order mass transport coefficient.

#### Arguments

- `pressure_delta` - The difference in partial pressure between the gas
phase and the particle phase.
- `first_order_mass_transport` - The first-order mass transport coefficient
per particle.
- `temperature` - The temperature at which the mass transfer rate is to be
calculated.

#### Returns

- `Union[float,` *NDArray[np.float64]]* - The mass transfer rate for the
particle [kg/s].

#### References

- Aerosol Modeling Chapter 2, Equation 2.41 (excluding particle number)
- Seinfeld and Pandis: "Atmospheric Chemistry and Physics",
    Equation 13.3

#### Signature

```python
def mass_transfer_rate(
    pressure_delta: Union[float, NDArray[np.float64]],
    first_order_mass_transport: Union[float, NDArray[np.float64]],
    temperature: Union[float, NDArray[np.float64]],
    molar_mass: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```
