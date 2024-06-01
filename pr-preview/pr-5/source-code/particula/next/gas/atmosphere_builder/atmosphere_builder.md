# AtmosphereBuilder

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Gas](./index.md#gas) / AtmosphereBuilder

> Auto-generated documentation for [particula.next.gas.atmosphere_builder](../../../../../particula/next/gas/atmosphere_builder.py) module.

## AtmosphereBuilder

[Show source in atmosphere_builder.py:13](../../../../../particula/next/gas/atmosphere_builder.py#L13)

A builder class for creating Atmosphere objects with a fluent interface.

#### Attributes

----------
- temperature (float): The temperature of the gas mixture in Kelvin.
- total_pressure (float): The total pressure of the gas mixture in Pascals.
- species (list[GasSpecies]): The list of gas species in the mixture.

#### Methods

-------
- `-` *set_temperature(temperature)* - Set the temperature of the gas mixture.
- `-` *set_total_pressure(total_pressure)* - Set the total pressure of the gas
mixture.
- `-` *add_species(species)* - Add a GasSpecies component to the gas mixture.
- `-` *set_parameters(parameters)* - Set the parameters from a dictionary.
- `-` *build()* - Validate and return the Atmosphere object.

#### Signature

```python
class AtmosphereBuilder(BuilderABC):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)

### AtmosphereBuilder().add_species

[Show source in atmosphere_builder.py:99](../../../../../particula/next/gas/atmosphere_builder.py#L99)

Add a GasSpecies component to the gas mixture.

#### Arguments

----
- species (GasSpecies): The GasSpecies object to be added to the
mixture.

#### Returns

-------
- `-` *self* - The AtmosphereBuilder object.

#### Signature

```python
def add_species(self, species: GasSpecies): ...
```

#### See also

- [GasSpecies](./species.md#gasspecies)

### AtmosphereBuilder().build

[Show source in atmosphere_builder.py:114](../../../../../particula/next/gas/atmosphere_builder.py#L114)

Validate and return the Atmosphere object.

#### Returns

-------
- `-` *Atmosphere* - The Atmosphere object.

#### Signature

```python
def build(self) -> Atmosphere: ...
```

#### See also

- [Atmosphere](./atmosphere.md#atmosphere)

### AtmosphereBuilder().set_temperature

[Show source in atmosphere_builder.py:39](../../../../../particula/next/gas/atmosphere_builder.py#L39)

Set the temperature of the gas mixture, in Kelvin.

#### Arguments

----
- temperature (float): The temperature of the gas mixture.
- temperature_units (str): The units of the temperature.
options are 'degC', 'degF', 'degR', 'K'. Default is 'K'.

#### Returns

-------
- `-` *self* - The AtmosphereBuilder object.

#### Raises

------
- `-` *ValueError* - If the temperature is below absolute zero.

#### Signature

```python
def set_temperature(self, temperature: float, temperature_units: str = "K"): ...
```

### AtmosphereBuilder().set_total_pressure

[Show source in atmosphere_builder.py:71](../../../../../particula/next/gas/atmosphere_builder.py#L71)

Set the total pressure of the gas mixture, in Pascals.

#### Arguments

----
- total_pressure (float): The total pressure of the gas mixture.
- pressure_units (str): The units of the pressure. Options are 'Pa',
'kPa', 'MPa', 'psi', 'bar', 'atm'. Default is 'Pa'.

#### Returns

-------
- `-` *self* - The AtmosphereBuilder object.

#### Raises

------
- `-` *ValueError* - If the total pressure is below zero.

#### Signature

```python
def set_total_pressure(self, total_pressure: float, pressure_units: str = "Pa"): ...
```
