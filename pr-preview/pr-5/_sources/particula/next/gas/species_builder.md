# Species Builder

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Gas](./index.md#gas) / Species Builder

> Auto-generated documentation for [particula.next.gas.species_builder](../../../../particula/next/gas/species_builder.py) module.

- [Species Builder](#species-builder)
  - [GasSpeciesBuilder](#gasspeciesbuilder)
    - [GasSpeciesBuilder().build](#gasspeciesbuilder()build)
    - [GasSpeciesBuilder().set_concentration](#gasspeciesbuilder()set_concentration)
    - [GasSpeciesBuilder().set_condensable](#gasspeciesbuilder()set_condensable)
    - [GasSpeciesBuilder().set_molar_mass](#gasspeciesbuilder()set_molar_mass)
    - [GasSpeciesBuilder().set_name](#gasspeciesbuilder()set_name)
    - [GasSpeciesBuilder().set_vapor_pressure_strategy](#gasspeciesbuilder()set_vapor_pressure_strategy)

## GasSpeciesBuilder

[Show source in species_builder.py:19](../../../../particula/next/gas/species_builder.py#L19)

Builder class for GasSpecies objects, allowing for a more fluent and
readable creation of GasSpecies instances with optional parameters.

#### Attributes

----------
- name (str): The name of the gas species.
- molar_mass (float): The molar mass of the gas species in kg/mol.
- vapor_pressure_strategy (VaporPressureStrategy): The vapor pressure
    strategy for the gas species.
- condensable (bool): Whether the gas species is condensable.
- concentration (float): The concentration of the gas species in the
    mixture, in kg/m^3.

#### Methods

-------
- `-` *set_name(name)* - Set the name of the gas species.
- set_molar_mass(molar_mass, molar_mass_units): Set the molar mass of the
    gas species in kg/mol.
- `-` *set_vapor_pressure_strategy(strategy)* - Set the vapor pressure strategy
    for the gas species.
- `-` *set_condensable(condensable)* - Set the condensable bool of the gas
    species.
- set_concentration(concentration, concentration_units): Set the
    concentration of the gas species in the mixture, in kg/m^3.
- `-` *set_parameters(params)* - Set the parameters of the GasSpecies object from
    a dictionary including optional units.
- `-` *build()* - Validate and return the GasSpecies object.

#### Raises

------
- `-` *ValueError* - If any required key is missing. During check_keys and
    pre_build_check. Or if trying to set an invalid parameter.
- `-` *Warning* - If using default units for any parameter.

#### Signature

```python
class GasSpeciesBuilder(BuilderABC):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)

### GasSpeciesBuilder().build

[Show source in species_builder.py:114](../../../../particula/next/gas/species_builder.py#L114)

Validate and return the GasSpecies object.

#### Signature

```python
def build(self) -> GasSpecies: ...
```

#### See also

- [GasSpecies](./species.md#gasspecies)

### GasSpeciesBuilder().set_concentration

[Show source in species_builder.py:99](../../../../particula/next/gas/species_builder.py#L99)

Set the concentration of the gas species in the mixture,
in kg/m^3.

#### Signature

```python
def set_concentration(
    self,
    concentration: Union[float, NDArray[np.float_]],
    concentration_units: str = "kg/m^3",
): ...
```

### GasSpeciesBuilder().set_condensable

[Show source in species_builder.py:91](../../../../particula/next/gas/species_builder.py#L91)

Set the condensable bool of the gas species.

#### Signature

```python
def set_condensable(self, condensable: Union[bool, NDArray[np.bool_]]): ...
```

### GasSpeciesBuilder().set_molar_mass

[Show source in species_builder.py:70](../../../../particula/next/gas/species_builder.py#L70)

Set the molar mass of the gas species. Units in kg/mol.

#### Signature

```python
def set_molar_mass(
    self, molar_mass: Union[float, NDArray[np.float_]], molar_mass_units: str = "kg/mol"
): ...
```

### GasSpeciesBuilder().set_name

[Show source in species_builder.py:65](../../../../particula/next/gas/species_builder.py#L65)

Set the name of the gas species.

#### Signature

```python
def set_name(self, name: Union[str, NDArray[np.str_]]): ...
```

### GasSpeciesBuilder().set_vapor_pressure_strategy

[Show source in species_builder.py:83](../../../../particula/next/gas/species_builder.py#L83)

Set the vapor pressure strategy for the gas species.

#### Signature

```python
def set_vapor_pressure_strategy(
    self, strategy: Union[VaporPressureStrategy, list[VaporPressureStrategy]]
): ...
```

#### See also

- [VaporPressureStrategy](./vapor_pressure_strategies.md#vaporpressurestrategy)