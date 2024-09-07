# Species Builders

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Gas](./index.md#gas) / Species Builders

> Auto-generated documentation for [particula.next.gas.species_builders](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/species_builders.py) module.

## GasSpeciesBuilder

[Show source in species_builders.py:24](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/species_builders.py#L24)

Builder class for GasSpecies objects, allowing for a more fluent and
readable creation of GasSpecies instances with optional parameters.

#### Attributes

- `name` - The name of the gas species.
- `molar_mass` - The molar mass of the gas species in kg/mol.
- `vapor_pressure_strategy` - The vapor pressure strategy for the
    gas species.
- `condensable` - Whether the gas species is condensable.
- `concentration` - The concentration of the gas species in the
    mixture, in kg/m^3.

#### Methods

- `set_name` - Set the name of the gas species.
- `set_molar_mass` - Set the molar mass of the gas species in kg/mol.
- `set_vapor_pressure_strategy` - Set the vapor pressure strategy
    for the gas species.
- `set_condensable` - Set the condensable bool of the gas species.
- `set_concentration` - Set the concentration of the gas species in the
    mixture, in kg/m^3.
- `set_parameters` - Set the parameters of the GasSpecies object from
    a dictionary including optional units.

#### Raises

- `ValueError` - If any required key is missing. During check_keys and
    pre_build_check. Or if trying to set an invalid parameter.
- `Warning` - If using default units for any parameter.

#### Signature

```python
class GasSpeciesBuilder(BuilderABC, BuilderMolarMassMixin, BuilderConcentrationMixin):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)
- [BuilderConcentrationMixin](../abc_builder.md#builderconcentrationmixin)
- [BuilderMolarMassMixin](../abc_builder.md#buildermolarmassmixin)

### GasSpeciesBuilder().build

[Show source in species_builders.py:92](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/species_builders.py#L92)

Validate and return the GasSpecies object.

#### Signature

```python
def build(self) -> GasSpecies: ...
```

#### See also

- [GasSpecies](./species.md#gasspecies)

### GasSpeciesBuilder().set_condensable

[Show source in species_builders.py:84](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/species_builders.py#L84)

Set the condensable bool of the gas species.

#### Signature

```python
def set_condensable(self, condensable: Union[bool, NDArray[np.bool_]]): ...
```

### GasSpeciesBuilder().set_name

[Show source in species_builders.py:71](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/species_builders.py#L71)

Set the name of the gas species.

#### Signature

```python
def set_name(self, name: Union[str, NDArray[np.str_]]): ...
```

### GasSpeciesBuilder().set_vapor_pressure_strategy

[Show source in species_builders.py:76](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/species_builders.py#L76)

Set the vapor pressure strategy for the gas species.

#### Signature

```python
def set_vapor_pressure_strategy(
    self, strategy: Union[VaporPressureStrategy, list[VaporPressureStrategy]]
): ...
```

#### See also

- [VaporPressureStrategy](./vapor_pressure_strategies.md#vaporpressurestrategy)



## PresetGasSpeciesBuilder

[Show source in species_builders.py:105](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/species_builders.py#L105)

Builder class for GasSpecies objects, allowing for a more fluent and
readable creation of GasSpecies instances with optional parameters.

#### Signature

```python
class PresetGasSpeciesBuilder(GasSpeciesBuilder):
    def __init__(self): ...
```

#### See also

- [GasSpeciesBuilder](#gasspeciesbuilder)

### PresetGasSpeciesBuilder().build

[Show source in species_builders.py:123](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/species_builders.py#L123)

#### Signature

```python
def build(self) -> GasSpecies: ...
```

#### See also

- [GasSpecies](./species.md#gasspecies)
