# Species

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Gas](./index.md#gas) / Species

> Auto-generated documentation for [particula.next.gas.species](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/species.py) module.

## GasSpecies

[Show source in species.py:34](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/species.py#L34)

Represents a single species of gas, including its properties such as
name, mass, vapor pressure, and whether it is condensable.

#### Attributes

- name (str): The name of the gas species.
- mass (float): The mass of the gas species.
- vapor_pressure (Optional[float]): The vapor pressure of the gas
    species. None if not applicable.
- condensable (bool): Indicates whether the gas species is condensable.
    Default is True.
- concentration (float): The concentration of the gas species in the
    mixture. Default is 0.0 kg/m^3.

#### Methods

--------
- `-` *get_molar_mass* - Get the molar mass of the gas species.
- `-` *get_condensable* - Check if the gas species is condensable.
- `-` *get_concentration* - Get the concentration of the gas species in the
    mixture.
- `-` *get_pure_vapor_pressure* - Calculate the pure vapor pressure of the gas
    species at a given temperature.
- `-` *get_partial_pressure* - Calculate the partial pressure of the gas species.
- `-` *get_saturation_ratio* - Calculate the saturation ratio of the gas species.
- `-` *get_saturation_concentration* - Calculate the saturation concentration of
    the gas species.
- `-` *add_concentration* - Add concentration to the gas species.

#### Signature

```python
class GasSpecies:
    def __init__(
        self,
        name: Union[str, NDArray[np.str_]],
        molar_mass: Union[float, NDArray[np.float_]],
        vapor_pressure_strategy: Union[
            VaporPressureStrategy, list[VaporPressureStrategy]
        ] = ConstantVaporPressureStrategy(0.0),
        condensable: Union[bool, NDArray[np.bool_]] = True,
        concentration: Union[float, NDArray[np.float_]] = 0.0,
    ) -> None: ...
```

#### See also

- [VaporPressureStrategy](./vapor_pressure_strategies.md#vaporpressurestrategy)

### GasSpecies().__len__

[Show source in species.py:84](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/species.py#L84)

Return the number of gas species.

#### Signature

```python
def __len__(self): ...
```

### GasSpecies().__str__

[Show source in species.py:80](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/species.py#L80)

Return a string representation of the GasSpecies object.

#### Signature

```python
def __str__(self): ...
```

### GasSpecies().add_concentration

[Show source in species.py:282](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/species.py#L282)

Add concentration to the gas species.

#### Arguments

- added_concentration (float): The concentration to add to the gas
    species.

#### Signature

```python
def add_concentration(self, added_concentration: Union[float, NDArray[np.float_]]): ...
```

### GasSpecies().get_concentration

[Show source in species.py:116](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/species.py#L116)

Get the concentration of the gas species in the mixture, in kg/m^3.

#### Returns

- concentration (float or NDArray[np.float_]): The concentration of the
    gas species in the mixture.

#### Signature

```python
def get_concentration(self) -> Union[float, NDArray[np.float_]]: ...
```

### GasSpecies().get_condensable

[Show source in species.py:108](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/species.py#L108)

Check if the gas species is condensable or not.

#### Returns

- condensable (bool): True if the gas species is condensable, False
    otherwise.

#### Signature

```python
def get_condensable(self) -> Union[bool, NDArray[np.bool_]]: ...
```

### GasSpecies().get_molar_mass

[Show source in species.py:100](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/species.py#L100)

Get the molar mass of the gas species in kg/mol.

#### Returns

- molar_mass (float or NDArray[np.float_]): The molar mass of the gas
    species, in kg/mol.

#### Signature

```python
def get_molar_mass(self) -> Union[float, NDArray[np.float_]]: ...
```

### GasSpecies().get_name

[Show source in species.py:92](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/species.py#L92)

Get the name of the gas species.

#### Returns

- `np.float64` - The mass of the gas species.

#### Signature

```python
def get_name(self) -> Union[str, NDArray[np.str_]]: ...
```

### GasSpecies().get_partial_pressure

[Show source in species.py:157](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/species.py#L157)

Calculate the partial pressure of the gas based on the vapor
pressure strategy. This method accounts for multiple strategies if
assigned and calculates partial pressure for each strategy based on
the corresponding concentration and molar mass.

#### Arguments

- temperature (float or NDArray[np.float_]): The temperature in
Kelvin at which to calculate the partial pressure.

#### Returns

- partial_pressure (float or NDArray[np.float_]): Partial pressure
of the gas in Pascals.

#### Raises

- `-` *ValueError* - If the vapor pressure strategy is not set.

#### Signature

```python
def get_partial_pressure(
    self, temperature: Union[float, NDArray[np.float_]]
) -> Union[float, NDArray[np.float_]]: ...
```

### GasSpecies().get_pure_vapor_pressure

[Show source in species.py:124](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/species.py#L124)

Calculate the pure vapor pressure of the gas species at a given
temperature in Kelvin.

This method supports both a single strategy or a list of strategies
for calculating vapor pressure.

#### Arguments

- temperature (float or NDArray[np.float_]): The temperature in
Kelvin at which to calculate vapor pressure.

#### Returns

- vapor_pressure (float or NDArray[np.float_]): The calculated pure
vapor pressure in Pascals.

#### Raises

- `ValueError` - If no vapor pressure strategy is set.

#### Signature

```python
def get_pure_vapor_pressure(
    self, temperature: Union[float, NDArray[np.float_]]
) -> Union[float, NDArray[np.float_]]: ...
```

### GasSpecies().get_saturation_concentration

[Show source in species.py:242](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/species.py#L242)

Returns the mass of a specific condensable species or the masses of
all condensable species in the gas mixture as an np.ndarray. If a name
is provided, only the mass of that specific condensable species is
returned.

#### Arguments

- temperature (float or NDArray[np.float_]): The temperature in
Kelvin at which to calculate the partial pressure.

#### Returns

- saturation_concentration (float or NDArray[np.float_]): The
saturation concentration of the gas

#### Raises

- `ValueError` - If a specific species name is provided but not found
in the mixture, or if it's not condensable.

#### Signature

```python
def get_saturation_concentration(
    self, temperature: Union[float, NDArray[np.float_]]
) -> Union[float, NDArray[np.float_]]: ...
```

### GasSpecies().get_saturation_ratio

[Show source in species.py:200](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/species.py#L200)

Calculate the saturation ratio of the gas based on the vapor
pressure strategy. This method accounts for multiple strategies if
assigned and calculates saturation ratio for each strategy based on
the corresponding concentration and molar mass.

#### Arguments

- temperature (float or NDArray[np.float_]): The temperature in
Kelvin at which to calculate the partial pressure.

#### Returns

- saturation_ratio (float or NDArray[np.float_]): The saturation ratio
of the gas

#### Raises

- `ValueError` - If the specified name is not found in the mixture.

#### Signature

```python
def get_saturation_ratio(
    self, temperature: Union[float, NDArray[np.float_]]
) -> Union[float, NDArray[np.float_]]: ...
```
