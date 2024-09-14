# Vapor Pressure Strategies

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Gas](./index.md#gas) / Vapor Pressure Strategies

> Auto-generated documentation for [particula.next.gas.vapor_pressure_strategies](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py) module.

## AntoineVaporPressureStrategy

[Show source in vapor_pressure_strategies.py:164](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py#L164)

Concrete implementation of the VaporPressureStrategy using the
Antoine equation for vapor pressure calculations.

#### Signature

```python
class AntoineVaporPressureStrategy(VaporPressureStrategy):
    def __init__(
        self,
        a: Union[float, NDArray[np.float64]] = 0.0,
        b: Union[float, NDArray[np.float64]] = 0.0,
        c: Union[float, NDArray[np.float64]] = 0.0,
    ): ...
```

#### See also

- [VaporPressureStrategy](#vaporpressurestrategy)

### AntoineVaporPressureStrategy().pure_vapor_pressure

[Show source in vapor_pressure_strategies.py:180](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py#L180)

Calculate vapor pressure using the Antoine equation.

#### Arguments

a, b, c: Antoine equation parameters.
- `temperature` - Temperature in Kelvin.

#### Returns

Vapor pressure in Pascals.

#### References

- `-` *Equation* - log10(P) = a - b / (T - c)
- https://en.wikipedia.org/wiki/Antoine_equation (but in Kelvin)
- Kelvin form:
    https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781118135341.app1

#### Signature

```python
def pure_vapor_pressure(
    self, temperature: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]: ...
```



## ClausiusClapeyronStrategy

[Show source in vapor_pressure_strategies.py:204](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py#L204)

Concrete implementation of the VaporPressureStrategy using the
Clausius-Clapeyron equation for vapor pressure calculations.

#### Signature

```python
class ClausiusClapeyronStrategy(VaporPressureStrategy):
    def __init__(
        self,
        latent_heat: Union[float, NDArray[np.float64]],
        temperature_initial: Union[float, NDArray[np.float64]],
        pressure_initial: Union[float, NDArray[np.float64]],
    ): ...
```

#### See also

- [VaporPressureStrategy](#vaporpressurestrategy)

### ClausiusClapeyronStrategy().pure_vapor_pressure

[Show source in vapor_pressure_strategies.py:233](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py#L233)

Calculate vapor pressure using Clausius-Clapeyron equation.

#### Arguments

- `latent_heat` - Latent heat of vaporization in J/mol.
- `temperature_initial` - Initial temperature in Kelvin.
- `pressure_initial` - Initial vapor pressure in Pascals.
- `temperature` - Final temperature in Kelvin.
- `gas_constant` - gas constant (default is 8.314 J/(mol·K)).

#### Returns

Pure vapor pressure in Pascals.

#### References

- https://en.wikipedia.org/wiki/Clausius%E2%80%93Clapeyron_relation

#### Signature

```python
def pure_vapor_pressure(
    self, temperature: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]: ...
```



## ConstantVaporPressureStrategy

[Show source in vapor_pressure_strategies.py:138](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py#L138)

Concrete implementation of the VaporPressureStrategy using a constant
vapor pressure value.

#### Signature

```python
class ConstantVaporPressureStrategy(VaporPressureStrategy):
    def __init__(self, vapor_pressure: Union[float, NDArray[np.float64]]): ...
```

#### See also

- [VaporPressureStrategy](#vaporpressurestrategy)

### ConstantVaporPressureStrategy().pure_vapor_pressure

[Show source in vapor_pressure_strategies.py:145](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py#L145)

Return the constant vapor pressure value.

#### Arguments

----
- temperature (float or NDArray[np.float64]): Not used.

#### Returns

-------
- vapor_pressure (float or NDArray[np.float64]): The constant vapor
pressure value in Pascals.

#### Signature

```python
def pure_vapor_pressure(
    self, temperature: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]: ...
```



## VaporPressureStrategy

[Show source in vapor_pressure_strategies.py:30](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py#L30)

Abstract class for vapor pressure calculations. The methods
defined here must be implemented by subclasses below.

#### Signature

```python
class VaporPressureStrategy(ABC): ...
```

### VaporPressureStrategy().concentration

[Show source in vapor_pressure_strategies.py:58](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py#L58)

Calculate the concentration of the gas at a given pressure and
temperature.

#### Arguments

- [VaporPressureStrategy().partial_pressure](#vaporpressurestrategypartial_pressure) - Pressure in Pascals.
- `molar_mass` - Molar mass of the gas in kg/mol.
- `temperature` - Temperature in Kelvin.

#### Returns

- `concentration` - The concentration of the gas in kg/m^3.

#### Signature

```python
def concentration(
    self,
    partial_pressure: Union[float, NDArray[np.float64]],
    molar_mass: Union[float, NDArray[np.float64]],
    temperature: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```

### VaporPressureStrategy().partial_pressure

[Show source in vapor_pressure_strategies.py:34](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py#L34)

Calculate the partial pressure of the gas from its concentration, molar
mass, and temperature.

#### Arguments

concentration (float or NDArray[np.float64]): Concentration of the
    gas in kg/m^3.
molar_mass (float or NDArray[np.float64]): Molar mass of the gas in
    kg/mol.
temperature (float or NDArray[np.float64]): Temperature in Kelvin.

#### Returns

- `partial_pressure` - Partial pressure of the gas in Pascals.

#### Signature

```python
def partial_pressure(
    self,
    concentration: Union[float, NDArray[np.float64]],
    molar_mass: Union[float, NDArray[np.float64]],
    temperature: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```

### VaporPressureStrategy().pure_vapor_pressure

[Show source in vapor_pressure_strategies.py:126](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py#L126)

Calculate the pure (saturation) vapor pressure at a given
temperature. Units are in Pascals Pa=kg/(m·s²).

#### Arguments

temperature (float or NDArray[np.float64]): Temperature in Kelvin.

#### Signature

```python
@abstractmethod
def pure_vapor_pressure(
    self, temperature: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]: ...
```

### VaporPressureStrategy().saturation_concentration

[Show source in vapor_pressure_strategies.py:101](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py#L101)

Calculate the saturation concentration of the gas at a given
temperature.

#### Arguments

----
- molar_mass (float or NDArray[np.float64]): Molar mass of the gas in
kg/mol.
- temperature (float or NDArray[np.float64]): Temperature in Kelvin.

#### Returns

-------
- saturation_concentration (float or NDArray[np.float64]):
The saturation concentration of the gas in kg/m^3.

#### Signature

```python
def saturation_concentration(
    self,
    molar_mass: Union[float, NDArray[np.float64]],
    temperature: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```

### VaporPressureStrategy().saturation_ratio

[Show source in vapor_pressure_strategies.py:80](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py#L80)

Calculate the saturation ratio of the gas at a given pressure and
temperature.

#### Arguments

- `pressure` - Pressure in Pascals.
- `temperature` - Temperature in Kelvin.

#### Returns

- `saturation_ratio` - The saturation ratio of the gas.

#### Signature

```python
def saturation_ratio(
    self,
    concentration: Union[float, NDArray[np.float64]],
    molar_mass: Union[float, NDArray[np.float64]],
    temperature: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## WaterBuckStrategy

[Show source in vapor_pressure_strategies.py:260](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py#L260)

Concrete implementation of the VaporPressureStrategy using the
Buck equation for water vapor pressure calculations.

#### Signature

```python
class WaterBuckStrategy(VaporPressureStrategy): ...
```

#### See also

- [VaporPressureStrategy](#vaporpressurestrategy)

### WaterBuckStrategy().pure_vapor_pressure

[Show source in vapor_pressure_strategies.py:264](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py#L264)

Calculate vapor pressure using the Buck equation for water vapor.

#### Arguments

- `temperature` - Temperature in Kelvin.

#### Returns

Vapor pressure in Pascals.

#### References

- Buck, A. L., 1981: New Equations for Computing Vapor Pressure and
    Enhancement Factor. J. Appl. Meteor. Climatol., 20, 1527-1532,
    https://doi.org/10.1175/1520-0450(1981)020<1527:NEFCVP>2.0.CO;2.
- https://en.wikipedia.org/wiki/Arden_Buck_equation

#### Signature

```python
def pure_vapor_pressure(
    self, temperature: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]: ...
```
