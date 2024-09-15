# Vapor Pressure Strategies

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Gas](./index.md#gas) / Vapor Pressure Strategies

> Auto-generated documentation for [particula.next.gas.vapor_pressure_strategies](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py) module.

## AntoineVaporPressureStrategy

[Show source in vapor_pressure_strategies.py:168](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py#L168)

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

[Show source in vapor_pressure_strategies.py:184](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py#L184)

Calculate the pure (saturation) vapor pressure using the Antoine
equation.

#### Arguments

----
- temperature (float or NDArray[np.float64]): Temperature in Kelvin.

#### Returns

-------
- vapor_pressure (float or NDArray[np.float64]): The vapor pressure in
Pascals.

#### References

----------
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

[Show source in vapor_pressure_strategies.py:214](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py#L214)

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

[Show source in vapor_pressure_strategies.py:243](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py#L243)

Calculate the vapor pressure at a new temperature using the
Clausius-Clapeyron equation. For ideal gases at low temperatures.

#### Arguments

----
- temperature_initial (float or NDArray[np.float64]): Initial
temperature in Kelvin.
- pressure_initial (float or NDArray[np.float64]): Initial vapor
pressure in Pascals.
- temperature_final (float or NDArray[np.float64]): Final temperature
in Kelvin.

#### Returns

- vapor_pressure_final (float or NDArray[np.float64]): Final vapor
pressure in Pascals.

#### References

----------
- https://en.wikipedia.org/wiki/Clausius%E2%80%93Clapeyron_relation
Ideal_gas_approximation_at_low_temperatures

#### Signature

```python
def pure_vapor_pressure(
    self, temperature: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]: ...
```



## ConstantVaporPressureStrategy

[Show source in vapor_pressure_strategies.py:142](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py#L142)

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

[Show source in vapor_pressure_strategies.py:149](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py#L149)

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

[Show source in vapor_pressure_strategies.py:24](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py#L24)

Abstract class for vapor pressure calculations. The methods
defined here must be implemented by subclasses below.

#### Signature

```python
class VaporPressureStrategy(ABC): ...
```

### VaporPressureStrategy().concentration

[Show source in vapor_pressure_strategies.py:55](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py#L55)

Calculate the concentration of the gas at a given pressure and
temperature.

#### Arguments

----
- partial_pressure (float or NDArray[np.float64]): Pressure in Pascals.
- molar_mass (float or NDArray[np.float64]): Molar mass of the gas in
kg/mol.
- temperature (float or NDArray[np.float64]): Temperature in Kelvin.

#### Returns

-------
- concentration (float or NDArray[np.float64]): The concentration of the
gas in kg/m^3.

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

[Show source in vapor_pressure_strategies.py:28](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py#L28)

Calculate the partial pressure of the gas from its concentration, molar
mass, and temperature.

#### Arguments

----
- concentration (float or NDArray[np.float64]): Concentration of the gas
in kg/m^3.
- molar_mass (float or NDArray[np.float64]): Molar mass of the gas in
kg/mol.
- temperature (float or NDArray[np.float64]): Temperature in Kelvin.

#### Returns

-------
- partial_pressure (float or NDArray[np.float64]): Partial pressure of
the gas in Pascals.

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

[Show source in vapor_pressure_strategies.py:130](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py#L130)

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

[Show source in vapor_pressure_strategies.py:105](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py#L105)

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

[Show source in vapor_pressure_strategies.py:81](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py#L81)

Calculate the saturation ratio of the gas at a given pressure and
temperature.

#### Arguments

----
- pressure (float or NDArray[np.float64]): Pressure in Pascals.
- temperature (float or NDArray[np.float64]): Temperature in Kelvin.

#### Returns

-------
- saturation_ratio (float or NDArray[np.float64]): The saturation ratio
of the gas.

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

[Show source in vapor_pressure_strategies.py:274](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py#L274)

Concrete implementation of the VaporPressureStrategy using the
Buck equation for water vapor pressure calculations.

#### Signature

```python
class WaterBuckStrategy(VaporPressureStrategy): ...
```

#### See also

- [VaporPressureStrategy](#vaporpressurestrategy)

### WaterBuckStrategy().pure_vapor_pressure

[Show source in vapor_pressure_strategies.py:278](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py#L278)

Calculate the pure (saturation) vapor pressure using the Buck
equation for water vapor.

#### Arguments

----
- temperature (float or NDArray[np.float64]): Temperature in Kelvin.

#### Returns

-------
- vapor_pressure (float or NDArray[np.float64]): The vapor pressure in
Pascals.

#### References

----------
Buck, A. L., 1981: New Equations for Computing Vapor Pressure and
Enhancement Factor. J. Appl. Meteor. Climatol., 20, 1527-1532,
https://doi.org/10.1175/1520-0450(1981)020<1527:NEFCVP>2.0.CO;2.

https://en.wikipedia.org/wiki/Arden_Buck_equation

#### Signature

```python
def pure_vapor_pressure(
    self, temperature: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]: ...
```