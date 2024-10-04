# Vapor Pressure Builders

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Gas](./index.md#gas) / Vapor Pressure Builders

> Auto-generated documentation for [particula.next.gas.vapor_pressure_builders](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_builders.py) module.

## AntoineBuilder

[Show source in vapor_pressure_builders.py:16](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_builders.py#L16)

Builder class for AntoineVaporPressureStrategy. It allows setting the
coefficients 'a', 'b', and 'c' separately and then building the strategy
object.

- Equation: log10(P_mmHG) = a - b / (Temperature_K - c)
- Units: 'a_units' = None, 'b_units' = 'K', 'c_units' = 'K'

#### Methods

--------
- set_a(a, a_units): Set the coefficient 'a' of the Antoine equation.
- set_b(b, b_units): Set the coefficient 'b' of the Antoine equation.
- set_c(c, c_units): Set the coefficient 'c' of the Antoine equation.
- `-` *set_parameters(params)* - Set coefficients from a dictionary including
    optional units.
- `-` *build()* - Build the AntoineVaporPressureStrategy object with the set
    coefficients.

#### Signature

```python
class AntoineBuilder:
    def __init__(self): ...
```

### AntoineBuilder().build

[Show source in vapor_pressure_builders.py:86](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_builders.py#L86)

Build the AntoineVaporPressureStrategy object with the set
coefficients.

#### Signature

```python
def build(self): ...
```

### AntoineBuilder().set_a

[Show source in vapor_pressure_builders.py:40](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_builders.py#L40)

Set the coefficient 'a' of the Antoine equation.

#### Signature

```python
def set_a(self, a: float, a_units: Optional[str] = None): ...
```

### AntoineBuilder().set_b

[Show source in vapor_pressure_builders.py:50](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_builders.py#L50)

Set the coefficient 'b' of the Antoine equation.

#### Signature

```python
def set_b(self, b: float, b_units: str = "K"): ...
```

### AntoineBuilder().set_c

[Show source in vapor_pressure_builders.py:58](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_builders.py#L58)

Set the coefficient 'c' of the Antoine equation.

#### Signature

```python
def set_c(self, c: float, c_units: str = "K"): ...
```

### AntoineBuilder().set_parameters

[Show source in vapor_pressure_builders.py:66](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_builders.py#L66)

Set coefficients from a dictionary including optional units.

#### Signature

```python
def set_parameters(self, parameters: ignore): ...
```



## ClausiusClapeyronBuilder

[Show source in vapor_pressure_builders.py:96](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_builders.py#L96)

Builder class for ClausiusClapeyronStrategy. This class facilitates
setting the latent heat of vaporization, initial temperature, and initial
pressure with unit handling and then builds the strategy object.

- Equation: dP/dT = L / (R * T^2)
- Units: 'latent_heat_units' = 'J/kg', 'temperature_initial_units' = 'K',
    'pressure_initial_units' = 'Pa'

#### Methods

--------
- set_latent_heat(latent_heat, latent_heat_units): Set the latent heat of
    vaporization.
- set_temperature_initial(temperature_initial, temperature_initial_units):
    Set the initial temperature.
- set_pressure_initial(pressure_initial, pressure_initial_units): Set the
    initial pressure.
- `-` *set_parameters(parameters)* - Set parameters from a dictionary including
    optional units.
- `-` *build()* - Build the ClausiusClapeyronStrategy object with the set
    parameters.

#### Signature

```python
class ClausiusClapeyronBuilder:
    def __init__(self): ...
```

### ClausiusClapeyronBuilder().build

[Show source in vapor_pressure_builders.py:184](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_builders.py#L184)

Build and return a ClausiusClapeyronStrategy object with the set
parameters.

#### Signature

```python
def build(self): ...
```

### ClausiusClapeyronBuilder().set_latent_heat

[Show source in vapor_pressure_builders.py:124](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_builders.py#L124)

Set the latent heat of vaporization: Default units J/kg.

#### Signature

```python
def set_latent_heat(self, latent_heat: float, latent_heat_units: str = "J/kg"): ...
```

### ClausiusClapeyronBuilder().set_parameters

[Show source in vapor_pressure_builders.py:162](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_builders.py#L162)

Set parameters from a dictionary including optional units.

#### Signature

```python
def set_parameters(self, parameters: ignore): ...
```

### ClausiusClapeyronBuilder().set_pressure_initial

[Show source in vapor_pressure_builders.py:149](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_builders.py#L149)

Set the initial pressure. Default units: Pa.

#### Signature

```python
def set_pressure_initial(
    self, pressure_initial: float, pressure_initial_units: str = "Pa"
): ...
```

### ClausiusClapeyronBuilder().set_temperature_initial

[Show source in vapor_pressure_builders.py:136](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_builders.py#L136)

Set the initial temperature. Default units: K.

#### Signature

```python
def set_temperature_initial(
    self, temperature_initial: float, temperature_initial_units: str = "K"
): ...
```



## ConstantBuilder

[Show source in vapor_pressure_builders.py:204](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_builders.py#L204)

Builder class for ConstantVaporPressureStrategy. This class facilitates
setting the constant vapor pressure and then building the strategy object.

- Equation: P = vapor_pressure
- Units: 'vapor_pressure_units' = 'Pa'

#### Methods

--------
- set_vapor_pressure(constant, constant_units): Set the constant vapor
pressure.
- `-` *set_parameters(parameters)* - Set parameters from a dictionary including
    optional units.
- `-` *build()* - Build the ConstantVaporPressureStrategy object with the set
    parameters.

#### Signature

```python
class ConstantBuilder:
    def __init__(self): ...
```

### ConstantBuilder().build

[Show source in vapor_pressure_builders.py:255](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_builders.py#L255)

Build and return a ConstantVaporPressureStrategy object with the set
parameters.

#### Signature

```python
def build(self): ...
```

### ConstantBuilder().set_parameters

[Show source in vapor_pressure_builders.py:236](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_builders.py#L236)

Set parameters from a dictionary including optional units.

#### Signature

```python
def set_parameters(self, parameters: ignore): ...
```

### ConstantBuilder().set_vapor_pressure

[Show source in vapor_pressure_builders.py:224](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_builders.py#L224)

Set the constant vapor pressure.

#### Signature

```python
def set_vapor_pressure(
    self, vapor_pressure: float, vapor_pressure_units: str = "Pa"
): ...
```



## WaterBuckBuilder

[Show source in vapor_pressure_builders.py:263](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_builders.py#L263)

Builder class for WaterBuckStrategy. This class facilitates
the building of the WaterBuckStrategy object. Which as of now has no
additional parameters to set. But could be extended in the future for
ice only calculations. We keep the builder for consistency.

#### Methods

--------
- `-` *build()* - Build the WaterBuckStrategy object.

#### Signature

```python
class WaterBuckBuilder: ...
```

### WaterBuckBuilder().build

[Show source in vapor_pressure_builders.py:273](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_builders.py#L273)

Build and return a WaterBuckStrategy object.

#### Signature

```python
def build(self): ...
```
