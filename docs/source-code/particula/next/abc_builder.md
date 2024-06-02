# Abc Builder

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Next](./index.md#next) / Abc Builder

> Auto-generated documentation for [particula.next.abc_builder](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py) module.

## BuilderABC

[Show source in abc_builder.py:20](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L20)

Abstract base class for builders with common methods to check keys and
set parameters from a dictionary.

#### Attributes

----------
- required_parameters (list): List of required parameters for the builder.

#### Methods

--------
- check_keys (parameters): Check if the keys you want to set are
    present in the parameters dictionary.
- set_parameters (parameters): Set parameters from a dictionary including
    optional suffix for units as '_units'.
- `-` *pre_build_check()* - Check if all required attribute parameters are set
    before building.

Abstract Methods:
-----------------
- `-` *build()* - Build and return the strategy object with the set parameters.

#### Raises

------
- `-` *ValueError* - If any required key is missing during check_keys or
    pre_build_check, or if trying to set an invalid parameter.
- `-` *Warning* - If using default units for any parameter.

#### Signature

```python
class BuilderABC(ABC):
    def __init__(self, required_parameters: Optional[list[str]] = None): ...
```

### BuilderABC().build

[Show source in abc_builder.py:144](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L144)

Build and return the strategy object with the set parameters.

#### Returns

-------
- `-` *strategy* - The built strategy object.

#### Signature

```python
@abstractmethod
def build(self) -> Any: ...
```

### BuilderABC().check_keys

[Show source in abc_builder.py:51](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L51)

Check if the keys you want to set are present in the
parameters dictionary and if all keys are valid.

#### Arguments

----
- parameters (dict): The parameters dictionary to check.

#### Returns

-------
- None

#### Raises

------
- `-` *ValueError* - If any required key is missing or if trying to set an
invalid parameter.

#### Signature

```python
def check_keys(self, parameters: dict[str, Any]): ...
```

### BuilderABC().pre_build_check

[Show source in abc_builder.py:126](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L126)

Check if all required attribute parameters are set before building.

#### Returns

-------
- None

#### Raises

------
- `-` *ValueError* - If any required parameter is missing.

#### Signature

```python
def pre_build_check(self): ...
```

### BuilderABC().set_parameters

[Show source in abc_builder.py:93](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L93)

Set parameters from a dictionary including optional suffix for
units as '_units'.

#### Arguments

----
- parameters (dict): The parameters dictionary to set.

#### Returns

-------
- `-` *self* - The builder object with the set parameters.

#### Raises

------
- `-` *ValueError* - If any required key is missing.
- `-` *Warning* - If using default units for any parameter.

#### Signature

```python
def set_parameters(self, parameters: dict[str, Any]): ...
```



## BuilderConcentrationMixin

[Show source in abc_builder.py:255](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L255)

Mixin class for Builder classes to set concentration and
concentration_units.

#### Methods

-------
- `-` *set_concentration(concentration* - float, concentration_units: str):
Set the concentration attribute and units.

#### Signature

```python
class BuilderConcentrationMixin:
    def __init__(self): ...
```

### BuilderConcentrationMixin().set_concentration

[Show source in abc_builder.py:268](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L268)

Set the concentration of the particle in kg/m^3.

#### Arguments

-----
- concentration (float or NDArray[float]): Concentration of the
species or particle in the mixture.
- concentration_units (str, optional): Units of the concentration.
    Default is 'kg/m^3'.

#### Signature

```python
def set_concentration(
    self,
    concentration: Union[float, NDArray[np.float_]],
    concentration_units: Optional[str] = "kg/m^3",
): ...
```



## BuilderDensityMixin

[Show source in abc_builder.py:154](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L154)

Mixin class for Builder classes to set density and density_units.

#### Methods

-------
- `-` *set_density(density* - float, density_units: str): Set the density
    attribute and units.

#### Signature

```python
class BuilderDensityMixin:
    def __init__(self): ...
```

### BuilderDensityMixin().set_density

[Show source in abc_builder.py:166](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L166)

Set the density of the particle in kg/m^3.

#### Arguments

-----
- density (float or NDArray[float]): Density of the particle [kg/m^3].
- density_units (str, optional): Units of the density. Default is
    'kg/m^3'.

#### Signature

```python
def set_density(
    self,
    density: Union[float, NDArray[np.float_]],
    density_units: Optional[str] = "kg/m^3",
): ...
```



## BuilderMolarMassMixin

[Show source in abc_builder.py:221](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L221)

Mixin class for Builder classes to set molar_mass and molar_mass_units.

#### Methods

-------
- `-` *set_molar_mass(molar_mass* - float, molar_mass_units: str): Set the
    molar_mass attribute and units.

#### Signature

```python
class BuilderMolarMassMixin:
    def __init__(self): ...
```

### BuilderMolarMassMixin().set_molar_mass

[Show source in abc_builder.py:233](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L233)

Set the molar mass of the particle in kg/mol.

#### Arguments

-----
- molar_mass (float or NDArray[float]): Molar mass of the particle
    [kg/mol].
- molar_mass_units (str, optional): Units of the molar mass. Default is
    'kg/mol'.

#### Signature

```python
def set_molar_mass(
    self,
    molar_mass: Union[float, NDArray[np.float_]],
    molar_mass_units: Optional[str] = "kg/mol",
): ...
```



## BuilderSurfaceTensionMixin

[Show source in abc_builder.py:186](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L186)

Mixin class for Builder classes to set surface_tension and
surface_tension_units.

#### Methods

-------
- `-` *set_surface_tension(surface_tension* - float, surface_tension_units: str):
    Set the surface_tension attribute and units.

#### Signature

```python
class BuilderSurfaceTensionMixin:
    def __init__(self): ...
```

### BuilderSurfaceTensionMixin().set_surface_tension

[Show source in abc_builder.py:199](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L199)

Set the surface tension of the particle in N/m.

#### Arguments

-----
- surface_tension (float or NDArray[float]): Surface tension of the
    particle [N/m].
- surface_tension_units (str, optional): Units of the surface tension.
    Default is 'N/m'.

#### Signature

```python
def set_surface_tension(
    self,
    surface_tension: Union[float, NDArray[np.float_]],
    surface_tension_units: Optional[str] = "N/m",
): ...
```
