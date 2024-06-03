# Abc Builder

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Next](./index.md#next) / Abc Builder

> Auto-generated documentation for [particula.next.abc_builder](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py) module.

## BuilderABC

[Show source in abc_builder.py:20](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L20)

Abstract base class for builders with common methods to check keys and
set parameters from a dictionary.

#### Attributes

- `required_parameters` - List of required parameters for the builder.

#### Methods

- `check_keys` *parameters* - Check if the keys you want to set are
present in the parameters dictionary.
- `set_parameters` *parameters* - Set parameters from a dictionary including
optional suffix for units as '_units'.
- `pre_build_check()` - Check if all required attribute parameters are set
before building.
- `build` *abstract* - Build and return the strategy object.

#### Raises

- `ValueError` - If any required key is missing during check_keys or
pre_build_check, or if trying to set an invalid parameter.
- `Warning` - If using default units for any parameter.

#### References

This module also defines mixin classes for the Builder classes to set
some optional method to be used in the Builder classes.
[Mixin Wikipedia](https://en.wikipedia.org/wiki/Mixin)

#### Signature

```python
class BuilderABC(ABC):
    def __init__(self, required_parameters: Optional[list[str]] = None): ...
```

### BuilderABC().build

[Show source in abc_builder.py:128](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L128)

Build and return the strategy object with the set parameters.

#### Returns

- `strategy` - The built strategy object.

#### Signature

```python
@abstractmethod
def build(self) -> Any: ...
```

### BuilderABC().check_keys

[Show source in abc_builder.py:50](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L50)

Check if the keys are present and valid.

#### Arguments

- `parameters` - The parameters dictionary to check.

#### Raises

- `ValueError` - If any required key is missing or if trying to set an
invalid parameter.

#### Signature

```python
def check_keys(self, parameters: dict[str, Any]): ...
```

### BuilderABC().pre_build_check

[Show source in abc_builder.py:115](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L115)

Check if all required attribute parameters are set before building.

#### Raises

- `ValueError` - If any required parameter is missing.

#### Signature

```python
def pre_build_check(self): ...
```

### BuilderABC().set_parameters

[Show source in abc_builder.py:85](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L85)

Set parameters from a dictionary including optional suffix for
units as '_units'.

#### Arguments

- `parameters` - The parameters dictionary to set.

#### Returns

- `self` - The builder object with the set parameters.

#### Raises

- `ValueError` - If any required key is missing.
- `Warning` - If using default units for any parameter.

#### Signature

```python
def set_parameters(self, parameters: dict[str, Any]): ...
```



## BuilderChargeMixin

[Show source in abc_builder.py:262](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L262)

Mixin class for Builder classes to set charge and charge_units.

#### Methods

-------
    - `set_charge` - Set the charge attribute and units.

#### Signature

```python
class BuilderChargeMixin:
    def __init__(self): ...
```

### BuilderChargeMixin().set_charge

[Show source in abc_builder.py:273](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L273)

Set the number of elemental charges on the particle.

#### Arguments

- `charge` - Charge of the particle [C].
- `charge_units` - Not used. (for interface consistency)

#### Signature

```python
def set_charge(
    self,
    charge: Union[float, NDArray[np.float_]],
    charge_units: Optional[str] = "unitless",
): ...
```



## BuilderConcentrationMixin

[Show source in abc_builder.py:225](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L225)

Mixin class for Builder classes to set concentration and
concentration_units.

#### Arguments

- `default_units` - Default units of concentration. Default is *kg/m^3*.

#### Methods

- `set_concentration` - Set the concentration attribute and units.

#### Signature

```python
class BuilderConcentrationMixin:
    def __init__(self, default_units: Optional[str] = "kg/m^3"): ...
```

### BuilderConcentrationMixin().set_concentration

[Show source in abc_builder.py:240](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L240)

Set the concentration.

#### Arguments

- `concentration` - Concentration in the mixture.
- `concentration_units` - Units of the concentration.
Default is *kg/m^3*.

#### Signature

```python
def set_concentration(
    self,
    concentration: Union[float, NDArray[np.float_]],
    concentration_units: Optional[str] = None,
): ...
```



## BuilderDensityMixin

[Show source in abc_builder.py:137](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L137)

Mixin class for Builder classes to set density and density_units.

#### Methods

- `set_density` - Set the density attribute and units.

#### Signature

```python
class BuilderDensityMixin:
    def __init__(self): ...
```

### BuilderDensityMixin().set_density

[Show source in abc_builder.py:147](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L147)

Set the density of the particle in kg/m^3.

#### Arguments

- `density` - Density of the particle.
- `density_units` - Units of the density. Default is *kg/m^3*

#### Signature

```python
def set_density(
    self,
    density: Union[float, NDArray[np.float_]],
    density_units: Optional[str] = "kg/m^3",
): ...
```



## BuilderMolarMassMixin

[Show source in abc_builder.py:195](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L195)

Mixin class for Builder classes to set molar_mass and molar_mass_units.

#### Methods

- `set_molar_mass` - Set the molar_mass attribute and units.

#### Signature

```python
class BuilderMolarMassMixin:
    def __init__(self): ...
```

### BuilderMolarMassMixin().set_molar_mass

[Show source in abc_builder.py:205](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L205)

Set the molar mass of the particle in kg/mol.

#### Arguments

-----
- `-` *molar_mass* - Molar mass of the particle.
- `-` *molar_mass_units* - Units of the molar mass. Default is *kg/mol*.

#### Signature

```python
def set_molar_mass(
    self,
    molar_mass: Union[float, NDArray[np.float_]],
    molar_mass_units: Optional[str] = "kg/mol",
): ...
```



## BuilderSurfaceTensionMixin

[Show source in abc_builder.py:165](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L165)

Mixin class for Builder classes to set surface_tension.

#### Methods

-------
    - `set_surface_tension` - Set the surface_tension attribute and units.

#### Signature

```python
class BuilderSurfaceTensionMixin:
    def __init__(self): ...
```

### BuilderSurfaceTensionMixin().set_surface_tension

[Show source in abc_builder.py:176](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L176)

Set the surface tension of the particle in N/m.

#### Arguments

- `surface_tension` - Surface tension of the particle.
- `surface_tension_units` - Surface tension units. Default is *N/m*.

#### Signature

```python
def set_surface_tension(
    self,
    surface_tension: Union[float, NDArray[np.float_]],
    surface_tension_units: Optional[str] = "N/m",
): ...
```
