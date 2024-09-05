# Abc Builder

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Next](./index.md#next) / Abc Builder

> Auto-generated documentation for [particula.next.abc_builder](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py) module.

## BuilderABC

[Show source in abc_builder.py:32](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L32)

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

[Show source in abc_builder.py:142](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L142)

Build and return the strategy object with the set parameters.

#### Returns

- `strategy` - The built strategy object.

#### Signature

```python
@abstractmethod
def build(self) -> Any: ...
```

### BuilderABC().check_keys

[Show source in abc_builder.py:62](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L62)

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

[Show source in abc_builder.py:127](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L127)

Check if all required attribute parameters are set before building.

#### Raises

- `ValueError` - If any required parameter is missing.

#### Signature

```python
def pre_build_check(self): ...
```

### BuilderABC().set_parameters

[Show source in abc_builder.py:99](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L99)

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



## BuilderActivityStrategyMixin

[Show source in abc_builder.py:475](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L475)

Mixin class for Builder classes to set activity_strategy.

#### Methods

- `set_activity_strategy` - Set the activity_strategy attribute.

#### Signature

```python
class BuilderActivityStrategyMixin:
    def __init__(self): ...
```

### BuilderActivityStrategyMixin().set_activity_strategy

[Show source in abc_builder.py:485](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L485)

Set the activity strategy of the particle.

#### Arguments

- `activity_strategy` - Activity strategy of the particle.
- `activity_strategy_units` - Not used. (for interface consistency)

#### Signature

```python
def set_activity_strategy(
    self,
    activity_strategy: ActivityStrategy,
    activity_strategy_units: Optional[str] = None,
): ...
```

#### See also

- [ActivityStrategy](particles/activity_strategies.md#activitystrategy)



## BuilderChargeMixin

[Show source in abc_builder.py:283](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L283)

Mixin class for Builder classes to set charge and charge_units.

#### Methods

- `set_charge` - Set the charge attribute and units.

#### Signature

```python
class BuilderChargeMixin:
    def __init__(self): ...
```

### BuilderChargeMixin().set_charge

[Show source in abc_builder.py:293](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L293)

Set the number of elemental charges on the particle.

#### Arguments

- `charge` - Charge of the particle [C].
- `charge_units` - Not used. (for interface consistency)

#### Signature

```python
def set_charge(
    self, charge: Union[float, NDArray[np.float64]], charge_units: Optional[str] = None
): ...
```



## BuilderConcentrationMixin

[Show source in abc_builder.py:244](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L244)

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

[Show source in abc_builder.py:259](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L259)

Set the concentration.

#### Arguments

- `concentration` - Concentration in the mixture.
- `concentration_units` - Units of the concentration.
Default is *kg/m^3*.

#### Signature

```python
def set_concentration(
    self,
    concentration: Union[float, NDArray[np.float64]],
    concentration_units: Optional[str] = None,
): ...
```



## BuilderDensityMixin

[Show source in abc_builder.py:151](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L151)

Mixin class for Builder classes to set density and density_units.

#### Methods

- `set_density` - Set the density attribute and units.

#### Signature

```python
class BuilderDensityMixin:
    def __init__(self): ...
```

### BuilderDensityMixin().set_density

[Show source in abc_builder.py:161](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L161)

Set the density of the particle in kg/m^3.

#### Arguments

- `density` - Density of the particle.
- `density_units` - Units of the density. Default is *kg/m^3*

#### Signature

```python
def set_density(
    self,
    density: Union[float, NDArray[np.float64]],
    density_units: Optional[str] = "kg/m^3",
): ...
```



## BuilderDistributionStrategyMixin

[Show source in abc_builder.py:502](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L502)

Mixin class for Builder classes to set distribution_strategy.

#### Methods

- `set_distribution_strategy` - Set the distribution_strategy attribute.

#### Signature

```python
class BuilderDistributionStrategyMixin:
    def __init__(self): ...
```

### BuilderDistributionStrategyMixin().set_distribution_strategy

[Show source in abc_builder.py:512](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L512)

Set the distribution strategy of the particle.

#### Arguments

- `distribution_strategy` - Distribution strategy of the particle.
- `distribution_strategy_units` - Not used. (for interface consistency)

#### Signature

```python
def set_distribution_strategy(
    self,
    distribution_strategy: DistributionStrategy,
    distribution_strategy_units: Optional[str] = None,
): ...
```

#### See also

- [DistributionStrategy](particles/distribution_strategies.md#distributionstrategy)



## BuilderMassMixin

[Show source in abc_builder.py:310](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L310)

Mixin class for Builder classes to set mass and mass_units.

#### Methods

- `set_mass` - Set the mass attribute and units.

#### Signature

```python
class BuilderMassMixin:
    def __init__(self): ...
```

### BuilderMassMixin().set_mass

[Show source in abc_builder.py:320](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L320)

Set the mass of the particle in kg.

#### Arguments

- `mass` - Mass of the particle.
- `mass_units` - Units of the mass. Default is *kg*.

#### Raises

- `ValueError` - If mass is negative

#### Signature

```python
def set_mass(
    self, mass: Union[float, NDArray[np.float64]], mass_units: Optional[str] = "kg"
): ...
```



## BuilderMolarMassMixin

[Show source in abc_builder.py:212](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L212)

Mixin class for Builder classes to set molar_mass and molar_mass_units.

#### Methods

- `set_molar_mass` - Set the molar_mass attribute and units.

#### Signature

```python
class BuilderMolarMassMixin:
    def __init__(self): ...
```

### BuilderMolarMassMixin().set_molar_mass

[Show source in abc_builder.py:222](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L222)

Set the molar mass of the particle in kg/mol.

#### Arguments

-----
- `-` *molar_mass* - Molar mass of the particle.
- `-` *molar_mass_units* - Units of the molar mass. Default is *kg/mol*.

#### Signature

```python
def set_molar_mass(
    self,
    molar_mass: Union[float, NDArray[np.float64]],
    molar_mass_units: Optional[str] = "kg/mol",
): ...
```



## BuilderPressureMixin

[Show source in abc_builder.py:410](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L410)

Mixin class for AtmosphereBuilder to set total pressure.

#### Methods

- `set_pressure` - Set the total pressure attribute and units.

#### Signature

```python
class BuilderPressureMixin:
    def __init__(self): ...
```

### BuilderPressureMixin().set_pressure

[Show source in abc_builder.py:420](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L420)

Set the total pressure of the atmosphere.

#### Arguments

- `total_pressure` - Total pressure of the gas mixture.
- `pressure_units` - Units of the pressure. Options include
    'Pa', 'kPa', 'MPa', 'psi', 'bar', 'atm'. Default is 'Pa'.

#### Returns

- `AtmosphereBuilderMixin` - This object instance with updated pressure.

#### Raises

- `ValueError` - If the total pressure is below zero.

#### Signature

```python
def set_pressure(
    self, pressure: Union[float, NDArray[np.float64]], pressure_units: str = "Pa"
): ...
```



## BuilderRadiusMixin

[Show source in abc_builder.py:342](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L342)

Mixin class for Builder classes to set radius and radius_units.

#### Methods

- `set_radius` - Set the radius attribute and units.

#### Signature

```python
class BuilderRadiusMixin:
    def __init__(self): ...
```

### BuilderRadiusMixin().set_radius

[Show source in abc_builder.py:352](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L352)

Set the radius of the particle in meters.

#### Arguments

- `radius` - Radius of the particle.
- `radius_units` - Units of the radius. Default is *m*.

#### Raises

- `ValueError` - If radius is negative

#### Signature

```python
def set_radius(
    self, radius: Union[float, NDArray[np.float64]], radius_units: Optional[str] = "m"
): ...
```



## BuilderSurfaceStrategyMixin

[Show source in abc_builder.py:448](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L448)

Mixin class for Builder classes to set surface_strategy.

#### Methods

- `set_surface_strategy` - Set the surface_strategy attribute.

#### Signature

```python
class BuilderSurfaceStrategyMixin:
    def __init__(self): ...
```

### BuilderSurfaceStrategyMixin().set_surface_strategy

[Show source in abc_builder.py:458](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L458)

Set the surface strategy of the particle.

#### Arguments

- `surface_strategy` - Surface strategy of the particle.
- `surface_strategy_units` - Not used. (for interface consistency)

#### Signature

```python
def set_surface_strategy(
    self, surface_strategy: SurfaceStrategy, surface_strategy_units: Optional[str] = None
): ...
```

#### See also

- [SurfaceStrategy](particles/surface_strategies.md#surfacestrategy)



## BuilderSurfaceTensionMixin

[Show source in abc_builder.py:180](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L180)

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

[Show source in abc_builder.py:191](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L191)

Set the surface tension of the particle in N/m.

#### Arguments

- `surface_tension` - Surface tension of the particle.
- `surface_tension_units` - Surface tension units. Default is *N/m*.

#### Signature

```python
def set_surface_tension(
    self,
    surface_tension: Union[float, NDArray[np.float64]],
    surface_tension_units: Optional[str] = "N/m",
): ...
```



## BuilderTemperatureMixin

[Show source in abc_builder.py:374](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L374)

Mixin class for AtmosphereBuilder to set temperature.

#### Methods

- `set_temperature` - Set the temperature attribute and units.

#### Signature

```python
class BuilderTemperatureMixin:
    def __init__(self): ...
```

### BuilderTemperatureMixin().set_temperature

[Show source in abc_builder.py:384](https://github.com/Gorkowski/particula/blob/main/particula/next/abc_builder.py#L384)

Set the temperature of the atmosphere.

#### Arguments

- `temperature` *float* - Temperature of the gas mixture.
- `temperature_units` *str* - Units of the temperature.
    Options include 'degC', 'degF', 'degR', 'K'. Default is 'K'.

#### Returns

- `AtmosphereBuilderMixin` - This object instance with updated
    temperature.

#### Raises

- `ValueError` - If the converted temperature is below absolute zero.

#### Signature

```python
def set_temperature(self, temperature: float, temperature_units: str = "K"): ...
```
