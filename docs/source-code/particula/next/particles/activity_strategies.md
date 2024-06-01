# Activity Strategies

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Particles](./index.md#particles) / Activity Strategies

> Auto-generated documentation for [particula.next.particles.activity_strategies](../../../../../particula/next/particles/activity_strategies.py) module.

## ActivityStrategy

[Show source in activity_strategies.py:19](../../../../../particula/next/particles/activity_strategies.py#L19)

Abstract base class for implementing vapor pressure strategies based on
particle activity calculations.

#### Signature

```python
class ActivityStrategy(ABC): ...
```

### ActivityStrategy().activity

[Show source in activity_strategies.py:25](../../../../../particula/next/particles/activity_strategies.py#L25)

Calculate the activity of a species based on its mass concentration.

#### Arguments

- mass_concentration (float or NDArray[float]): Concentration of the
species [kg/m^3]

#### Returns

- float or NDArray[float]: Activity of the particle, unitless.

#### Signature

```python
@abstractmethod
def activity(
    self, mass_concentration: Union[float, NDArray[np.float_]]
) -> Union[float, NDArray[np.float_]]: ...
```

### ActivityStrategy().partial_pressure

[Show source in activity_strategies.py:40](../../../../../particula/next/particles/activity_strategies.py#L40)

Calculate the vapor pressure of species in the particle phase based on
activity.

#### Arguments

- pure_vapor_pressure (float or NDArray[float]): Pure vapor pressure
of the species [Pa]
- mass_concentration (float or NDArray[float]): Concentration of the
species [kg/m^3]

#### Returns

- float or NDArray[float]: Vapor pressure of the particle [Pa].

#### Signature

```python
def partial_pressure(
    self,
    pure_vapor_pressure: Union[float, NDArray[np.float_]],
    mass_concentration: Union[float, NDArray[np.float_]],
) -> Union[float, NDArray[np.float_]]: ...
```



## IdealActivityMass

[Show source in activity_strategies.py:105](../../../../../particula/next/particles/activity_strategies.py#L105)

Ideal activity strategy, based on mass fractions.

#### Arguments

------------------
- None needed

#### References

-----------
- Mass Based Raoult's Law https://en.wikipedia.org/wiki/Raoult%27s_law

#### Signature

```python
class IdealActivityMass(ActivityStrategy): ...
```

#### See also

- [ActivityStrategy](#activitystrategy)

### IdealActivityMass().activity

[Show source in activity_strategies.py:117](../../../../../particula/next/particles/activity_strategies.py#L117)

Calculate the activity of a species.

#### Arguments

-----
- mass_concentration (float): Concentration of the species [kg/m^3]

#### Returns

--------
- `-` *float* - Activity of the particle [unitless].

#### Signature

```python
def activity(
    self, mass_concentration: Union[float, NDArray[np.float_]]
) -> Union[float, NDArray[np.float_]]: ...
```



## IdealActivityMolar

[Show source in activity_strategies.py:62](../../../../../particula/next/particles/activity_strategies.py#L62)

Ideal activity strategy, based on mole fractions.

#### Arguments

------------------
- molar_mass (Union[float, NDArray[np.float_]]): Molar mass of the species
[kg/mol]. If a single value is provided, it will be used for all species.

#### References

-----------
- Molar Based Raoult's Law https://en.wikipedia.org/wiki/Raoult%27s_law

#### Signature

```python
class IdealActivityMolar(ActivityStrategy):
    def __init__(self, molar_mass: Union[float, NDArray[np.float_]] = 0.0): ...
```

#### See also

- [ActivityStrategy](#activitystrategy)

### IdealActivityMolar().activity

[Show source in activity_strategies.py:81](../../../../../particula/next/particles/activity_strategies.py#L81)

Calculate the activity of a species.

#### Arguments

-----
- mass_concentration (float): Concentration of the species [kg/m^3]

#### Returns

--------
- `-` *float* - Activity of the particle [unitless].

#### Signature

```python
def activity(
    self, mass_concentration: Union[float, NDArray[np.float_]]
) -> Union[float, NDArray[np.float_]]: ...
```



## KappaParameterActivity

[Show source in activity_strategies.py:138](../../../../../particula/next/particles/activity_strategies.py#L138)

Non-ideal activity strategy, based on kappa hygroscopic parameter for
non-ideal water, and mole fraction for other species.

#### Arguments

------------------
- kappa (NDArray[np.float_]): Kappa hygroscopic parameter [unitless],
include a value for water (that will be removed in the calculation).
- density (NDArray[np.float_]): Density of the species [kg/m^3].
- molar_mass (NDArray[np.float_]): Molar mass of the species [kg/mol].
- water_index (int): Index of water in the mass_concentration array.

#### Signature

```python
class KappaParameterActivity(ActivityStrategy):
    def __init__(
        self,
        kappa: NDArray[np.float_] = np.array([0.0], dtype=np.float_),
        density: NDArray[np.float_] = np.array([0.0], dtype=np.float_),
        molar_mass: NDArray[np.float_] = np.array([0.0], dtype=np.float_),
        water_index: int = 0,
    ): ...
```

#### See also

- [ActivityStrategy](#activitystrategy)

### KappaParameterActivity().activity

[Show source in activity_strategies.py:163](../../../../../particula/next/particles/activity_strategies.py#L163)

Calculate the activity of a species.

#### Arguments

-----
- mass_concentration (float): Concentration of the species [kg/m^3]

#### Returns

--------
- `-` *float* - Activity of the particle [unitless].

#### References

-----------
Petters, M. D., & Kreidenweis, S. M. (2007). A single parameter
representation of hygroscopic growth and cloud condensation nucleus
activity. Atmospheric Chemistry and Physics, 7(8), 1961-1971.
https://doi.org/10.5194/acp-7-1961-2007
EQ 2 and 7

#### Signature

```python
def activity(
    self, mass_concentration: Union[float, NDArray[np.float_]]
) -> Union[float, NDArray[np.float_]]: ...
```
