# Surface Strategies

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Particles](./index.md#particles) / Surface Strategies

> Auto-generated documentation for [particula.next.particles.surface_strategies](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/surface_strategies.py) module.

## SurfaceStrategy

[Show source in surface_strategies.py:18](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/surface_strategies.py#L18)

ABC class for Surface Strategies.

Abstract class for implementing strategies to calculate surface tension
and the Kelvin effect for species in particulate phases.

#### Methods

- `effective_surface_tension` - Calculate the effective surface tension of
species based on their concentration.
- `effective_density` - Calculate the effective density of species based on
their concentration.
- `kelvin_radius` - Calculate the Kelvin radius which determines the
curvature effect on vapor pressure.
- `kelvin_term` - Calculate the Kelvin term, which quantifies the effect of
particle curvature on vapor pressure.

#### Signature

```python
class SurfaceStrategy(ABC): ...
```

### SurfaceStrategy().effective_density

[Show source in surface_strategies.py:48](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/surface_strategies.py#L48)

Calculate the effective density of the species mixture.

#### Arguments

- `mass_concentration` - Concentration of the species [kg/m^3].

#### Returns

float or NDArray[float]: Effective density of the species [kg/m^3].

#### Signature

```python
@abstractmethod
def effective_density(
    self, mass_concentration: Union[float, NDArray[np.float_]]
) -> float: ...
```

### SurfaceStrategy().effective_surface_tension

[Show source in surface_strategies.py:35](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/surface_strategies.py#L35)

Calculate the effective surface tension of the species mixture.

#### Arguments

- `mass_concentration` - Concentration of the species [kg/m^3].

#### Returns

float or NDArray[float]: Effective surface tension [N/m].

#### Signature

```python
@abstractmethod
def effective_surface_tension(
    self, mass_concentration: Union[float, NDArray[np.float_]]
) -> float: ...
```

### SurfaceStrategy().kelvin_radius

[Show source in surface_strategies.py:61](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/surface_strategies.py#L61)

Calculate the Kelvin radius which determines the curvature effect.

The kelvin radius is molecule specific and depends on the surface
tension, molar mass, density, and temperature of the system. It is
used to calculate the Kelvin term, which quantifies the effect of
particle curvature on vapor pressure.

#### Arguments

- `surface_tension` - Surface tension of the mixture [N/m].
- `molar_mass` - Molar mass of the species [kg/mol].
- `mass_concentration` - Concentration of the species [kg/m^3].
- `temperature` - Temperature of the system [K].

#### Returns

float or NDArray[float]: Kelvin radius [m].

#### References

- Based on Neil Donahue's approach to the Kelvin equation:
r = 2 * surface_tension * molar_mass / (R * T * density)
[Kelvin Wikipedia](https://en.wikipedia.org/wiki/Kelvin_equation)

#### Signature

```python
def kelvin_radius(
    self,
    molar_mass: Union[float, NDArray[np.float_]],
    mass_concentration: Union[float, NDArray[np.float_]],
    temperature: float,
) -> Union[float, NDArray[np.float_]]: ...
```

### SurfaceStrategy().kelvin_term

[Show source in surface_strategies.py:95](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/surface_strategies.py#L95)

Calculate the Kelvin term, which multiplies the vapor pressure.

The Kelvin term is used to adjust the vapor pressure of a species due
to the curvature of the particle.

#### Arguments

    - `radius` - Radius of the particle [m].
    - `molar_mass` - Molar mass of the species a [kg/mol].
    mass_concentration (float or NDArray[float]): Concentration of the
species [kg/m^3].
- temperature (float): Temperature of the system [K].

#### Returns

--------
- float or NDArray[float]: The exponential factor adjusting vapor
pressure due to curvature.

#### References

Based on Neil Donahue's approach to the Kelvin equation:
exp(kelvin_radius / particle_radius)
- `See` *more* - https://en.wikipedia.org/wiki/Kelvin_equation

#### Signature

```python
def kelvin_term(
    self,
    radius: Union[float, NDArray[np.float_]],
    molar_mass: Union[float, NDArray[np.float_]],
    mass_concentration: Union[float, NDArray[np.float_]],
    temperature: float,
) -> Union[float, NDArray[np.float_]]: ...
```



## SurfaceStrategyMass

[Show source in surface_strategies.py:186](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/surface_strategies.py#L186)

Surface tension and density, based on mass fraction weighted values.

#### Arguments

------------------
- surface_tension (Union[float, NDArray[np.float_]]): Surface tension of
the species [N/m]. If a single value is provided, it will be used for all
species.
- density (Union[float, NDArray[np.float_]]): Density of the species
[kg/m^3]. If a single value is provided, it will be used for all species.

#### References

-----------
- Mass Fractions https://en.wikipedia.org/wiki/Mass_fraction_(chemistry)

#### Signature

```python
class SurfaceStrategyMass(SurfaceStrategy):
    def __init__(
        self,
        surface_tension: Union[float, NDArray[np.float_]] = 0.072,
        density: Union[float, NDArray[np.float_]] = 1000,
    ): ...
```

#### See also

- [SurfaceStrategy](#surfacestrategy)

### SurfaceStrategyMass().effective_density

[Show source in surface_strategies.py:222](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/surface_strategies.py#L222)

#### Signature

```python
def effective_density(
    self, mass_concentration: Union[float, NDArray[np.float_]]
) -> float: ...
```

### SurfaceStrategyMass().effective_surface_tension

[Show source in surface_strategies.py:210](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/surface_strategies.py#L210)

#### Signature

```python
def effective_surface_tension(
    self, mass_concentration: Union[float, NDArray[np.float_]]
) -> float: ...
```



## SurfaceStrategyMolar

[Show source in surface_strategies.py:131](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/surface_strategies.py#L131)

Surface tension and density, based on mole fraction weighted values.

#### Arguments

------------------
- surface_tension (Union[float, NDArray[np.float_]]): Surface tension of
the species [N/m]. If a single value is provided, it will be used for all
species.
- density (Union[float, NDArray[np.float_]]): Density of the species
[kg/m^3]. If a single value is provided, it will be used for all species.
- molar_mass (Union[float, NDArray[np.float_]]): Molar mass of the species
[kg/mol]. If a single value is provided, it will be used for all species.

#### References

-----------
- Mole Fractions https://en.wikipedia.org/wiki/Mole_fraction

#### Signature

```python
class SurfaceStrategyMolar(SurfaceStrategy):
    def __init__(
        self,
        surface_tension: Union[float, NDArray[np.float_]] = 0.072,
        density: Union[float, NDArray[np.float_]] = 1000,
        molar_mass: Union[float, NDArray[np.float_]] = 0.01815,
    ): ...
```

#### See also

- [SurfaceStrategy](#surfacestrategy)

### SurfaceStrategyMolar().effective_density

[Show source in surface_strategies.py:172](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/surface_strategies.py#L172)

#### Signature

```python
def effective_density(
    self, mass_concentration: Union[float, NDArray[np.float_]]
) -> float: ...
```

### SurfaceStrategyMolar().effective_surface_tension

[Show source in surface_strategies.py:159](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/surface_strategies.py#L159)

#### Signature

```python
def effective_surface_tension(
    self, mass_concentration: Union[float, NDArray[np.float_]]
) -> float: ...
```



## SurfaceStrategyVolume

[Show source in surface_strategies.py:233](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/surface_strategies.py#L233)

Surface tension and density, based on volume fraction weighted values.

#### Arguments

------------------
- surface_tension (Union[float, NDArray[np.float_]]): Surface tension of
the species [N/m]. If a single value is provided, it will be used for all
species.
- density (Union[float, NDArray[np.float_]]): Density of the species
[kg/m^3]. If a single value is provided, it will be used for all species.

#### References

-----------
- Volume Fractions https://en.wikipedia.org/wiki/Volume_fraction

#### Signature

```python
class SurfaceStrategyVolume(SurfaceStrategy):
    def __init__(
        self,
        surface_tension: Union[float, NDArray[np.float_]] = 0.072,
        density: Union[float, NDArray[np.float_]] = 1000,
    ): ...
```

#### See also

- [SurfaceStrategy](#surfacestrategy)

### SurfaceStrategyVolume().effective_density

[Show source in surface_strategies.py:270](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/surface_strategies.py#L270)

#### Signature

```python
def effective_density(
    self, mass_concentration: Union[float, NDArray[np.float_]]
) -> float: ...
```

### SurfaceStrategyVolume().effective_surface_tension

[Show source in surface_strategies.py:257](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/surface_strategies.py#L257)

#### Signature

```python
def effective_surface_tension(
    self, mass_concentration: Union[float, NDArray[np.float_]]
) -> float: ...
```
