# Environment

[Particula Index](../README.md#particula-index) / [Particula](./index.md#particula) / Environment

> Auto-generated documentation for [particula.environment](../../particula/environment.py) module.

- [Environment](#environment)
  - [Environment](#environment-1)
    - [Environment().dynamic_viscosity](#environment()dynamic_viscosity)
    - [Environment().mean_free_path](#environment()mean_free_path)
    - [Environment().water_vapor_concentration](#environment()water_vapor_concentration)
  - [SharedProperties](#sharedproperties)
    - [SharedProperties().dilution_rate_coefficient](#sharedproperties()dilution_rate_coefficient)

## Environment

[Show source in environment.py:67](../../particula/environment.py#L67)

creating the environment class

For now, the environment class takes properties such as
temperature and pressure to calculate derived properties
such as viscosity and mean free path.

#### Signature

```python
class Environment(SharedProperties):
    def __init__(self, **kwargs): ...
```

#### See also

- [SharedProperties](#sharedproperties)

### Environment().dynamic_viscosity

[Show source in environment.py:110](../../particula/environment.py#L110)

Returns the dynamic viscosity in Pa*s.

#### Signature

```python
def dynamic_viscosity(self): ...
```

### Environment().mean_free_path

[Show source in environment.py:120](../../particula/environment.py#L120)

Returns the mean free path in m.

#### Signature

```python
def mean_free_path(self): ...
```

### Environment().water_vapor_concentration

[Show source in environment.py:131](../../particula/environment.py#L131)

Returns the water vapor concentration in kg/m^3.

#### Signature

```python
def water_vapor_concentration(self): ...
```



## SharedProperties

[Show source in environment.py:41](../../particula/environment.py#L41)

 a hidden class for sharing properties like
coagulation_approximation

#### Signature

```python
class SharedProperties:
    def __init__(self, **kwargs): ...
```

### SharedProperties().dilution_rate_coefficient

[Show source in environment.py:58](../../particula/environment.py#L58)

get the dilution rate coefficient

#### Signature

```python
def dilution_rate_coefficient(self): ...
```