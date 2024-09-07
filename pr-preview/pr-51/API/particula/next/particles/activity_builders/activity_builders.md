# Activity Builders

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Particles](./index.md#particles) / Activity Builders

> Auto-generated documentation for [particula.next.particles.activity_builders](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/activity_builders.py) module.

## ActivityIdealMassBuilder

[Show source in activity_builders.py:25](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/activity_builders.py#L25)

Builder class for IdealActivityMass objects. No additional parameters.

#### Methods

- `build()` - Validate and return the IdealActivityMass object.

#### Signature

```python
class ActivityIdealMassBuilder(BuilderABC):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)

### ActivityIdealMassBuilder().build

[Show source in activity_builders.py:36](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/activity_builders.py#L36)

Validate and return the IdealActivityMass object.

#### Returns

- `IdealActivityMass` - The validated IdealActivityMass object.

#### Signature

```python
def build(self) -> ActivityStrategy: ...
```

#### See also

- [ActivityStrategy](./activity_strategies.md#activitystrategy)



## ActivityIdealMolarBuilder

[Show source in activity_builders.py:45](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/activity_builders.py#L45)

Builder class for IdealActivityMolar objects.

#### Methods

- `set_molar_mass(molar_mass,` *molar_mass_units)* - Set the molar mass of the
    particle in kg/mol. Default units are 'kg/mol'.
- `set_parameters(params)` - Set the parameters of the IdealActivityMolar
    object from a dictionary including optional units.
- `build()` - Validate and return the IdealActivityMolar object.

#### Signature

```python
class ActivityIdealMolarBuilder(BuilderABC, BuilderMolarMassMixin):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)
- [BuilderMolarMassMixin](../abc_builder.md#buildermolarmassmixin)

### ActivityIdealMolarBuilder().build

[Show source in activity_builders.py:61](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/activity_builders.py#L61)

Validate and return the IdealActivityMolar object.

#### Returns

- `IdealActivityMolar` - The validated IdealActivityMolar object.

#### Signature

```python
def build(self) -> ActivityStrategy: ...
```

#### See also

- [ActivityStrategy](./activity_strategies.md#activitystrategy)



## ActivityKappaParameterBuilder

[Show source in activity_builders.py:71](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/activity_builders.py#L71)

Builder class for KappaParameterActivity objects.

#### Methods

- `set_kappa(kappa)` - Set the kappa parameter for the activity calculation.
- `set_density(density,density_units)` - Set the density of the species in
    kg/m^3. Default units are 'kg/m^3'.
- `set_molar_mass(molar_mass,molar_mass_units)` - Set the molar mass of the
    species in kg/mol. Default units are 'kg/mol'.
- `set_water_index(water_index)` - Set the array index of the species.
- `set_parameters(dict)` - Set the parameters of the KappaParameterActivity
    object from a dictionary including optional units.
- `build()` - Validate and return the KappaParameterActivity object.

#### Signature

```python
class ActivityKappaParameterBuilder(
    BuilderABC, BuilderDensityMixin, BuilderMolarMassMixin
):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)
- [BuilderDensityMixin](../abc_builder.md#builderdensitymixin)
- [BuilderMolarMassMixin](../abc_builder.md#buildermolarmassmixin)

### ActivityKappaParameterBuilder().build

[Show source in activity_builders.py:134](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/activity_builders.py#L134)

Validate and return the KappaParameterActivity object.

#### Returns

- `KappaParameterActivity` - The validated KappaParameterActivity object.

#### Signature

```python
def build(self) -> ActivityStrategy: ...
```

#### See also

- [ActivityStrategy](./activity_strategies.md#activitystrategy)

### ActivityKappaParameterBuilder().set_kappa

[Show source in activity_builders.py:96](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/activity_builders.py#L96)

Set the kappa parameter for the activity calculation.

#### Arguments

- `kappa` - The kappa parameter for the activity calculation.
- `kappa_units` - Not used. (for interface consistency)

#### Signature

```python
def set_kappa(
    self, kappa: Union[float, NDArray[np.float64]], kappa_units: Optional[str] = None
): ...
```

### ActivityKappaParameterBuilder().set_water_index

[Show source in activity_builders.py:116](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/activity_builders.py#L116)

Set the array index of the species.

#### Arguments

- `water_index` - The array index of the species.
- `water_index_units` - Not used. (for interface consistency)

#### Signature

```python
def set_water_index(self, water_index: int, water_index_units: Optional[str] = None): ...
```
