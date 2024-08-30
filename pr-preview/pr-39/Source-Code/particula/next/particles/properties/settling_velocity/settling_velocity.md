# Settling Velocity

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Next](../../index.md#next) / [Particles](../index.md#particles) / [Properties](./index.md#properties) / Settling Velocity

> Auto-generated documentation for [particula.next.particles.properties.settling_velocity](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/properties/settling_velocity.py) module.

## particle_settling_velocity

[Show source in settling_velocity.py:10](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/properties/settling_velocity.py#L10)

Calculate the settling velocity of a particle in a fluid.

#### Arguments

- `particle_radius` - The radius of the particle [m].
- `particle_density` - The density of the particle [kg/m³].
- `slip_correction_factor` - The slip correction factor to
    account for non-continuum effects [dimensionless].
- `gravitational_acceleration` - The gravitational acceleration.
    Defaults to standard gravity [9.80665 m/s²].
- `dynamic_viscosity` - The dynamic viscosity of the fluid [Pa*s].

#### Returns

The settling velocity of the particle in the fluid [m/s].

#### Signature

```python
def particle_settling_velocity(
    particle_radius: Union[float, NDArray[np.float64]],
    particle_density: Union[float, NDArray[np.float64]],
    slip_correction_factor: Union[float, NDArray[np.float64]],
    dynamic_viscosity: float,
    gravitational_acceleration: float = STANDARD_GRAVITY.m,
) -> Union[float, NDArray[np.float64]]: ...
```
