# Diffusion Coefficient

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Next](../../index.md#next) / [Particles](../index.md#particles) / [Properties](./index.md#properties) / Diffusion Coefficient

> Auto-generated documentation for [particula.next.particles.properties.diffusion_coefficient](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/properties/diffusion_coefficient.py) module.

## particle_diffusion_coefficient

[Show source in diffusion_coefficient.py:10](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/properties/diffusion_coefficient.py#L10)

Calculate the diffusion coefficient of a particle.

#### Arguments

- `temperature` - The temperature at which the particle is
    diffusing, in Kelvin. Defaults to 298.15 K.
- `boltzmann_constant` - The Boltzmann constant. Defaults to the
    standard value of 1.380649 x 10^-23 J/K.
- `particle_aerodynamic_mobility` - The aerodynamic mobility of
    the particle [m^2/s].

#### Returns

The diffusion coefficient of the particle [m^2/s].

#### Signature

```python
def particle_diffusion_coefficient(
    temperature: Union[float, NDArray[np.float64]],
    particle_aerodynamic_mobility: Union[float, NDArray[np.float64]],
    boltzmann_constant: float = BOLTZMANN_CONSTANT.m,
) -> Union[float, NDArray[np.float64]]: ...
```
