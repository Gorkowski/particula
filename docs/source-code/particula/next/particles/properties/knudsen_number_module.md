# Knudsen Number Module

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Next](../../index.md#next) / [Particles](../index.md#particles) / [Properties](./index.md#properties) / Knudsen Number Module

> Auto-generated documentation for [particula.next.particles.properties.knudsen_number_module](../../../../../../particula/next/particles/properties/knudsen_number_module.py) module.

## calculate_knudsen_number

[Show source in knudsen_number_module.py:9](../../../../../../particula/next/particles/properties/knudsen_number_module.py#L9)

Calculate the Knudsen number using the mean free path of the gas and the
radius of the particle. The Knudsen number is a dimensionless number that
indicates the regime of gas flow relative to the size of particles.

#### Arguments

-----
- mean_free_path (Union[float, NDArray[np.float_]]): The mean free path of
the gas molecules [meters (m)].
- particle_radius (Union[float, NDArray[np.float_]]): The radius of the
particle [meters (m)].

#### Returns

--------
- Union[float, NDArray[np.float_]]: The Knudsen number, which is the
ratio of the mean free path to the particle radius.

#### References

-----------
- For more information at https://en.wikipedia.org/wiki/Knudsen_number

#### Signature

```python
def calculate_knudsen_number(
    mean_free_path: Union[float, NDArray[np.float_]],
    particle_radius: Union[float, NDArray[np.float_]],
) -> Union[float, NDArray[np.float_]]: ...
```