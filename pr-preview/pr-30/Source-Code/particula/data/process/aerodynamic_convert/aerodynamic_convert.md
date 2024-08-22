# Aerodynamic Convert

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Data](../index.md#data) / [Process](./index.md#process) / Aerodynamic Convert

> Auto-generated documentation for [particula.data.process.aerodynamic_convert](https://github.com/Gorkowski/particula/blob/main/particula/data/process/aerodynamic_convert.py) module.

## _cost_aerodynamic_radius

[Show source in aerodynamic_convert.py:15](https://github.com/Gorkowski/particula/blob/main/particula/data/process/aerodynamic_convert.py#L15)

Optimization cost function to find the aerodynamic radius of a particle.

#### Arguments

guess_aerodynamic_radius : The initial guess for the
    aerodynamic radius.
mean_free_path_air : The mean free path of air.
particle_radius : The known physical radius of
    the particle.
kwargs : Additional keyword arguments for the optimization function
    - density (float): The density of the particle. Default is 1500
        kg/m^3.
    - reference_density (float): The reference density for the
        aerodynamic radius. Default is 1000 kg/m^3.
    - aerodynamic_shape_factor (float): The aerodynamic shape factor.
        Default is 1.0.

#### Returns

The squared error between the guessed aerodynamic radius and
the calculated aerodynamic radius.

#### Signature

```python
def _cost_aerodynamic_radius(
    guess_aerodynamic_radius: Union[float, NDArray[np.float64]],
    mean_free_path_air: Union[float, NDArray[np.float64]],
    particle_radius: Union[float, NDArray[np.float64]],
    **kwargs
) -> Union[float, NDArray[np.float64]]: ...
```



## _cost_physical_radius

[Show source in aerodynamic_convert.py:75](https://github.com/Gorkowski/particula/blob/main/particula/data/process/aerodynamic_convert.py#L75)

Optimization cost function to find the physical radius of a particle.

#### Arguments

guess_physical_radius : The initial guess for the physical radius.
mean_free_path_air : The mean free path of air.
aerodynamic_radius : The known aerodynamic radius of
    the particle.
kwargs : Additional keyword arguments for the optimization function
    - density (float): The density of the particle. Default is 1500
        kg/m^3.
    - reference_density (float): The reference density for the
        aerodynamic radius. Default is 1000 kg/m^3.
    - aerodynamic_shape_factor (float): The aerodynamic shape factor.
        Default is 1.0.

#### Returns

- `float` - The squared error between the guessed physical radius and the
    calculated aerodynamic radius.

#### Signature

```python
def _cost_physical_radius(
    guess_physical_radius: Union[float, NDArray[np.float64]],
    mean_free_path_air: Union[float, NDArray[np.float64]],
    aerodynamic_radius: Union[float, NDArray[np.float64]],
    **kwargs
) -> Union[float, NDArray[np.float64]]: ...
```



## convert_aerodynamic_to_physical_radius

[Show source in aerodynamic_convert.py:135](https://github.com/Gorkowski/particula/blob/main/particula/data/process/aerodynamic_convert.py#L135)

Convert aerodynamic radius to physical radius for an array of particles.

#### Arguments

aerodynamic_radius : Array of aerodynamic radii to be
    converted.
pressure : The ambient pressure.
temperature : The ambient temperature.
particle_density : The density of the particles.
aerodynamic_shape_factor : The aerodynamic shape factor. Default is 1.
reference_density : The reference density for the aerodynamic radius.
    Default is 1000 kg/m^3.

#### Returns

- `np.ndarray` - Array of physical radii corresponding to the aerodynamic
    radii.

#### Signature

```python
def convert_aerodynamic_to_physical_radius(
    aerodynamic_radius: Union[float, NDArray[np.float64]],
    pressure: float,
    temperature: float,
    particle_density: float,
    aerodynamic_shape_factor: float = 1.0,
    reference_density: float = 1000.0,
) -> Union[float, NDArray[np.float64]]: ...
```



## convert_physical_to_aerodynamic_radius

[Show source in aerodynamic_convert.py:193](https://github.com/Gorkowski/particula/blob/main/particula/data/process/aerodynamic_convert.py#L193)

Convert physical to aerodynamic radius for an array of particles.

#### Arguments

physical_radius : Array of physical radii to be converted.
pressure : The ambient pressure.
temperature : The ambient temperature.
particle_density : The density of the particles.
aerodynamic_shape_factor : The aerodynamic shape factor. Default is 1.
reference_density : The reference density for the aerodynamic radius.
    Default is 1000 kg/m^3.

#### Returns

- `np.ndarray` - Array of aerodynamic radii corresponding to the physical
    radii.

#### Signature

```python
def convert_physical_to_aerodynamic_radius(
    physical_radius: Union[float, NDArray[np.float64]],
    pressure: float,
    temperature: float,
    particle_density: float,
    aerodynamic_shape_factor: float = 1.0,
    reference_density: float = 1000.0,
) -> Union[float, NDArray[np.float64]]: ...
```
