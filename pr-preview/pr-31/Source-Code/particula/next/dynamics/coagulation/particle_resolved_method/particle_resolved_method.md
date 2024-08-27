# Particle Resolved Method

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Next](../../index.md#next) / [Dynamics](../index.md#dynamics) / [Coagulation](./index.md#coagulation) / Particle Resolved Method

> Auto-generated documentation for [particula.next.dynamics.coagulation.particle_resolved_method](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/particle_resolved_method.py) module.

## coagulation_step

[Show source in particle_resolved_method.py:66](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/particle_resolved_method.py#L66)

#### Signature

```python
def coagulation_step(
    particle_radius: NDArray[np.float64],
    kernel: NDArray[np.float64],
    kernel_radius: NDArray[np.float64],
    volume: float,
    time_step: float,
    random_generator: np.random.Generator,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```



## particle_resolved_update_step

[Show source in particle_resolved_method.py:21](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/particle_resolved_method.py#L21)

Update the particle radii and concentrations after coagulation events.

#### Arguments

- `particle_radius` *NDArray[float64]* - Array of particle radii.
- `small_index` *NDArray[int64]* - Indices corresponding to smaller
    particles.
- `large_index` *NDArray[int64]* - Indices corresponding to larger
    particles.

#### Returns

- Updated array of particle radii.
- Updated array for the radii of particles that were lost.
- Updated array for the radii of particles that were gained.

#### Signature

```python
def particle_resolved_update_step(
    particle_radius: NDArray[np.float64],
    loss: NDArray[np.float64],
    gain: NDArray[np.float64],
    small_index: NDArray[np.int64],
    large_index: NDArray[np.int64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...
```
