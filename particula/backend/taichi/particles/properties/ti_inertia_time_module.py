"""
Taichi-accelerated implementation of ``get_particle_inertia_time``.
"""
import taichi as ti
import numpy as np

from particula.backend import register


@ti.func
def fget_particle_inertia_time(                       # scalar version
    particle_radius: ti.f64,
    particle_density: ti.f64,
    fluid_density: ti.f64,
    kinematic_viscosity: ti.f64,
) -> ti.f64:
    return (2.0 / 9.0) * (particle_density / fluid_density) * (
        particle_radius * particle_radius / kinematic_viscosity
    )


@ti.kernel
def kget_particle_inertia_time(                      # vectorised kernel
    particle_radius: ti.types.ndarray(dtype=ti.f64, ndim=1),
    particle_density: ti.types.ndarray(dtype=ti.f64, ndim=1),
    fluid_density: ti.types.ndarray(dtype=ti.f64, ndim=1),
    kinematic_viscosity: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(result.shape[0]):
        result[i] = fget_particle_inertia_time(
            particle_radius[i],
            particle_density[i],
            fluid_density[i],
            kinematic_viscosity[i],
        )


@register("get_particle_inertia_time", backend="taichi")  # public wrapper
def ti_get_particle_inertia_time(
    particle_radius,
    particle_density,
    fluid_density,
    kinematic_viscosity,
):
    if not (
        isinstance(particle_radius, np.ndarray)
        and isinstance(particle_density, np.ndarray)
        and isinstance(fluid_density, np.ndarray)
        and isinstance(kinematic_viscosity, np.ndarray)
    ):
        raise TypeError("Taichi backend expects NumPy arrays for all inputs.")

    pr = np.atleast_1d(particle_radius)
    pd = np.atleast_1d(particle_density)
    fd = np.atleast_1d(fluid_density)
    kv = np.atleast_1d(kinematic_viscosity)

    if not (pr.size == pd.size == fd.size == kv.size):
        raise ValueError("All input arrays must have the same length.")

    n = pr.size
    pr_ti = ti.ndarray(dtype=ti.f64, shape=n)
    pd_ti = ti.ndarray(dtype=ti.f64, shape=n)
    fd_ti = ti.ndarray(dtype=ti.f64, shape=n)
    kv_ti = ti.ndarray(dtype=ti.f64, shape=n)
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)

    pr_ti.from_numpy(pr)
    pd_ti.from_numpy(pd)
    fd_ti.from_numpy(fd)
    kv_ti.from_numpy(kv)

    kget_particle_inertia_time(pr_ti, pd_ti, fd_ti, kv_ti, res_ti)
    return res_ti.to_numpy()
