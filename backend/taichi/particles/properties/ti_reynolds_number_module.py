"""Taichi-accelerated calculation of the particle Reynolds number."""
import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

@ti.func
def fget_particle_reynolds_number(
    particle_radius: ti.f64,
    particle_velocity: ti.f64,
    kinematic_viscosity: ti.f64
) -> ti.f64:
    """Elementwise Taichi function for particle Reynolds number."""
    return (2.0 * particle_radius * particle_velocity) / kinematic_viscosity

@ti.kernel
def kget_particle_reynolds_number(
    particle_radius: ti.types.ndarray(dtype=ti.f64, ndim=1),
    particle_velocity: ti.types.ndarray(dtype=ti.f64, ndim=1),
    kinematic_viscosity: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """Vectorized Taichi kernel for particle Reynolds number."""
    for i in range(result.shape[0]):
        result[i] = fget_particle_reynolds_number(
            particle_radius[i],
            particle_velocity[i],
            kinematic_viscosity[i]
        )

@register("get_particle_reynolds_number", backend="taichi")
def ti_get_particle_reynolds_number(
    particle_radius,
    particle_velocity,
    kinematic_viscosity
):
    """Taichi-accelerated wrapper for particle Reynolds number."""
    if not (
        isinstance(particle_radius, np.ndarray)
        and isinstance(particle_velocity, np.ndarray)
        and isinstance(kinematic_viscosity, np.ndarray)
    ):
        raise TypeError("Taichi backend expects NumPy arrays for all inputs.")

    a1 = np.atleast_1d(particle_radius)
    a2 = np.atleast_1d(particle_velocity)
    a3 = np.atleast_1d(kinematic_viscosity)
    n = a1.size

    a1_ti = ti.ndarray(dtype=ti.f64, shape=n)
    a2_ti = ti.ndarray(dtype=ti.f64, shape=n)
    a3_ti = ti.ndarray(dtype=ti.f64, shape=n)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n)
    a1_ti.from_numpy(a1)
    a2_ti.from_numpy(a2)
    a3_ti.from_numpy(a3)

    kget_particle_reynolds_number(a1_ti, a2_ti, a3_ti, result_ti)

    result_np = result_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np
