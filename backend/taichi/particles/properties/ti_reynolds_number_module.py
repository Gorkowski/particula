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
def get_particle_reynolds_number_taichi(
    particle_radius,
    particle_velocity,
    kinematic_viscosity
):
    """Taichi-accelerated wrapper for particle Reynolds number."""
    for var_name, var in dict(r=particle_radius, v=particle_velocity, nu=kinematic_viscosity).items():
        if not isinstance(var, np.ndarray):
            locals()[var_name] = np.asarray(var, dtype=np.float64)

    r_arr, v_arr, nu_arr = np.broadcast_arrays(
        np.atleast_1d(particle_radius),
        np.atleast_1d(particle_velocity),
        np.atleast_1d(kinematic_viscosity),
    )
    n = r_arr.size

    a1_ti = ti.ndarray(dtype=ti.f64, shape=n)
    a2_ti = ti.ndarray(dtype=ti.f64, shape=n)
    a3_ti = ti.ndarray(dtype=ti.f64, shape=n)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n)
    a1_ti.from_numpy(r_arr)
    a2_ti.from_numpy(v_arr)
    a3_ti.from_numpy(nu_arr)

    kget_particle_reynolds_number(a1_ti, a2_ti, a3_ti, result_ti)

    result_np = result_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np
