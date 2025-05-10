import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

@ti.func
def fget_particle_reynolds_number(          # scalar version
    particle_radius: ti.f64,
    particle_velocity: ti.f64,
    kinematic_viscosity: ti.f64,
) -> ti.f64:
    return (2.0 * particle_radius * particle_velocity) / kinematic_viscosity

@ti.kernel
def kget_particle_reynolds_number(          # vectorised kernel
    particle_radius: ti.types.ndarray(dtype=ti.f64, ndim=1),
    particle_velocity: ti.types.ndarray(dtype=ti.f64, ndim=1),
    kinematic_viscosity: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(result.shape[0]):
        result[i] = fget_particle_reynolds_number(
            particle_radius[i], particle_velocity[i], kinematic_viscosity[i]
        )

@register("get_particle_reynolds_number", backend="taichi")
def ti_get_particle_reynolds_number(
    particle_radius,
    particle_velocity,
    kinematic_viscosity,
):
    """Taichi backend wrapper for particle Reynolds number."""
    if not (
        isinstance(particle_radius, np.ndarray)
        and isinstance(particle_velocity, np.ndarray)
        and isinstance(kinematic_viscosity, np.ndarray)
    ):
        raise TypeError("Taichi backend expects NumPy arrays for all inputs.")

    pr, pv, kv = map(np.atleast_1d, (particle_radius, particle_velocity, kinematic_viscosity))
    if not (pr.shape == pv.shape == kv.shape):
        raise ValueError("All input arrays must share the same shape.")
    n = pr.size

    pr_ti = ti.ndarray(dtype=ti.f64, shape=n)
    pv_ti = ti.ndarray(dtype=ti.f64, shape=n)
    kv_ti = ti.ndarray(dtype=ti.f64, shape=n)
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)
    pr_ti.from_numpy(pr)
    pv_ti.from_numpy(pv)
    kv_ti.from_numpy(kv)

    kget_particle_reynolds_number(pr_ti, pv_ti, kv_ti, res_ti)
    result_np = res_ti.to_numpy().reshape(pr.shape)
    return result_np.item() if result_np.size == 1 else result_np
