"""Taichi implementation of aerodynamic mobility of a particle in a fluid."""
import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

@ti.func
def fget_aerodynamic_mobility(
    particle_radius: ti.f64,
    slip_correction_factor: ti.f64,
    dynamic_viscosity: ti.f64,
) -> ti.f64:
    """Elementwise Taichi version of aerodynamic mobility."""
    return slip_correction_factor / (6.0 * ti.math.pi * dynamic_viscosity * particle_radius)

@ti.kernel
def kget_aerodynamic_mobility(
    particle_radius: ti.types.ndarray(dtype=ti.f64, ndim=1),
    slip_correction_factor: ti.types.ndarray(dtype=ti.f64, ndim=1),
    dynamic_viscosity: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """Vectorized Taichi kernel for aerodynamic mobility."""
    for i in range(result.shape[0]):
        result[i] = fget_aerodynamic_mobility(
            particle_radius[i],
            slip_correction_factor[i],
            dynamic_viscosity[i]
        )

@register("get_aerodynamic_mobility", backend="taichi")
def ti_get_aerodynamic_mobility(
    particle_radius,
    slip_correction_factor,
    dynamic_viscosity,
):
    """Taichi wrapper for aerodynamic mobility."""
    if not (
        isinstance(particle_radius, np.ndarray)
        and isinstance(slip_correction_factor, np.ndarray)
        and isinstance(dynamic_viscosity, np.ndarray)
    ):
        raise TypeError("Taichi backend expects NumPy arrays for all inputs.")

    a1 = np.atleast_1d(particle_radius)
    a2 = np.atleast_1d(slip_correction_factor)
    a3 = np.atleast_1d(dynamic_viscosity)
    n = a1.size

    a1_ti = ti.ndarray(dtype=ti.f64, shape=n)
    a2_ti = ti.ndarray(dtype=ti.f64, shape=n)
    a3_ti = ti.ndarray(dtype=ti.f64, shape=n)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n)
    a1_ti.from_numpy(a1)
    a2_ti.from_numpy(a2)
    a3_ti.from_numpy(a3)

    kget_aerodynamic_mobility(a1_ti, a2_ti, a3_ti, result_ti)

    result_np = result_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np
