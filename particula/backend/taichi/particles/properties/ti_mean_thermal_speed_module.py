"""Taichi implementation of mean thermal speed calculation."""

import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register
from particula.util.constants import BOLTZMANN_CONSTANT

@ti.func
def fget_mean_thermal_speed(
    particle_mass: ti.f64,
    temperature: ti.f64
) -> ti.f64:
    """Elementwise mean thermal speed (Taichi version)."""
    return ti.sqrt(
        (8.0 * BOLTZMANN_CONSTANT * temperature) / (ti.math.pi * particle_mass)
    )

@ti.kernel
def kget_mean_thermal_speed(
    particle_mass: ti.types.ndarray(dtype=ti.f64, ndim=1),
    temperature: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """Vectorized mean thermal speed (Taichi version)."""
    for i in range(result.shape[0]):
        result[i] = fget_mean_thermal_speed(particle_mass[i], temperature[i])

@register("get_mean_thermal_speed", backend="taichi")
def ti_get_mean_thermal_speed(particle_mass, temperature):
    """Public Taichi wrapper for mean thermal speed."""
    if not (isinstance(particle_mass, np.ndarray) and isinstance(temperature, np.ndarray)):
        raise TypeError("Taichi backend expects NumPy arrays for both inputs.")

    pm, temp = np.atleast_1d(particle_mass), np.atleast_1d(temperature)
    if pm.shape != temp.shape:
        raise ValueError("particle_mass and temperature must share the same shape.")

    # flatten to 1-D for Taichi
    pm_flat, temp_flat = pm.ravel(), temp.ravel()
    n = pm_flat.size

    pm_ti   = ti.ndarray(dtype=ti.f64, shape=n)
    temp_ti = ti.ndarray(dtype=ti.f64, shape=n)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n)
    pm_ti.from_numpy(pm_flat)
    temp_ti.from_numpy(temp_flat)

    kget_mean_thermal_speed(pm_ti, temp_ti, result_ti)
    result_np = result_ti.to_numpy().reshape(pm.shape)
    return result_np.item() if result_np.size == 1 else result_np
