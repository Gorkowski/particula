"""Taichi implementation of mean thermal speed calculation."""

import taichi as ti
import numpy as np
from particula.backend import register

# Boltzmann constant in J/K, must match the value used in the NumPy version
BOLTZMANN_CONSTANT = 1.380649e-23

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
    n = pm.size

    pm_ti = ti.ndarray(dtype=ti.f64, shape=n)
    temp_ti = ti.ndarray(dtype=ti.f64, shape=n)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n)
    pm_ti.from_numpy(pm)
    temp_ti.from_numpy(temp)

    kget_mean_thermal_speed(pm_ti, temp_ti, result_ti)
    return result_ti.to_numpy()
