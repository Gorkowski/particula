"""Taichi implementation of mean thermal speed calculation."""

import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

# Boltzmann constant in J/K (should match particula.util.constants.BOLTZMANN_CONSTANT)
BOLTZMANN_CONSTANT = 1.380649e-23

@ti.func
def fget_mean_thermal_speed(particle_mass: ti.f64, temperature: ti.f64) -> ti.f64:
    """Elementwise mean thermal speed calculation."""
    return ti.sqrt(
        (8.0 * BOLTZMANN_CONSTANT * temperature) / (ti.math.pi * particle_mass)
    )

@ti.kernel
def kget_mean_thermal_speed(
    particle_mass: ti.types.ndarray(dtype=ti.f64, ndim=1),
    temperature: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """Vectorized mean thermal speed calculation."""
    for i in range(result.shape[0]):
        result[i] = fget_mean_thermal_speed(particle_mass[i], temperature[i])

@register("get_mean_thermal_speed", backend="taichi")
def ti_get_mean_thermal_speed(particle_mass, temperature):
    """Taichi-accelerated mean thermal speed calculation."""
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

    result_np = result_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np
