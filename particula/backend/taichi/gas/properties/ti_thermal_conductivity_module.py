"""Taichi implementation of thermal conductivity of air."""
import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

@ti.func
def fget_thermal_conductivity(temperature: ti.f64) -> ti.f64:
    """Element-wise Taichi function for thermal conductivity."""
    return 1e-3 * (4.39 + 0.071 * temperature)

@ti.kernel
def kget_thermal_conductivity(
    temperature: ti.types.ndarray(dtype=ti.f64, ndim=1),
    thermal_conductivity: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """Kernel for vectorized thermal conductivity."""
    for i in range(thermal_conductivity.shape[0]):
        thermal_conductivity[i] = fget_thermal_conductivity(temperature[i])

@register("get_thermal_conductivity", backend="taichi")
def ti_get_thermal_conductivity(temperature):
    """Taichi-accelerated wrapper for thermal conductivity."""
    if not isinstance(temperature, np.ndarray):
        raise TypeError("Taichi backend expects NumPy arrays for the input.")

    temperature_array = np.atleast_1d(temperature)
    n_values = temperature_array.size

    temperature_field = ti.ndarray(dtype=ti.f64, shape=n_values)
    thermal_conductivity_field = ti.ndarray(dtype=ti.f64, shape=n_values)

    temperature_field.from_numpy(temperature_array)
    kget_thermal_conductivity(temperature_field, thermal_conductivity_field)

    thermal_conductivity_array = thermal_conductivity_field.to_numpy()
    return (
        thermal_conductivity_array.item()
        if thermal_conductivity_array.size == 1
        else thermal_conductivity_array
    )
