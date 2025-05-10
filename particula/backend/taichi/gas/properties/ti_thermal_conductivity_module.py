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
    result:      ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """Kernel for vectorized thermal conductivity."""
    for i in range(result.shape[0]):
        result[i] = fget_thermal_conductivity(temperature[i])

@register("get_thermal_conductivity", backend="taichi")
def ti_get_thermal_conductivity(temperature):
    """Taichi-accelerated wrapper for thermal conductivity."""
    if not isinstance(temperature, np.ndarray):
        raise TypeError("Taichi backend expects NumPy arrays for the input.")
    t = np.atleast_1d(temperature)
    n = t.size
    t_ti  = ti.ndarray(dtype=ti.f64, shape=n)
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)
    t_ti.from_numpy(t)
    kget_thermal_conductivity(t_ti, res_ti)
    out = res_ti.to_numpy()
    return out.item() if out.size == 1 else out
