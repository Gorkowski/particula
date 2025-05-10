"""Taichi-accelerated pressure function module."""
import taichi as ti
import numpy as np
from particula.util.constants import GAS_CONSTANT
from particula.backend.dispatch_register import register

@ti.func
def fget_partial_pressure(
    concentration: ti.f64,
    molar_mass: ti.f64,
    temperature: ti.f64,
) -> ti.f64:
    """Elementwise partial pressure calculation (Taichi)."""
    return (concentration * ti.cast(GAS_CONSTANT, ti.f64) * temperature) / molar_mass

@ti.func
def fget_saturation_ratio_from_pressure(
    partial_pressure: ti.f64,
    pure_vapor_pressure: ti.f64,
) -> ti.f64:
    """Elementwise saturation ratio calculation (Taichi)."""
    return partial_pressure / pure_vapor_pressure

@ti.kernel
def kget_partial_pressure(
    concentration: ti.types.ndarray(dtype=ti.f64, ndim=1),
    molar_mass: ti.types.ndarray(dtype=ti.f64, ndim=1),
    temperature: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """Vectorized partial pressure calculation (Taichi)."""
    for i in range(result.shape[0]):
        result[i] = fget_partial_pressure(
            concentration[i], molar_mass[i], temperature[i]
        )

@ti.kernel
def kget_saturation_ratio_from_pressure(
    partial_pressure: ti.types.ndarray(dtype=ti.f64, ndim=1),
    pure_vapor_pressure: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """Vectorized saturation ratio calculation (Taichi)."""
    for i in range(result.shape[0]):
        result[i] = fget_saturation_ratio_from_pressure(
            partial_pressure[i], pure_vapor_pressure[i]
        )

@register("get_partial_pressure", backend="taichi")
def ti_get_partial_pressure(concentration, molar_mass, temperature):
    """Taichi wrapper for partial pressure calculation."""
    if not (
        isinstance(concentration, np.ndarray)
        and isinstance(molar_mass, np.ndarray)
        and isinstance(temperature, np.ndarray)
    ):
        raise TypeError("Taichi backend expects NumPy arrays for all inputs.")

    a1, a2, a3 = (
        np.atleast_1d(concentration),
        np.atleast_1d(molar_mass),
        np.atleast_1d(temperature),
    )
    n = a1.size
    arr1 = ti.ndarray(dtype=ti.f64, shape=n)
    arr2 = ti.ndarray(dtype=ti.f64, shape=n)
    arr3 = ti.ndarray(dtype=ti.f64, shape=n)
    result = ti.ndarray(dtype=ti.f64, shape=n)
    arr1.from_numpy(a1)
    arr2.from_numpy(a2)
    arr3.from_numpy(a3)
    kget_partial_pressure(arr1, arr2, arr3, result)
    result_np = result.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np

@register("get_saturation_ratio_from_pressure", backend="taichi")
def ti_get_saturation_ratio_from_pressure(partial_pressure, pure_vapor_pressure):
    """Taichi wrapper for saturation ratio calculation."""
    if not (
        isinstance(partial_pressure, np.ndarray)
        and isinstance(pure_vapor_pressure, np.ndarray)
    ):
        raise TypeError("Taichi backend expects NumPy arrays for both inputs.")

    a1, a2 = np.atleast_1d(partial_pressure), np.atleast_1d(pure_vapor_pressure)
    n = a1.size
    arr1 = ti.ndarray(dtype=ti.f64, shape=n)
    arr2 = ti.ndarray(dtype=ti.f64, shape=n)
    result = ti.ndarray(dtype=ti.f64, shape=n)
    arr1.from_numpy(a1)
    arr2.from_numpy(a2)
    kget_saturation_ratio_from_pressure(arr1, arr2, result)
    result_np = result.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np
