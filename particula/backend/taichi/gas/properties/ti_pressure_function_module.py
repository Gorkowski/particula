"""Taichi-accelerated pressure function module."""
import taichi as ti
import numpy as np
from particula.util.constants import GAS_CONSTANT
from particula.backend.dispatch_register import register

_GAS_CONSTANT = float(GAS_CONSTANT)    # avoid python → kernel capture

@ti.func
def fget_partial_pressure(
    concentration: ti.f64,
    molar_mass: ti.f64,
    temperature: ti.f64,
) -> ti.f64:
    """Element-wise partial pressure (Pa)."""
    return (concentration * _GAS_CONSTANT * temperature) / molar_mass

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
    """Vectorised Taichi wrapper for gas.properties.get_partial_pressure."""
    # 1 · normalise & broadcast
    concentration_np = np.asarray(concentration, dtype=np.float64)
    molar_mass_np = np.asarray(molar_mass, dtype=np.float64)
    temperature_np = np.asarray(temperature, dtype=np.float64)
    concentration_b, molar_mass_b, temperature_b = np.broadcast_arrays(
        concentration_np, molar_mass_np, temperature_np
    )

    # 2 · flatten → Taichi ndarrays
    concentration_flat, molar_mass_flat, temperature_flat = map(
        np.ravel, (concentration_b, molar_mass_b, temperature_b)
    )
    n_elements = concentration_flat.size
    concentration_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    molar_mass_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    temperature_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    concentration_ti.from_numpy(concentration_flat)
    molar_mass_ti.from_numpy(molar_mass_flat)
    temperature_ti.from_numpy(temperature_flat)

    # 3 · kernel launch
    kget_partial_pressure(
        concentration_ti, molar_mass_ti, temperature_ti, result_ti
    )

    # 4 · reshape back & return scalar or array
    result_np = result_ti.to_numpy().reshape(concentration_b.shape)
    return result_np.item() if result_np.size == 1 else result_np


@register("get_saturation_ratio_from_pressure", backend="taichi")
def ti_get_saturation_ratio_from_pressure(partial_pressure, pure_vapor_pressure):
    """Vectorised Taichi wrapper for saturation-ratio calculation."""
    partial_pressure_np = np.asarray(partial_pressure, dtype=np.float64)
    pure_vapor_pressure_np = np.asarray(
        pure_vapor_pressure, dtype=np.float64
    )
    partial_pressure_b, pure_vapor_pressure_b = np.broadcast_arrays(
        partial_pressure_np, pure_vapor_pressure_np
    )

    partial_pressure_flat, pure_vapor_pressure_flat = map(
        np.ravel, (partial_pressure_b, pure_vapor_pressure_b)
    )
    n_elements = partial_pressure_flat.size
    partial_pressure_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    pure_vapor_pressure_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    partial_pressure_ti.from_numpy(partial_pressure_flat)
    pure_vapor_pressure_ti.from_numpy(pure_vapor_pressure_flat)

    kget_saturation_ratio_from_pressure(
        partial_pressure_ti, pure_vapor_pressure_ti, result_ti
    )

    result_np = result_ti.to_numpy().reshape(partial_pressure_b.shape)
    return result_np.item() if result_np.size == 1 else result_np
