"""Taichi version of get_concentration_from_pressure."""
import taichi as ti
import numpy as np

from particula.backend.dispatch_register import register
from particula.util.constants import GAS_CONSTANT

_GAS_CONSTANT = float(GAS_CONSTANT)

@ti.func
def fget_concentration_from_pressure(
    partial_pressure: ti.f64,
    molar_mass: ti.f64,
    temperature: ti.f64,
) -> ti.f64:
    return (partial_pressure * molar_mass) / (_GAS_CONSTANT * temperature)

@ti.kernel
def kget_concentration_from_pressure(
    partial_pressure: ti.types.ndarray(dtype=ti.f64, ndim=1),
    molar_mass: ti.types.ndarray(dtype=ti.f64, ndim=1),
    temperature: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(result.shape[0]):
        result[i] = fget_concentration_from_pressure(
            partial_pressure[i], molar_mass[i], temperature[i]
        )

@register("get_concentration_from_pressure", backend="taichi")
def ti_get_concentration_from_pressure(partial_pressure, molar_mass, temperature):
    """Taichi wrapper for get_concentration_from_pressure (vectorised)."""
    # 5 a – type guard  (explicit float64)
    partial_pressure = np.asarray(partial_pressure, dtype=np.float64)
    molar_mass       = np.asarray(molar_mass,       dtype=np.float64)
    temperature      = np.asarray(temperature,      dtype=np.float64)

    # 5 b – broadcast to common shape, then flatten
    pp, mm, tt  = np.broadcast_arrays(partial_pressure, molar_mass, temperature)
    flat_pp, flat_mm, flat_tt = map(np.ravel, (pp, mm, tt))
    n = flat_pp.size

    # 5 c – allocate buffers
    pp_ti = ti.ndarray(dtype=ti.f64, shape=n)
    mm_ti = ti.ndarray(dtype=ti.f64, shape=n)
    tt_ti = ti.ndarray(dtype=ti.f64, shape=n)
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)
    pp_ti.from_numpy(flat_pp)
    mm_ti.from_numpy(flat_mm)
    tt_ti.from_numpy(flat_tt)

    # 5 d – launch kernel
    kget_concentration_from_pressure(pp_ti, mm_ti, tt_ti, res_ti)

    # 5 e – return NumPy / scalar, restoring broadcasted shape
    res_np = res_ti.to_numpy().reshape(pp.shape)
    return res_np.item() if res_np.size == 1 else res_np
