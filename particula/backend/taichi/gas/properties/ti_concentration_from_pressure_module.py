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
    # 5 a – type guard
    if not (
        isinstance(partial_pressure, np.ndarray)
        and isinstance(molar_mass, np.ndarray)
        and isinstance(temperature, np.ndarray)
    ):
        raise TypeError("Taichi backend expects NumPy arrays for all inputs.")

    # 5 b – ensure 1-D
    pp, mm, tt = map(np.atleast_1d, (partial_pressure, molar_mass, temperature))
    n = pp.size

    # 5 c – allocate buffers
    pp_ti = ti.ndarray(dtype=ti.f64, shape=n)
    mm_ti = ti.ndarray(dtype=ti.f64, shape=n)
    tt_ti = ti.ndarray(dtype=ti.f64, shape=n)
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)
    pp_ti.from_numpy(pp)
    mm_ti.from_numpy(mm)
    tt_ti.from_numpy(tt)

    # 5 d – launch kernel
    kget_concentration_from_pressure(pp_ti, mm_ti, tt_ti, res_ti)

    # 5 e – return NumPy / scalar
    res_np = res_ti.to_numpy()
    return res_np.item() if res_np.size == 1 else res_np
