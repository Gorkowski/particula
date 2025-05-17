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
    conc_np = np.asarray(concentration, dtype=np.float64)
    mm_np   = np.asarray(molar_mass,    dtype=np.float64)
    tt_np   = np.asarray(temperature,   dtype=np.float64)
    conc_b, mm_b, tt_b = np.broadcast_arrays(conc_np, mm_np, tt_np)

    # 2 · flatten → Taichi ndarrays
    flat_c, flat_m, flat_t = map(np.ravel, (conc_b, mm_b, tt_b))
    n = flat_c.size
    c_ti  = ti.ndarray(dtype=ti.f64, shape=n)
    m_ti  = ti.ndarray(dtype=ti.f64, shape=n)
    t_ti  = ti.ndarray(dtype=ti.f64, shape=n)
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)
    c_ti.from_numpy(flat_c)
    m_ti.from_numpy(flat_m)
    t_ti.from_numpy(flat_t)

    # 3 · kernel launch
    kget_partial_pressure(c_ti, m_ti, t_ti, res_ti)

    # 4 · reshape back & return scalar or array
    res_np = res_ti.to_numpy().reshape(conc_b.shape)
    return res_np.item() if res_np.size == 1 else res_np


@register("get_saturation_ratio_from_pressure", backend="taichi")
def ti_get_saturation_ratio_from_pressure(partial_pressure, pure_vapor_pressure):
    """Vectorised Taichi wrapper for saturation-ratio calculation."""
    pp_np  = np.asarray(partial_pressure,    dtype=np.float64)
    pvp_np = np.asarray(pure_vapor_pressure, dtype=np.float64)
    pp_b, pvp_b = np.broadcast_arrays(pp_np, pvp_np)

    flat_pp, flat_pvp = map(np.ravel, (pp_b, pvp_b))
    n = flat_pp.size
    pp_ti  = ti.ndarray(dtype=ti.f64, shape=n)
    pvp_ti = ti.ndarray(dtype=ti.f64, shape=n)
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)
    pp_ti.from_numpy(flat_pp)
    pvp_ti.from_numpy(flat_pvp)

    kget_saturation_ratio_from_pressure(pp_ti, pvp_ti, res_ti)

    res_np = res_ti.to_numpy().reshape(pp_b.shape)
    return res_np.item() if res_np.size == 1 else res_np
