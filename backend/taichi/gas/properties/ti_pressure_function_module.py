import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register
from particula.util.constants import GAS_CONSTANT as _R

_R64 = np.float64(_R)

@ti.func
def fget_partial_pressure(
    concentration: ti.f64,
    molar_mass: ti.f64,
    temperature: ti.f64,
) -> ti.f64:
    return (concentration * _R64 * temperature) / molar_mass

@ti.func
def fget_saturation_ratio(
    partial_pressure: ti.f64,
    pure_vapor_pressure: ti.f64,
) -> ti.f64:
    return partial_pressure / pure_vapor_pressure

@ti.kernel
def kget_partial_pressure(
    concentration: ti.types.ndarray(dtype=ti.f64, ndim=1),
    molar_mass: ti.types.ndarray(dtype=ti.f64, ndim=1),
    temperature: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(result.shape[0]):
        result[i] = fget_partial_pressure(
            concentration[i], molar_mass[i], temperature[i]
        )

@ti.kernel
def kget_saturation_ratio(
    partial_pressure:    ti.types.ndarray(dtype=ti.f64, ndim=1),
    pure_vapor_pressure: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result:              ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(result.shape[0]):
        result[i] = fget_saturation_ratio(
            partial_pressure[i], pure_vapor_pressure[i]
        )

@register("get_partial_pressure", backend="taichi")
def ti_get_partial_pressure(concentration, molar_mass, temperature):
    # 1 – type guard
    if not (isinstance(concentration, np.ndarray)
            and isinstance(molar_mass,   np.ndarray)
            and isinstance(temperature,  np.ndarray)):
        raise TypeError("Taichi backend expects NumPy arrays.")
    # 2 – flatten to 1-D NumPy
    c, m, T = map(np.atleast_1d, (concentration, molar_mass, temperature))
    n = c.size
    # 3 – create / populate Ti NDArrays
    c_ti = ti.ndarray(dtype=ti.f64, shape=n);  c_ti.from_numpy(c)
    m_ti = ti.ndarray(dtype=ti.f64, shape=n);  m_ti.from_numpy(m)
    T_ti = ti.ndarray(dtype=ti.f64, shape=n);  T_ti.from_numpy(T)
    out  = ti.ndarray(dtype=ti.f64, shape=n)
    # 4 – launch kernel
    kget_partial_pressure(c_ti, m_ti, T_ti, out)
    # 5 – return NumPy (scalar if size==1)
    res = out.to_numpy()
    return res.item() if res.size == 1 else res

@register("get_saturation_ratio_from_pressure", backend="taichi")
def ti_get_saturation_ratio(partial_pressure, pure_vapor_pressure):
    p, pv = map(np.atleast_1d, (partial_pressure, pure_vapor_pressure))
    n = p.size
    p_ti  = ti.ndarray(dtype=ti.f64, shape=n);  p_ti.from_numpy(p)
    pv_ti = ti.ndarray(dtype=ti.f64, shape=n);  pv_ti.from_numpy(pv)
    out   = ti.ndarray(dtype=ti.f64, shape=n)
    kget_saturation_ratio(p_ti, pv_ti, out)
    res = out.to_numpy()
    return res.item() if res.size == 1 else res
