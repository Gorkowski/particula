"""Taichi implementation of get_dynamic_viscosity."""
import taichi as ti
import numpy as np
from particula.util.constants import (
    REF_TEMPERATURE_STP,
    REF_VISCOSITY_AIR_STP,
    SUTHERLAND_CONSTANT,
)
from particula.backend.dispatch_register import register

# ------------------------------------------------------------------
# 3. scalar (element-wise) Taichi function
@ti.func
def fget_dynamic_viscosity(
    temperature: ti.f64,
    reference_viscosity: ti.f64,
    reference_temperature: ti.f64,
) -> ti.f64:
    # μ(T) = μ₀ × (T / T₀)^(3/2) × (T₀ + S) / (T + S)
    return (
        reference_viscosity
        * (temperature / reference_temperature) ** 1.5
        * (reference_temperature + SUTHERLAND_CONSTANT)
        / (temperature + SUTHERLAND_CONSTANT)
    )

# ------------------------------------------------------------------
# 4. vectorised Taichi kernel
@ti.kernel
def kget_dynamic_viscosity(
    temperature: ti.types.ndarray(dtype=ti.f64, ndim=1),
    reference_viscosity: ti.types.ndarray(dtype=ti.f64, ndim=1),
    reference_temperature: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(result.shape[0]):
        result[i] = fget_dynamic_viscosity(
            temperature[i], reference_viscosity[i], reference_temperature[i]
        )

# ------------------------------------------------------------------
# 5. public wrapper registered for the Taichi backend
@register("get_dynamic_viscosity", backend="taichi")
def get_dynamic_viscosity_taichi(
    temperature,
    reference_viscosity=REF_VISCOSITY_AIR_STP,
    reference_temperature=REF_TEMPERATURE_STP,
):
    """Taichi-accelerated wrapper."""
    # 5 a – type guard
    if not isinstance(temperature, np.ndarray):
        temperature = np.asarray(temperature, dtype=np.float64)

    # broadcast/atleast_1d for the optional parameters so that shapes match
    t_arr = np.atleast_1d(temperature).astype(np.float64)
    rv_arr = np.full_like(t_arr, reference_viscosity, dtype=np.float64)
    rt_arr = np.full_like(t_arr, reference_temperature, dtype=np.float64)
    n = t_arr.size

    # 5 c – allocate Taichi buffers
    t_ti  = ti.ndarray(dtype=ti.f64, shape=n);  t_ti.from_numpy(t_arr)
    rv_ti = ti.ndarray(dtype=ti.f64, shape=n);  rv_ti.from_numpy(rv_arr)
    rt_ti = ti.ndarray(dtype=ti.f64, shape=n);  rt_ti.from_numpy(rt_arr)
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)

    # 5 d – launch
    kget_dynamic_viscosity(t_ti, rv_ti, rt_ti, res_ti)

    # 5 e – back to NumPy & unwrap scalar
    out = res_ti.to_numpy()
    return out.item() if out.size == 1 else out
