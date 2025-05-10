from taichi import f64, func, kernel, ndarray, types
import taichi as ti
import numpy as np

from particula.backend.dispatch_register import register
from particula.util.constants import (
    REF_TEMPERATURE_STP,
    REF_VISCOSITY_AIR_STP,
    SUTHERLAND_CONSTANT,
)

# ── 3. element-wise function ──────────────────────────────────────────
@ti.func
def fget_dynamic_viscosity(
    temperature: f64,
    reference_viscosity: f64,
    reference_temperature: f64,
) -> f64:
    return (
        reference_viscosity
        * (temperature / reference_temperature) ** 1.5
        * (reference_temperature + SUTHERLAND_CONSTANT)
        / (temperature + SUTHERLAND_CONSTANT)
    )

# ── 4. vectorised kernel ──────────────────────────────────────────────
@ti.kernel
def kget_dynamic_viscosity(                    # 1-D only
    temperature: ti.types.ndarray(dtype=f64, ndim=1),
    reference_viscosity: ti.types.ndarray(dtype=f64, ndim=1),
    reference_temperature: ti.types.ndarray(dtype=f64, ndim=1),
    result: ti.types.ndarray(dtype=f64, ndim=1),
):
    for i in range(result.shape[0]):
        result[i] = fget_dynamic_viscosity(
            temperature[i],
            reference_viscosity[i],
            reference_temperature[i],
        )

# ── 5. public wrapper, backend registration ───────────────────────────
@register("get_dynamic_viscosity", backend="taichi")
def ti_get_dynamic_viscosity(
    temperature,
    reference_viscosity: float = REF_VISCOSITY_AIR_STP,
    reference_temperature: float = REF_TEMPERATURE_STP,
):
    # 5 a – type / shape guards
    t_np = np.atleast_1d(temperature).astype(np.float64)

    rv_np = (
        np.full_like(t_np, reference_viscosity, dtype=np.float64)
        if np.isscalar(reference_viscosity)
        else np.asarray(reference_viscosity, dtype=np.float64)
    )
    rt_np = (
        np.full_like(t_np, reference_temperature, dtype=np.float64)
        if np.isscalar(reference_temperature)
        else np.asarray(reference_temperature, dtype=np.float64)
    )

    n = t_np.size

    # 5 c – allocate Taichi NDArrays
    t_ti = ti.ndarray(dtype=f64, shape=n)
    rv_ti = ti.ndarray(dtype=f64, shape=n)
    rt_ti = ti.ndarray(dtype=f64, shape=n)
    res_ti = ti.ndarray(dtype=f64, shape=n)

    t_ti.from_numpy(t_np)
    rv_ti.from_numpy(rv_np)
    rt_ti.from_numpy(rt_np)

    # 5 d – launch kernel
    kget_dynamic_viscosity(t_ti, rv_ti, rt_ti, res_ti)

    # 5 e – back to NumPy, squeeze scalar
    res_np = res_ti.to_numpy()
    return res_np.item() if res_np.size == 1 else res_np
