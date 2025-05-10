"""Taichi implementation of vapor pressure routines."""
import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

_MMHG_TO_PA = 133.32238741499998

# ── Antoine equation ──────────────────────────────────────────────
@ti.func
def fget_antoine_vapor_pressure(
    a: ti.f64, b: ti.f64, c: ti.f64, temperature: ti.f64
) -> ti.f64:
    """Element-wise Antoine vapor pressure (all ti.f64)."""
    vapor_pressure_log = a - (b / (temperature - c))
    vapor_pressure = ti.pow(10.0, vapor_pressure_log)
    return vapor_pressure * _MMHG_TO_PA

@ti.kernel
def kget_antoine_vapor_pressure(
    a: ti.types.ndarray(dtype=ti.f64, ndim=1),
    b: ti.types.ndarray(dtype=ti.f64, ndim=1),
    c: ti.types.ndarray(dtype=ti.f64, ndim=1),
    temperature: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(result.shape[0]):
        result[i] = fget_antoine_vapor_pressure(
            a[i], b[i], c[i], temperature[i]
        )

@register("get_antoine_vapor_pressure", backend="taichi")
def ti_get_antoine_vapor_pressure(a, b, c, temperature):
    """Taichi backend wrapper for Antoine vapor pressure."""
    a_np, b_np, c_np, t_np = np.broadcast_arrays(
        np.atleast_1d(a).astype(np.float64),
        np.atleast_1d(b).astype(np.float64),
        np.atleast_1d(c).astype(np.float64),
        np.atleast_1d(temperature).astype(np.float64),
    )
    n = a_np.size
    a_ti = ti.ndarray(dtype=ti.f64, shape=n)
    b_ti = ti.ndarray(dtype=ti.f64, shape=n)
    c_ti = ti.ndarray(dtype=ti.f64, shape=n)
    t_ti = ti.ndarray(dtype=ti.f64, shape=n)
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)
    a_ti.from_numpy(a_np.ravel())
    b_ti.from_numpy(b_np.ravel())
    c_ti.from_numpy(c_np.ravel())
    t_ti.from_numpy(t_np.ravel())
    kget_antoine_vapor_pressure(a_ti, b_ti, c_ti, t_ti, res_ti)
    result_np = res_ti.to_numpy().reshape(a_np.shape)
    return result_np.item() if result_np.size == 1 else result_np

# ── Clausius-Clapeyron equation ────────────────────────────────────
@ti.func
def fget_clausius_clapeyron_vapor_pressure(
    latent_heat: ti.f64,
    temperature_initial: ti.f64,
    pressure_initial: ti.f64,
    temperature: ti.f64,
    gas_constant: ti.f64,
) -> ti.f64:
    """Element-wise Clausius-Clapeyron vapor pressure (all ti.f64)."""
    return pressure_initial * ti.exp(
        (latent_heat / gas_constant)
        * (1.0 / temperature_initial - 1.0 / temperature)
    )

@ti.kernel
def kget_clausius_clapeyron_vapor_pressure(
    latent_heat: ti.types.ndarray(dtype=ti.f64, ndim=1),
    temperature_initial: ti.types.ndarray(dtype=ti.f64, ndim=1),
    pressure_initial: ti.types.ndarray(dtype=ti.f64, ndim=1),
    temperature: ti.types.ndarray(dtype=ti.f64, ndim=1),
    gas_constant: ti.f64,
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(result.shape[0]):
        result[i] = fget_clausius_clapeyron_vapor_pressure(
            latent_heat[i],
            temperature_initial[i],
            pressure_initial[i],
            temperature[i],
            gas_constant,
        )

@register("get_clausius_clapeyron_vapor_pressure", backend="taichi")
def ti_get_clausius_clapeyron_vapor_pressure(
    latent_heat,
    temperature_initial,
    pressure_initial,
    temperature,
    gas_constant=8.31446261815324,
):
    """Taichi backend wrapper for Clausius-Clapeyron vapor pressure."""
    lh_np, ti_np, pi_np, t_np = np.broadcast_arrays(
        np.atleast_1d(latent_heat).astype(np.float64),
        np.atleast_1d(temperature_initial).astype(np.float64),
        np.atleast_1d(pressure_initial).astype(np.float64),
        np.atleast_1d(temperature).astype(np.float64),
    )
    n = lh_np.size
    lh_ti = ti.ndarray(dtype=ti.f64, shape=n)
    ti_ti = ti.ndarray(dtype=ti.f64, shape=n)
    pi_ti = ti.ndarray(dtype=ti.f64, shape=n)
    t_ti = ti.ndarray(dtype=ti.f64, shape=n)
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)
    lh_ti.from_numpy(lh_np.ravel())
    ti_ti.from_numpy(ti_np.ravel())
    pi_ti.from_numpy(pi_np.ravel())
    t_ti.from_numpy(t_np.ravel())
    kget_clausius_clapeyron_vapor_pressure(
        lh_ti, ti_ti, pi_ti, t_ti, float(gas_constant), res_ti
    )
    result_np = res_ti.to_numpy().reshape(lh_np.shape)
    return result_np.item() if result_np.size == 1 else result_np

# ── Buck equation ────────────────────────────────────────────────────
@ti.func
def fget_buck_vapor_pressure(
    temperature: ti.f64,
) -> ti.f64:
    """Element-wise Buck vapor pressure (all ti.f64)."""
    temp_c = temperature - 273.15
    vapor_pressure = 0.0
    if temp_c < 0.0:
        vapor_pressure = (
            6.1115 * ti.exp((23.036 - temp_c / 333.7) * temp_c / (279.82 + temp_c)) * 100.0
        )
    else:
        vapor_pressure = (
            6.1121 * ti.exp((18.678 - temp_c / 234.5) * temp_c / (257.14 + temp_c)) * 100.0
        )
    return vapor_pressure

@ti.kernel
def kget_buck_vapor_pressure(
    temperature: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(result.shape[0]):
        result[i] = fget_buck_vapor_pressure(temperature[i])

@register("get_buck_vapor_pressure", backend="taichi")
def ti_get_buck_vapor_pressure(temperature):
    """Taichi backend wrapper for Buck vapor pressure."""
    t_np = np.atleast_1d(temperature).astype(np.float64)
    n = t_np.size
    t_ti = ti.ndarray(dtype=ti.f64, shape=n)
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)
    t_ti.from_numpy(t_np.ravel())
    kget_buck_vapor_pressure(t_ti, res_ti)
    result_np = res_ti.to_numpy().reshape(t_np.shape)
    return result_np.item() if result_np.size == 1 else result_np
