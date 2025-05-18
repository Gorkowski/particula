"""Taichi implementation of vapor pressure routines."""
import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

_MMHG_TO_PA = 133.32238741499998

# ── Antoine equation ──────────────────────────────────────────────
@ti.func
def fget_antoine_vapor_pressure(
    constant_a: ti.f64,
    constant_b: ti.f64,
    constant_c: ti.f64,
    temperature: ti.f64,
) -> ti.f64:
    """Element-wise Antoine vapor pressure (all ti.f64)."""
    vapor_pressure_log = constant_a - (constant_b / (temperature - constant_c))
    vapor_pressure = ti.pow(10.0, vapor_pressure_log)
    return vapor_pressure * _MMHG_TO_PA

@ti.kernel
def kget_antoine_vapor_pressure(
    constant_a: ti.types.ndarray(dtype=ti.f64, ndim=1),
    constant_b: ti.types.ndarray(dtype=ti.f64, ndim=1),
    constant_c: ti.types.ndarray(dtype=ti.f64, ndim=1),
    temperature: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(result.shape[0]):
        result[i] = fget_antoine_vapor_pressure(
            constant_a[i], constant_b[i], constant_c[i], temperature[i]
        )

@register("get_antoine_vapor_pressure", backend="taichi")
def ti_get_antoine_vapor_pressure(constant_a, constant_b, constant_c, temperature):
    """Taichi backend wrapper for Antoine vapor pressure."""
    constant_a_np, constant_b_np, constant_c_np, temperature_np = np.broadcast_arrays(
        np.atleast_1d(constant_a).astype(np.float64),
        np.atleast_1d(constant_b).astype(np.float64),
        np.atleast_1d(constant_c).astype(np.float64),
        np.atleast_1d(temperature).astype(np.float64),
    )
    n_elements = constant_a_np.size
    constant_a_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    constant_b_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    constant_c_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    temperature_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    constant_a_ti.from_numpy(constant_a_np.ravel())
    constant_b_ti.from_numpy(constant_b_np.ravel())
    constant_c_ti.from_numpy(constant_c_np.ravel())
    temperature_ti.from_numpy(temperature_np.ravel())
    kget_antoine_vapor_pressure(
        constant_a_ti,
        constant_b_ti,
        constant_c_ti,
        temperature_ti,
        result_ti,
    )
    result_np = result_ti.to_numpy().reshape(constant_a_np.shape)
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
    latent_heat_np, temperature_initial_np, pressure_initial_np, temperature_np = np.broadcast_arrays(
        np.atleast_1d(latent_heat).astype(np.float64),
        np.atleast_1d(temperature_initial).astype(np.float64),
        np.atleast_1d(pressure_initial).astype(np.float64),
        np.atleast_1d(temperature).astype(np.float64),
    )
    n_elements = latent_heat_np.size
    latent_heat_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    temperature_initial_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    pressure_initial_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    temperature_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    latent_heat_ti.from_numpy(latent_heat_np.ravel())
    temperature_initial_ti.from_numpy(temperature_initial_np.ravel())
    pressure_initial_ti.from_numpy(pressure_initial_np.ravel())
    temperature_ti.from_numpy(temperature_np.ravel())
    kget_clausius_clapeyron_vapor_pressure(
        latent_heat_ti, temperature_initial_ti, pressure_initial_ti, temperature_ti, float(gas_constant), result_ti
    )
    result_np = result_ti.to_numpy().reshape(latent_heat_np.shape)
    return result_np.item() if result_np.size == 1 else result_np

# ── Buck equation ────────────────────────────────────────────────────
@ti.func
def fget_buck_vapor_pressure(
    temperature: ti.f64,
) -> ti.f64:
    """Element-wise Buck vapor pressure (all ti.f64)."""
    temperature_celsius = temperature - 273.15
    vapor_pressure = 0.0
    if temperature_celsius < 0.0:
        vapor_pressure = (
            6.1115 * ti.exp((23.036 - temperature_celsius / 333.7) * temperature_celsius / (279.82 + temperature_celsius)) * 100.0
        )
    else:
        vapor_pressure = (
            6.1121 * ti.exp((18.678 - temperature_celsius / 234.5) * temperature_celsius / (257.14 + temperature_celsius)) * 100.0
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
    temperature_np = np.atleast_1d(temperature).astype(np.float64)
    n_elements = temperature_np.size
    temperature_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    temperature_ti.from_numpy(temperature_np.ravel())
    kget_buck_vapor_pressure(temperature_ti, result_ti)
    result_np = result_ti.to_numpy().reshape(temperature_np.shape)
    return result_np.item() if result_np.size == 1 else result_np
