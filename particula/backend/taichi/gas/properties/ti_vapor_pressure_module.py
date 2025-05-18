"""
Taichi implementation of vapor pressure routines.

This module provides Taichi-accelerated routines for calculating vapor
pressure using the Antoine, Clausius-Clapeyron, and Buck equations.
Taichi is used to enable efficient, parallelized computation on CPUs
and GPUs, making these routines suitable for large-scale or
performance-critical scientific simulations.

Examples:
    ```py
    import particula.backend.taichi.gas.properties.ti_vapor_pressure_module as vap
    vap.ti_get_antoine_vapor_pressure(8.07, 1730.63, 233.426, 373.15)
    ```

References:
    - "Antoine equation," Wikipedia,
      https://en.wikipedia.org/wiki/Antoine_equation
    - "Clausius–Clapeyron relation," Wikipedia,
      https://en.wikipedia.org/wiki/Clausius%E2%80%93Clapeyron_relation
    - "Buck equation (vapor pressure)," Wikipedia,
      https://en.wikipedia.org/wiki/Vapour_pressure_of_water#Empirical_formulas
"""
import taichi as ti
import numpy as np

MMHG_TO_PA = 133.32238741499998

# ── Antoine equation ──────────────────────────────────────────────
@ti.func
def fget_antoine_vapor_pressure(
    constant_a: ti.f64,
    constant_b: ti.f64,
    constant_c: ti.f64,
    temperature: ti.f64,
) -> ti.f64:
    """
    Element-wise Antoine vapor pressure (all ti.f64).

    Computes the vapor pressure using the Antoine equation for a single
    set of parameters and temperature.

    Equation:
        log₁₀(P) = A - B / (T - C)
        P = 10^(A - B / (T - C)) × MMHG_TO_PA Pa

    Arguments:
        - constant_a : Antoine constant A (unitless)
        - constant_b : Antoine constant B (unitless)
        - constant_c : Antoine constant C (unitless)
        - temperature : Temperature in Kelvin (ti.f64)

    Returns:
        - Vapor pressure in Pascals (ti.f64)

    Examples:
        ```py
        fget_antoine_vapor_pressure(8.07, 1730.63, 233.426, 373.15)
        ```

    References:
        - "Antoine equation," Wikipedia,
          https://en.wikipedia.org/wiki/Antoine_equation
    """
    vapor_pressure_log = constant_a - (constant_b / (temperature - constant_c))
    vapor_pressure = ti.pow(10.0, vapor_pressure_log)
    return vapor_pressure * MMHG_TO_PA

@ti.kernel
def kget_antoine_vapor_pressure(
    constant_a: ti.types.ndarray(dtype=ti.f64, ndim=1),
    constant_b: ti.types.ndarray(dtype=ti.f64, ndim=1),
    constant_c: ti.types.ndarray(dtype=ti.f64, ndim=1),
    temperature: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """
    Kernel for element-wise Antoine vapor pressure (internal use only).

    Applies the Antoine equation to each element of the input arrays.
    All arrays must be 1-D, dtype=ti.f64, and broadcasted to the same
    shape.

    Arguments:
        - constant_a : 1-D ti.types.ndarray of Antoine A constants
        - constant_b : 1-D ti.types.ndarray of Antoine B constants
        - constant_c : 1-D ti.types.ndarray of Antoine C constants
        - temperature : 1-D ti.types.ndarray of temperatures [K]
        - result : 1-D ti.types.ndarray for output [Pa]

    Returns:
        - None (results written in-place to result array)

    Examples:
        ```py
        kget_antoine_vapor_pressure(a, b, c, t, out)
        ```

    References:
        - "Antoine equation," Wikipedia,
          https://en.wikipedia.org/wiki/Antoine_equation
    """
    for i in range(result.shape[0]):
        result[i] = fget_antoine_vapor_pressure(
            constant_a[i], constant_b[i], constant_c[i], temperature[i]
        )

def ti_get_antoine_vapor_pressure(constant_a, constant_b, constant_c, temperature):
    """
    Taichi backend wrapper for Antoine vapor pressure.

    Broadcasts all input arrays to a common shape, converts to Taichi
    ndarrays, and computes vapor pressure using the Antoine equation.

    Arguments:
        - constant_a : Scalar or array-like of Antoine A constants
        - constant_b : Scalar or array-like of Antoine B constants
        - constant_c : Scalar or array-like of Antoine C constants
        - temperature : Scalar or array-like of temperatures [K]

    Returns:
        - Vapor pressure(s) in Pascals (scalar or ndarray)

    Examples:
        ```py
        ti_get_antoine_vapor_pressure(8.07, 1730.63, 233.426, 373.15)
        ```

    References:
        - "Antoine equation," Wikipedia,
          https://en.wikipedia.org/wiki/Antoine_equation
    """
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
    """
    Element-wise Clausius-Clapeyron vapor pressure (all ti.f64).

    Computes vapor pressure at a given temperature using the
    Clausius-Clapeyron relation.

    Equation:
        P = P₀ × exp[(L / R) × (1/T₀ - 1/T)]
        where:
            - P : vapor pressure at T
            - P₀ : reference pressure at T₀
            - L : latent heat [J/mol]
            - R : gas constant [J/(mol·K)]
            - T₀ : reference temperature [K]
            - T : temperature [K]

    Arguments:
        - latent_heat : Latent heat of vaporization [J/mol]
        - temperature_initial : Reference temperature [K]
        - pressure_initial : Reference pressure [Pa]
        - temperature : Target temperature [K]
        - gas_constant : Gas constant [J/(mol·K)]

    Returns:
        - Vapor pressure in Pascals (ti.f64)

    Examples:
        ```py
        fget_clausius_clapeyron_vapor_pressure(40000, 373.15, 101325, 350, 8.314)
        ```

    References:
        - "Clausius–Clapeyron relation," Wikipedia,
          https://en.wikipedia.org/wiki/Clausius%E2%80%93Clapeyron_relation
    """
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
    """
    Kernel for element-wise Clausius-Clapeyron vapor pressure (internal use only).

    Applies the Clausius-Clapeyron equation to each element of the input
    arrays. All arrays must be 1-D, dtype=ti.f64, and broadcasted to the
    same shape.

    Arguments:
        - latent_heat : 1-D ti.types.ndarray of latent heats [J/mol]
        - temperature_initial : 1-D ti.types.ndarray of reference T [K]
        - pressure_initial : 1-D ti.types.ndarray of reference P [Pa]
        - temperature : 1-D ti.types.ndarray of target T [K]
        - gas_constant : Scalar gas constant [J/(mol·K)]
        - result : 1-D ti.types.ndarray for output [Pa]

    Returns:
        - None (results written in-place to result array)

    Examples:
        ```py
        kget_clausius_clapeyron_vapor_pressure(lh, t0, p0, t, R, out)
        ```

    References:
        - "Clausius–Clapeyron relation," Wikipedia,
          https://en.wikipedia.org/wiki/Clausius%E2%80%93Clapeyron_relation
    """
    for i in range(result.shape[0]):
        result[i] = fget_clausius_clapeyron_vapor_pressure(
            latent_heat[i],
            temperature_initial[i],
            pressure_initial[i],
            temperature[i],
            gas_constant,
        )

def ti_get_clausius_clapeyron_vapor_pressure(
    latent_heat,
    temperature_initial,
    pressure_initial,
    temperature,
    gas_constant=8.31446261815324,
):
    """
    Taichi backend wrapper for Clausius-Clapeyron vapor pressure.

    Broadcasts all input arrays to a common shape, converts to Taichi
    ndarrays, and computes vapor pressure using the Clausius-Clapeyron
    equation.

    Arguments:
        - latent_heat : Scalar or array-like of latent heats [J/mol]
        - temperature_initial : Scalar or array-like of reference T [K]
        - pressure_initial : Scalar or array-like of reference P [Pa]
        - temperature : Scalar or array-like of target T [K]
        - gas_constant : Scalar gas constant [J/(mol·K)]

    Returns:
        - Vapor pressure(s) in Pascals (scalar or ndarray)

    Examples:
        ```py
        ti_get_clausius_clapeyron_vapor_pressure(40000, 373.15, 101325, 350)
        ```

    References:
        - "Clausius–Clapeyron relation," Wikipedia,
          https://en.wikipedia.org/wiki/Clausius%E2%80%93Clapeyron_relation
    """
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
    """
    Element-wise Buck vapor pressure (all ti.f64).

    Computes the vapor pressure of water using the Buck equation for a
    given temperature.

    Equation:
        For T < 0°C:
            P = 6.1115 × exp[(23.036 - T/333.7) × T / (279.82 + T)] × 100
        For T ≥ 0°C:
            P = 6.1121 × exp[(18.678 - T/234.5) × T / (257.14 + T)] × 100
        where T is temperature in Celsius, P in Pascals.

    Arguments:
        - temperature : Temperature in Kelvin (ti.f64)

    Returns:
        - Vapor pressure in Pascals (ti.f64)

    Examples:
        ```py
        fget_buck_vapor_pressure(298.15)
        ```

    References:
        - "Buck equation (vapor pressure)," Wikipedia,
          https://en.wikipedia.org/wiki/Vapour_pressure_of_water#Empirical_formulas
    """
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
    """
    Kernel for element-wise Buck vapor pressure (internal use only).

    Applies the Buck equation to each element of the input temperature
    array. All arrays must be 1-D, dtype=ti.f64, and broadcasted to the
    same shape.

    Arguments:
        - temperature : 1-D ti.types.ndarray of temperatures [K]
        - result : 1-D ti.types.ndarray for output [Pa]

    Returns:
        - None (results written in-place to result array)

    Examples:
        ```py
        kget_buck_vapor_pressure(temp, out)
        ```

    References:
        - "Buck equation (vapor pressure)," Wikipedia,
          https://en.wikipedia.org/wiki/Vapour_pressure_of_water#Empirical_formulas
    """
    for i in range(result.shape[0]):
        result[i] = fget_buck_vapor_pressure(temperature[i])

def ti_get_buck_vapor_pressure(temperature):
    """
    Taichi backend wrapper for Buck vapor pressure.

    Broadcasts the input temperature array to a 1-D shape, converts to
    Taichi ndarray, and computes vapor pressure using the Buck equation.

    Arguments:
        - temperature : Scalar or array-like of temperatures [K]

    Returns:
        - Vapor pressure(s) in Pascals (scalar or ndarray)

    Examples:
        ```py
        ti_get_buck_vapor_pressure(298.15)
        ```

    References:
        - "Buck equation (vapor pressure)," Wikipedia,
          https://en.wikipedia.org/wiki/Vapour_pressure_of_water#Empirical_formulas
    """
    temperature_np = np.atleast_1d(temperature).astype(np.float64)
    n_elements = temperature_np.size
    temperature_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    temperature_ti.from_numpy(temperature_np.ravel())
    kget_buck_vapor_pressure(temperature_ti, result_ti)
    result_np = result_ti.to_numpy().reshape(temperature_np.shape)
    return result_np.item() if result_np.size == 1 else result_np
