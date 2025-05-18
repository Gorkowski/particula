"""
Taichi helpers for partial-pressure and saturation-ratio calculations.

This module provides Taichi-accelerated functions and kernels for computing
elementwise and vectorized partial pressure and saturation ratio for gases,
using the ideal gas law and related relations.

Equations:
    • Partial pressure:
        P = (C × R × T) ⁄ M
        where:
            P : partial pressure [Pa]
            C : concentration [mol/m³]
            R : gas constant [J/(mol·K)]
            T : temperature [K]
            M : molar mass [kg/mol]
    • Saturation ratio:
        S = P ⁄ P*
        where:
            S : saturation ratio [unitless]
            P : partial pressure [Pa]
            P* : pure vapor pressure [Pa]

Parameter Symbols:
    - C : concentration [mol/m³]
    - M : molar mass [kg/mol]
    - T : temperature [K]
    - P : partial pressure [Pa]
    - P* : pure vapor pressure [Pa]
    - S : saturation ratio [unitless]

Functions:
    - fget_partial_pressure
    - fget_saturation_ratio_from_pressure
    - kget_partial_pressure
    - kget_saturation_ratio_from_pressure
    - ti_get_partial_pressure
    - ti_get_saturation_ratio_from_pressure

Examples:
    >>> import particula.backend.taichi.gas.properties.ti_pressure_function_module as tpf
    >>> tpf.ti_get_partial_pressure(2.0, 0.029, 300.0)
    17197.241379310345
    >>> tpf.ti_get_partial_pressure([2.0, 3.0], 0.029, 300.0)
    array([17197.24137931, 25795.86206897])
    >>> tpf.ti_get_saturation_ratio_from_pressure(1000.0, 1200.0)
    0.8333333333333334
    >>> tpf.ti_get_saturation_ratio_from_pressure([1000.0, 1200.0], 1200.0)
    array([0.83333333, 1.        ])

References:
    - "Ideal gas law," Wikipedia,
      https://en.wikipedia.org/wiki/Ideal_gas_law
"""

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
    """
    Compute the element-wise partial pressure using the ideal gas law.

    Equation:
        P = (C × R × T) ⁄ M

    Arguments:
        - concentration : Molar concentration [mol/m³].
        - molar_mass : Molar mass [kg/mol].
        - temperature : Temperature [K].

    Returns:
        - Partial pressure [Pa].

    Examples:
        >>> fget_partial_pressure(2.0, 0.029, 300.0)
        17197.241379310345

    References:
        - "Ideal gas law," Wikipedia,
          https://en.wikipedia.org/wiki/Ideal_gas_law
    """
    return (concentration * _GAS_CONSTANT * temperature) / molar_mass

@ti.func
def fget_saturation_ratio_from_pressure(
    partial_pressure: ti.f64,
    pure_vapor_pressure: ti.f64,
) -> ti.f64:
    """
    Compute the element-wise saturation ratio from partial and pure vapor
    pressures.

    Equation:
        S = P ⁄ P*

    Arguments:
        - partial_pressure : Partial pressure [Pa].
        - pure_vapor_pressure : Pure vapor pressure [Pa].

    Returns:
        - Saturation ratio [unitless].

    Examples:
        >>> fget_saturation_ratio_from_pressure(1000.0, 1200.0)
        0.8333333333333334

    References:
        - "Saturation vapor pressure," Wikipedia,
          https://en.wikipedia.org/wiki/Vapour_pressure
    """
    return partial_pressure / pure_vapor_pressure

@ti.kernel
def kget_partial_pressure(
    concentration: ti.types.ndarray(dtype=ti.f64, ndim=1),
    molar_mass: ti.types.ndarray(dtype=ti.f64, ndim=1),
    temperature: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """
    Taichi kernel for vectorized partial pressure calculation.

    Computes partial pressure for each element and stores the result in the
    provided output array.

    Arguments:
        - concentration : 1D ndarray of molar concentration [mol/m³].
        - molar_mass : 1D ndarray of molar mass [kg/mol].
        - temperature : 1D ndarray of temperature [K].
        - result : 1D ndarray to store output partial pressure [Pa].

    Returns:
        None (results are written to the 'result' array).

    Examples:
        Used internally by ti_get_partial_pressure.

    References:
        - "Ideal gas law," Wikipedia,
          https://en.wikipedia.org/wiki/Ideal_gas_law
    """
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
    """
    Taichi kernel for vectorized saturation ratio calculation.

    Computes saturation ratio for each element and stores the result in the
    provided output array.

    Arguments:
        - partial_pressure : 1D ndarray of partial pressure [Pa].
        - pure_vapor_pressure : 1D ndarray of pure vapor pressure [Pa].
        - result : 1D ndarray to store output saturation ratio [unitless].

    Returns:
        None (results are written to the 'result' array).

    Examples:
        Used internally by ti_get_saturation_ratio_from_pressure.

    References:
        - "Saturation vapor pressure," Wikipedia,
          https://en.wikipedia.org/wiki/Vapour_pressure
    """
    for i in range(result.shape[0]):
        result[i] = fget_saturation_ratio_from_pressure(
            partial_pressure[i], pure_vapor_pressure[i]
        )

@register("get_partial_pressure", backend="taichi")
def ti_get_partial_pressure(concentration, molar_mass, temperature):
    """
    Vectorized Taichi wrapper for partial pressure calculation.

    Accepts scalars or arrays for concentration, molar mass, and temperature.
    Inputs are broadcast to a common shape, and the result preserves this
    shape. Returns a NumPy array or a scalar if the result is a single value.

    Arguments:
        - concentration : Scalar or array of molar concentration [mol/m³].
        - molar_mass : Scalar or array of molar mass [kg/mol].
        - temperature : Scalar or array of temperature [K].

    Returns:
        - NumPy array or scalar of partial pressure [Pa].

    Examples:
        >>> ti_get_partial_pressure(2.0, 0.029, 300.0)
        17197.241379310345
        >>> ti_get_partial_pressure([2.0, 3.0], 0.029, 300.0)
        array([17197.24137931, 25795.86206897])

    References:
        - "Ideal gas law," Wikipedia,
          https://en.wikipedia.org/wiki/Ideal_gas_law
    """
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
    """
    Vectorized Taichi wrapper for saturation ratio calculation.

    Accepts scalars or arrays for partial pressure and pure vapor pressure.
    Inputs are broadcast to a common shape, and the result preserves this
    shape. Returns a NumPy array or a scalar if the result is a single value.

    Arguments:
        - partial_pressure : Scalar or array of partial pressure [Pa].
        - pure_vapor_pressure : Scalar or array of pure vapor pressure [Pa].

    Returns:
        - NumPy array or scalar of saturation ratio [unitless].

    Examples:
        >>> ti_get_saturation_ratio_from_pressure(1000.0, 1200.0)
        0.8333333333333334
        >>> ti_get_saturation_ratio_from_pressure([1000.0, 1200.0], 1200.0)
        array([0.83333333, 1.        ])

    References:
        - "Saturation vapor pressure," Wikipedia,
          https://en.wikipedia.org/wiki/Vapour_pressure
    """
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
