"""
Taichi-accelerated dilution utilities

This module provides scalar (`fget_*`) and array (`kget_*`) helpers for
computing:
    • the volume-dilution coefficient ϕ = Q ÷ V [s⁻¹]  
    • the dilution rate                r = −ϕ × n [mol m⁻³ s⁻¹]

Wrappers (`ti_*`) accept either numeric scalars or 1-D NumPy arrays and
dispatch the calculations to the Taichi kernels.

Examples
--------
>>> ti_get_volume_dilution_coefficient(1.0, 0.1)
0.1
>>> ti_get_dilution_rate(0.1, 2.0)
-0.2

References
----------
Continuous stirred-tank reactor, Wikipedia.  *URL omitted for brevity*.
"""
import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

@ti.func
def fget_volume_dilution_coefficient(volume: ti.f64,
                                     input_flow_rate: ti.f64) -> ti.f64:
    """
    Calculate the volume-dilution coefficient.

    ϕ = Q ÷ V

    Arguments:
        - volume : System volume, V [m³].
        - input_flow_rate : Volumetric inflow rate, Q [m³ s⁻¹].

    Returns:
        - Dilution coefficient, ϕ [s⁻¹].
    """
    return input_flow_rate / volume

@ti.func
def fget_dilution_rate(coefficient: ti.f64,
                       concentration: ti.f64) -> ti.f64:
    """
    Calculate the dilution rate for a species.

    r = −ϕ × n

    Arguments:
        - coefficient : Dilution coefficient, ϕ [s⁻¹].
        - concentration : Species concentration, n [mol m⁻³].

    Returns:
        - Dilution rate, r [mol m⁻³ s⁻¹].
    """
    return -coefficient * concentration

@ti.kernel
def kget_volume_dilution_coefficient(
    volume: ti.types.ndarray(dtype=ti.f64, ndim=1),
    input_flow_rate: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """Kernel version of `fget_volume_dilution_coefficient` for arrays."""
    for i in range(result.shape[0]):
        result[i] = fget_volume_dilution_coefficient(
            volume[i], input_flow_rate[i])

@ti.kernel
def kget_dilution_rate(
    coefficient: ti.types.ndarray(dtype=ti.f64, ndim=1),
    concentration: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """Kernel version of `fget_dilution_rate` for arrays."""
    for i in range(result.shape[0]):
        result[i] = fget_dilution_rate(coefficient[i], concentration[i])

@register("get_volume_dilution_coefficient", backend="taichi")
def ti_get_volume_dilution_coefficient(volume, input_flow_rate):
    """
    Get the volume-dilution coefficient using the Taichi backend.

    Accepts scalars or 1-D NumPy arrays and returns a value/array with the
    same shape.

    Arguments:
        - volume : Volume(s), V [m³].
        - input_flow_rate : Inflow rate(s), Q [m³ s⁻¹].

    Returns:
        - Dilution coefficient(s), ϕ [s⁻¹].

    Examples
    --------
    >>> ti_get_volume_dilution_coefficient(1.0, 0.1)
    0.1
    """
    # 5 a – type guard (allow scalars or NumPy arrays)
    if not (np.isscalar(volume) or isinstance(volume, np.ndarray)):
        raise TypeError("Taichi backend expects numeric scalars or NumPy arrays.")
    if not (np.isscalar(input_flow_rate) or isinstance(input_flow_rate, np.ndarray)):
        raise TypeError("Taichi backend expects numeric scalars or NumPy arrays.")

    # 5 b – ensure 1-D NumPy arrays
    volume_array = np.atleast_1d(volume)
    input_flow_rate_array = np.atleast_1d(input_flow_rate)
    n_elements = volume_array.size

    volume_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    input_flow_rate_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    volume_ti.from_numpy(volume_array)
    input_flow_rate_ti.from_numpy(input_flow_rate_array)

    kget_volume_dilution_coefficient(
        volume_ti, input_flow_rate_ti, result_ti
    )

    result_array = result_ti.to_numpy()
    return result_array.item() if result_array.size == 1 else result_array

@register("get_dilution_rate", backend="taichi")
def ti_get_dilution_rate(coefficient, concentration):
    """
    Get the dilution rate using the Taichi backend.

    Accepts scalars or 1-D NumPy arrays and returns a value/array with the
    same shape.

    Arguments:
        - coefficient : Dilution coefficient(s), ϕ [s⁻¹].
        - concentration : Species concentration(s), n [mol m⁻³].

    Returns:
        - Dilution rate(s), r [mol m⁻³ s⁻¹].

    Examples
    --------
    >>> ti_get_dilution_rate(0.1, 2.0)
    -0.2
    """
    # 5 a – type guard (allow scalars or NumPy arrays)
    if not (np.isscalar(coefficient) or isinstance(coefficient, np.ndarray)):
        raise TypeError("Taichi backend expects numeric scalars or NumPy arrays.")
    if not (np.isscalar(concentration) or isinstance(concentration, np.ndarray)):
        raise TypeError("Taichi backend expects numeric scalars or NumPy arrays.")

    # 5 b – ensure 1-D NumPy arrays
    coefficient_array = np.atleast_1d(coefficient)
    concentration_array = np.atleast_1d(concentration)
    n_elements = coefficient_array.size

    coefficient_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    concentration_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    coefficient_ti.from_numpy(coefficient_array)
    concentration_ti.from_numpy(concentration_array)

    kget_dilution_rate(coefficient_ti, concentration_ti, result_ti)

    result_array = result_ti.to_numpy()
    return result_array.item() if result_array.size == 1 else result_array
