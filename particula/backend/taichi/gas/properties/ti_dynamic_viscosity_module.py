"""
Taichi implementation of Sutherland’s dynamic-viscosity formula.

This module provides Taichi-accelerated routines for computing the dynamic
viscosity of gases as a function of temperature using Sutherland’s formula:

    μ = μ₀ (T/T₀)³ᐟ² × (T₀ + C)/(T + C)

where:
    - μ  : dynamic viscosity at temperature T
    - μ₀ : reference dynamic viscosity at temperature T₀
    - T  : temperature [K]
    - T₀ : reference temperature [K]
    - C  : Sutherland’s constant [K]

References:
    - "Sutherland's formula," Wikipedia.
      https://en.wikipedia.org/wiki/Sutherland%27s_formula
"""
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
    temperature: ti.f64,
    reference_viscosity: ti.f64,
    reference_temperature: ti.f64,
) -> ti.f64:
    """
    Compute the dynamic viscosity of a gas using Sutherland’s formula.

    Arguments:
        - temperature : Gas temperature [K].
        - reference_viscosity : Reference dynamic viscosity μ₀ [Pa·s].
        - reference_temperature : Reference temperature T₀ [K].

    Returns:
        - dynamic_viscosity : Dynamic viscosity μ at the given temperature [Pa·s].

    Examples:
        ```py title="Example"
        mu = fget_dynamic_viscosity(300.0, 1.8e-5, 273.15)
        # Output: μ ≈ 1.85e-5
        ```

    References:
        - "Sutherland's formula," Wikipedia.
          https://en.wikipedia.org/wiki/Sutherland%27s_formula
    """
    return (
        reference_viscosity
        * (temperature / reference_temperature) ** 1.5
        * (reference_temperature + SUTHERLAND_CONSTANT)
        / (temperature + SUTHERLAND_CONSTANT)
    )

# ── 4. vectorised kernel ──────────────────────────────────────────────
@ti.kernel
def kget_dynamic_viscosity(                     # 1-D only
    temperature: ti.types.ndarray(dtype=ti.f64, ndim=1),
    reference_viscosity: ti.types.ndarray(dtype=ti.f64, ndim=1),
    reference_temperature: ti.types.ndarray(dtype=ti.f64, ndim=1),
    dynamic_viscosity: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """
    Vectorized Taichi kernel for Sutherland’s formula on 1-D arrays.

    Applies the element-wise dynamic viscosity calculation to each entry
    in the input arrays.

    Arguments:
        - temperature : 1-D ndarray of temperatures [K].
        - reference_viscosity : 1-D ndarray of reference viscosities [Pa·s].
        - reference_temperature : 1-D ndarray of reference temperatures [K].
        - dynamic_viscosity : 1-D ndarray to store output viscosities [Pa·s].

    Returns:
        - None. Results are written in-place to dynamic_viscosity.
    """
    for i in range(dynamic_viscosity.shape[0]):
        dynamic_viscosity[i] = fget_dynamic_viscosity(
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
    """
    NumPy/Taichi wrapper for Sutherland’s dynamic viscosity formula.

    This function is registered as the Taichi backend for
    "get_dynamic_viscosity". It accepts scalar or array-like inputs,
    dispatches to Taichi for fast computation, and returns the result
    as a scalar or ndarray.

    Arguments:
        - temperature : Scalar or array-like of temperatures [K].
        - reference_viscosity : Scalar or array-like of reference
          viscosities μ₀ [Pa·s]. Default: REF_VISCOSITY_AIR_STP.
        - reference_temperature : Scalar or array-like of reference
          temperatures T₀ [K]. Default: REF_TEMPERATURE_STP.

    Returns:
        - dynamic_viscosity : Scalar or ndarray of dynamic viscosities
          [Pa·s], matching the input shape.

    Examples:
        ```py title="Example Usage"
        from particula.backend.dispatch_register import get_dynamic_viscosity
        mu = get_dynamic_viscosity(
            [300.0, 350.0],
            reference_viscosity=1.8e-5,
            reference_temperature=273.15,
            backend="taichi"
        )
        # Output: array([1.85e-5, 2.01e-5])
        ```

    References:
        - "Sutherland's formula," Wikipedia.
          https://en.wikipedia.org/wiki/Sutherland%27s_formula
    """
    # 5 a – type / shape guards
    temperature_array = np.atleast_1d(temperature).astype(np.float64)

    reference_viscosity_array = (
        np.full_like(temperature_array, reference_viscosity, dtype=np.float64)
        if np.isscalar(reference_viscosity)
        else np.asarray(reference_viscosity, dtype=np.float64)
    )
    reference_temperature_array = (
        np.full_like(
            temperature_array,
            reference_temperature,
            dtype=np.float64,
        )
        if np.isscalar(reference_temperature)
        else np.asarray(reference_temperature, dtype=np.float64)
    )

    # make sure all inputs share the same shape
    (
        temperature_array,
        reference_viscosity_array,
        reference_temperature_array,
    ) = np.broadcast_arrays(
        temperature_array,
        reference_viscosity_array,
        reference_temperature_array,
    )
    n_elements = temperature_array.size

    # 5 c – allocate Taichi NDArrays
    temperature_field = ti.ndarray(dtype=ti.f64, shape=n_elements)
    reference_viscosity_field = ti.ndarray(dtype=ti.f64, shape=n_elements)
    reference_temperature_field = ti.ndarray(dtype=ti.f64, shape=n_elements)
    dynamic_viscosity_field = ti.ndarray(dtype=ti.f64, shape=n_elements)

    temperature_field.from_numpy(temperature_array)
    reference_viscosity_field.from_numpy(reference_viscosity_array)
    reference_temperature_field.from_numpy(reference_temperature_array)

    # 5 d – launch kernel
    kget_dynamic_viscosity(
        temperature_field,
        reference_viscosity_field,
        reference_temperature_field,
        dynamic_viscosity_field,
    )

    # 5 e – back to NumPy, squeeze scalar
    dynamic_viscosity_array = dynamic_viscosity_field.to_numpy()
    return (
        dynamic_viscosity_array.item()
        if dynamic_viscosity_array.size == 1
        else dynamic_viscosity_array
    )
