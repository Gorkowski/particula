"""
Taichi implementation of concentration-from-pressure for ideal gases.

This module provides Taichi-accelerated functions and kernels to compute
the molar or mass concentration of a gas from its partial pressure, molar
mass, and temperature, using the ideal-gas law:

    C = (P × M) / (R × T)

where:
    - C is the concentration,
    - P is the partial pressure,
    - M is the molar mass,
    - R is the gas constant,
    - T is the temperature.

Examples:
    ```py
    from particula.backend.taichi.gas.properties import (
        ti_concentration_from_pressure_module as ti_cfp,
    )
    c = ti_cfp.ti_get_concentration_from_pressure(101325, 0.02897, 300)
    # Output: 1.176... (kg/m³)
    ```

References:
    - "Ideal gas law," Wikipedia.
      https://en.wikipedia.org/wiki/Ideal_gas_law
"""
import taichi as ti
import numpy as np

from particula.util.constants import GAS_CONSTANT

_GAS_CONSTANT = float(GAS_CONSTANT)

@ti.func
def fget_concentration_from_pressure(
    partial_pressure: ti.f64,
    molar_mass: ti.f64,
    temperature: ti.f64,
) -> ti.f64:
    """
    Compute gas concentration from pressure, molar mass, and temperature.

    Uses the ideal-gas law in the form:
        C = (P × M) / (R × T)

    Arguments:
        - partial_pressure : Partial pressure of the gas [Pa].
        - molar_mass : Molar mass of the gas [kg/mol].
        - temperature : Temperature [K].

    Returns:
        - Concentration [kg/m³] as a float.

    References:
        - "Ideal gas law," Wikipedia.
          https://en.wikipedia.org/wiki/Ideal_gas_law
    """
    return (partial_pressure * molar_mass) / (_GAS_CONSTANT * temperature)

@ti.kernel
def kget_concentration_from_pressure(
    partial_pressure: ti.types.ndarray(dtype=ti.f64, ndim=1),
    molar_mass: ti.types.ndarray(dtype=ti.f64, ndim=1),
    temperature: ti.types.ndarray(dtype=ti.f64, ndim=1),
    concentration: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """
    Taichi kernel: vectorized concentration-from-pressure for arrays.

    Computes the concentration for each element in the input arrays using
    the ideal-gas law. All arrays must be 1D and of the same length.

    Arguments:
        - partial_pressure : 1D ndarray of partial pressures [Pa].
        - molar_mass : 1D ndarray of molar masses [kg/mol].
        - temperature : 1D ndarray of temperatures [K].
        - concentration : 1D output ndarray for concentrations [kg/m³].

    Returns:
        - None (results written in-place to `concentration` buffer).
    """
    for i in range(concentration.shape[0]):
        concentration[i] = fget_concentration_from_pressure(
            partial_pressure[i], molar_mass[i], temperature[i]
        )

def ti_get_concentration_from_pressure(partial_pressure, molar_mass, temperature):
    """
    Taichi wrapper for get_concentration_from_pressure (vectorized).

    This function type-guards and broadcasts the input arrays, then
    computes the gas concentration using Taichi kernels. The result is
    returned as a NumPy array or scalar, matching the broadcasted shape.

    Arguments:
        - partial_pressure : Scalar or array of partial pressures [Pa].
        - molar_mass : Scalar or array of molar masses [kg/mol].
        - temperature : Scalar or array of temperatures [K].

    Returns:
        - Concentration(s) [kg/m³] as a NumPy array or scalar.

    Examples:
        ```py title="Example"
        ti_get_concentration_from_pressure(101325, 0.02897, 300)
        # Output: 1.176... (kg/m³)
        ```

        ```py title="Example with arrays"
        ti_get_concentration_from_pressure(
            [101325, 202650], [0.02897, 0.02897], [300, 400]
        )
        # Output: array([1.176..., 1.058...])
        ```

    References:
        - "Ideal gas law," Wikipedia.
          https://en.wikipedia.org/wiki/Ideal_gas_law
    """
    # 5 a – type guard  (explicit float64)
    partial_pressure = np.asarray(partial_pressure, dtype=np.float64)
    molar_mass = np.asarray(molar_mass, dtype=np.float64)
    temperature = np.asarray(temperature, dtype=np.float64)

    # 5 b – broadcast to common shape, then flatten
    (
        partial_pressure_broadcast,
        molar_mass_broadcast,
        temperature_broadcast,
    ) = np.broadcast_arrays(
        partial_pressure,
        molar_mass,
        temperature,
    )
    (
        flat_partial_pressure,
        flat_molar_mass,
        flat_temperature,
    ) = map(
        np.ravel,
        (
            partial_pressure_broadcast,
            molar_mass_broadcast,
            temperature_broadcast,
        ),
    )
    n_elements = flat_partial_pressure.size

    # 5 c – allocate buffers with explicit names
    partial_pressure_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    molar_mass_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    temperature_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    concentration_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    partial_pressure_ti.from_numpy(flat_partial_pressure)
    molar_mass_ti.from_numpy(flat_molar_mass)
    temperature_ti.from_numpy(flat_temperature)

    # 5 d – launch kernel with explicit buffer names
    kget_concentration_from_pressure(
        partial_pressure_ti,
        molar_mass_ti,
        temperature_ti,
        concentration_ti,
    )

    # 5 e – return NumPy/scalar, restoring broadcasted shape
    concentration_np = concentration_ti.to_numpy().reshape(
        partial_pressure_broadcast.shape
    )
    return concentration_np.item() if concentration_np.size == 1 else concentration_np
