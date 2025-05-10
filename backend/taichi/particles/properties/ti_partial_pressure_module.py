"""Taichi-accelerated module for calculating the partial pressure delta
of a species in a gas over particle phase, considering the Kelvin effect.
"""

import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

@ti.func
def fget_partial_pressure_delta(
    partial_pressure_gas: ti.f64,
    partial_pressure_particle: ti.f64,
    kelvin_term: ti.f64
) -> ti.f64:
    """Elementwise Taichi function for partial pressure delta."""
    return partial_pressure_gas - partial_pressure_particle * kelvin_term

@ti.kernel
def kget_partial_pressure_delta(
    partial_pressure_gas: ti.types.ndarray(dtype=ti.f64, ndim=1),
    partial_pressure_particle: ti.types.ndarray(dtype=ti.f64, ndim=1),
    kelvin_term: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """Vectorized Taichi kernel for partial pressure delta."""
    for i in range(result.shape[0]):
        result[i] = fget_partial_pressure_delta(
            partial_pressure_gas[i],
            partial_pressure_particle[i],
            kelvin_term[i]
        )

@register("get_partial_pressure_delta", backend="taichi")
def ti_get_partial_pressure_delta(
    partial_pressure_gas,
    partial_pressure_particle,
    kelvin_term
):
    """Taichi-accelerated wrapper for partial pressure delta."""
    # Type guard
    if not (
        isinstance(partial_pressure_gas, np.ndarray)
        and isinstance(partial_pressure_particle, np.ndarray)
        and isinstance(kelvin_term, np.ndarray)
    ):
        raise TypeError("Taichi backend expects NumPy arrays for all inputs.")

    # Ensure 1-D NumPy arrays
    ppg = np.atleast_1d(partial_pressure_gas)
    ppp = np.atleast_1d(partial_pressure_particle)
    kt = np.atleast_1d(kelvin_term)
    n = ppg.size

    # Allocate Taichi NDArray buffers
    ppg_ti = ti.ndarray(dtype=ti.f64, shape=n)
    ppp_ti = ti.ndarray(dtype=ti.f64, shape=n)
    kt_ti = ti.ndarray(dtype=ti.f64, shape=n)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n)
    ppg_ti.from_numpy(ppg)
    ppp_ti.from_numpy(ppp)
    kt_ti.from_numpy(kt)

    # Launch the kernel
    kget_partial_pressure_delta(ppg_ti, ppp_ti, kt_ti, result_ti)

    # Convert result back to NumPy and unwrap if it is a single value
    result_np = result_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np
