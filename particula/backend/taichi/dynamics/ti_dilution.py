"""Taichi-accelerated dilution utilities."""
import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

@ti.func
def fget_volume_dilution_coefficient(volume: ti.f64,
                                     input_flow_rate: ti.f64) -> ti.f64:
    return input_flow_rate / volume

@ti.func
def fget_dilution_rate(coefficient: ti.f64,
                       concentration: ti.f64) -> ti.f64:
    return -coefficient * concentration

@ti.kernel
def kget_volume_dilution_coefficient(
    volume: ti.types.ndarray(dtype=ti.f64, ndim=1),
    input_flow_rate: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(result.shape[0]):
        result[i] = fget_volume_dilution_coefficient(
            volume[i], input_flow_rate[i])

@ti.kernel
def kget_dilution_rate(
    coefficient: ti.types.ndarray(dtype=ti.f64, ndim=1),
    concentration: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(result.shape[0]):
        result[i] = fget_dilution_rate(coefficient[i], concentration[i])

@register("get_volume_dilution_coefficient", backend="taichi")
def ti_get_volume_dilution_coefficient(volume, input_flow_rate):
    """Taichi version of get_volume_dilution_coefficient."""
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
    """Taichi version of get_dilution_rate."""
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
