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
    if not (isinstance(volume, np.ndarray) and isinstance(input_flow_rate, np.ndarray)):
        raise TypeError("Taichi backend expects NumPy arrays for both inputs.")

    v, q = np.atleast_1d(volume), np.atleast_1d(input_flow_rate)
    n = v.size

    v_ti = ti.ndarray(dtype=ti.f64, shape=n)
    q_ti = ti.ndarray(dtype=ti.f64, shape=n)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n)
    v_ti.from_numpy(v)
    q_ti.from_numpy(q)

    kget_volume_dilution_coefficient(v_ti, q_ti, result_ti)

    result_np = result_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np

@register("get_dilution_rate", backend="taichi")
def ti_get_dilution_rate(coefficient, concentration):
    """Taichi version of get_dilution_rate."""
    if not (isinstance(coefficient, np.ndarray) and isinstance(concentration, np.ndarray)):
        raise TypeError("Taichi backend expects NumPy arrays for both inputs.")

    a, c = np.atleast_1d(coefficient), np.atleast_1d(concentration)
    n = a.size

    a_ti = ti.ndarray(dtype=ti.f64, shape=n)
    c_ti = ti.ndarray(dtype=ti.f64, shape=n)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n)
    a_ti.from_numpy(a)
    c_ti.from_numpy(c)

    kget_dilution_rate(a_ti, c_ti, result_ti)

    result_np = result_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np
