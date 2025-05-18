import numpy as np
import numpy.testing as npt
import taichi as ti

from particula.dynamics.dilution import (
    get_volume_dilution_coefficient,
    get_dilution_rate,
)
from particula.backend.taichi.dynamics.ti_dilution import (
    ti_get_volume_dilution_coefficient,
    ti_get_dilution_rate,
    kget_volume_dilution_coefficient,
    kget_dilution_rate,
)

ti.init(arch=ti.cpu)

def test_wrapper_volume():
    volume_array = np.array([10.0, 20.0, 30.0])
    flow_rate_array = np.array([0.1, 0.2, 0.3])
    expected_coefficient = get_volume_dilution_coefficient(
        volume_array, flow_rate_array
    )
    result_coefficient = ti_get_volume_dilution_coefficient(
        volume_array, flow_rate_array
    )
    npt.assert_allclose(result_coefficient, expected_coefficient)

def test_wrapper_rate():
    coefficient_array = np.array([0.01, 0.02, 0.03])
    concentration_array = np.array([100.0, 200.0, 300.0])
    expected_rate = get_dilution_rate(coefficient_array, concentration_array)
    result_rate = ti_get_dilution_rate(coefficient_array, concentration_array)
    npt.assert_allclose(result_rate, expected_rate)

def test_kernel_volume():
    volume_array = np.array([10.0], dtype=np.float64)
    flow_rate_array = np.array([0.1], dtype=np.float64)

    result_field = ti.ndarray(dtype=ti.f64, shape=volume_array.shape)
    volume_field = ti.ndarray(dtype=ti.f64, shape=volume_array.shape)
    flow_rate_field = ti.ndarray(dtype=ti.f64, shape=flow_rate_array.shape)

    volume_field.from_numpy(volume_array)
    flow_rate_field.from_numpy(flow_rate_array)
    kget_volume_dilution_coefficient(
        volume_field, flow_rate_field, result_field
    )
    npt.assert_allclose(
        result_field.to_numpy(),
        get_volume_dilution_coefficient(volume_array, flow_rate_array),
    )

def test_kernel_rate():
    coefficient_array = np.array([0.01], dtype=np.float64)
    concentration_array = np.array([100.0], dtype=np.float64)

    result_field = ti.ndarray(dtype=ti.f64, shape=coefficient_array.shape)
    coefficient_field = ti.ndarray(dtype=ti.f64, shape=coefficient_array.shape)
    concentration_field = ti.ndarray(
        dtype=ti.f64, shape=concentration_array.shape
    )

    coefficient_field.from_numpy(coefficient_array)
    concentration_field.from_numpy(concentration_array)
    kget_dilution_rate(
        coefficient_field, concentration_field, result_field
    )
    npt.assert_allclose(
        result_field.to_numpy(),
        get_dilution_rate(coefficient_array, concentration_array),
    )
