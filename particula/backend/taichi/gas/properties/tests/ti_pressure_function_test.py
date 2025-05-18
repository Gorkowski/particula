import taichi as ti
import numpy as np
from particula.backend.taichi.gas.properties import (
    ti_pressure_function_module as ti_pfm,
)
from particula.gas.properties.pressure_function import (
    get_partial_pressure,
    get_saturation_ratio_from_pressure,
)

ti.init(arch=ti.cpu)

def test_ti_get_partial_pressure_matches_numpy():
    concentration_array = np.array([1.2, 0.8], dtype=np.float64)
    molar_mass_array = np.array([0.02897, 0.032], dtype=np.float64)
    temperature_array = np.array([298.0, 310.0], dtype=np.float64)
    expected_partial_pressure = get_partial_pressure(
        concentration_array, molar_mass_array, temperature_array
    )
    result_partial_pressure = ti_pfm.ti_get_partial_pressure(
        concentration_array, molar_mass_array, temperature_array
    )
    np.testing.assert_allclose(
        result_partial_pressure, expected_partial_pressure, rtol=1e-8
    )

def test_ti_get_saturation_ratio_from_pressure_matches_numpy():
    partial_pressure_array = np.array([800.0, 500.0], dtype=np.float64)
    pure_vapor_pressure_array = np.array([1000.0, 1000.0], dtype=np.float64)
    expected_saturation_ratio = get_saturation_ratio_from_pressure(
        partial_pressure_array, pure_vapor_pressure_array
    )
    result_saturation_ratio = ti_pfm.ti_get_saturation_ratio_from_pressure(
        partial_pressure_array, pure_vapor_pressure_array
    )
    np.testing.assert_allclose(
        result_saturation_ratio, expected_saturation_ratio, rtol=1e-12
    )

def test_kget_partial_pressure_direct_kernel():
    concentration_array = np.array([1.2, 0.8], dtype=np.float64)
    molar_mass_array = np.array([0.02897, 0.032], dtype=np.float64)
    temperature_array = np.array([298.0, 310.0], dtype=np.float64)
    expected_partial_pressure = get_partial_pressure(
        concentration_array, molar_mass_array, temperature_array
    )
    concentration_ti_array = ti.ndarray(dtype=ti.f64, shape=2)
    molar_mass_ti_array = ti.ndarray(dtype=ti.f64, shape=2)
    temperature_ti_array = ti.ndarray(dtype=ti.f64, shape=2)
    result_ti_array = ti.ndarray(dtype=ti.f64, shape=2)
    concentration_ti_array.from_numpy(concentration_array)
    molar_mass_ti_array.from_numpy(molar_mass_array)
    temperature_ti_array.from_numpy(temperature_array)
    ti_pfm.kget_partial_pressure(
        concentration_ti_array,
        molar_mass_ti_array,
        temperature_ti_array,
        result_ti_array,
    )
    np.testing.assert_allclose(
        result_ti_array.to_numpy(), expected_partial_pressure, rtol=1e-8
    )

def test_kget_saturation_ratio_from_pressure_direct_kernel():
    partial_pressure_array = np.array([800.0, 500.0], dtype=np.float64)
    pure_vapor_pressure_array = np.array([1000.0, 1000.0], dtype=np.float64)
    expected_saturation_ratio = get_saturation_ratio_from_pressure(
        partial_pressure_array, pure_vapor_pressure_array
    )
    partial_pressure_ti_array = ti.ndarray(dtype=ti.f64, shape=2)
    pure_vapor_pressure_ti_array = ti.ndarray(dtype=ti.f64, shape=2)
    result_ti_array = ti.ndarray(dtype=ti.f64, shape=2)
    partial_pressure_ti_array.from_numpy(partial_pressure_array)
    pure_vapor_pressure_ti_array.from_numpy(pure_vapor_pressure_array)
    ti_pfm.kget_saturation_ratio_from_pressure(
        partial_pressure_ti_array,
        pure_vapor_pressure_ti_array,
        result_ti_array,
    )
    np.testing.assert_allclose(
        result_ti_array.to_numpy(), expected_saturation_ratio, rtol=1e-12
    )
