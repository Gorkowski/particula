import taichi as ti
import numpy as np
import pytest
from particula.backend.taichi.gas.properties.ti_pressure_function_module import (
    ti_get_partial_pressure,
    ti_get_saturation_ratio_from_pressure,
    kget_partial_pressure,
    kget_saturation_ratio_from_pressure,
)
from particula.gas.properties.pressure_function import (
    get_partial_pressure,
    get_saturation_ratio_from_pressure,
)
from particula.util.constants import GAS_CONSTANT

ti.init(arch=ti.cpu)

def test_ti_get_partial_pressure_matches_numpy():
    concentration = np.array([1.2, 0.8], dtype=np.float64)
    molar_mass = np.array([0.02897, 0.032], dtype=np.float64)
    temperature = np.array([298.0, 310.0], dtype=np.float64)
    expected = get_partial_pressure(concentration, molar_mass, temperature)
    result = ti_get_partial_pressure(concentration, molar_mass, temperature)
    np.testing.assert_allclose(result, expected, rtol=1e-8)

def test_ti_get_saturation_ratio_from_pressure_matches_numpy():
    partial_pressure = np.array([800.0, 500.0], dtype=np.float64)
    pure_vapor_pressure = np.array([1000.0, 1000.0], dtype=np.float64)
    expected = get_saturation_ratio_from_pressure(partial_pressure, pure_vapor_pressure)
    result = ti_get_saturation_ratio_from_pressure(partial_pressure, pure_vapor_pressure)
    np.testing.assert_allclose(result, expected, rtol=1e-12)

def test_kget_partial_pressure_direct_kernel():
    concentration = np.array([1.2, 0.8], dtype=np.float64)
    molar_mass = np.array([0.02897, 0.032], dtype=np.float64)
    temperature = np.array([298.0, 310.0], dtype=np.float64)
    expected = get_partial_pressure(concentration, molar_mass, temperature)
    arr1 = ti.ndarray(dtype=ti.f64, shape=2)
    arr2 = ti.ndarray(dtype=ti.f64, shape=2)
    arr3 = ti.ndarray(dtype=ti.f64, shape=2)
    result = ti.ndarray(dtype=ti.f64, shape=2)
    arr1.from_numpy(concentration)
    arr2.from_numpy(molar_mass)
    arr3.from_numpy(temperature)
    kget_partial_pressure(arr1, arr2, arr3, result)
    np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-8)

def test_kget_saturation_ratio_from_pressure_direct_kernel():
    partial_pressure = np.array([800.0, 500.0], dtype=np.float64)
    pure_vapor_pressure = np.array([1000.0, 1000.0], dtype=np.float64)
    expected = get_saturation_ratio_from_pressure(partial_pressure, pure_vapor_pressure)
    arr1 = ti.ndarray(dtype=ti.f64, shape=2)
    arr2 = ti.ndarray(dtype=ti.f64, shape=2)
    result = ti.ndarray(dtype=ti.f64, shape=2)
    arr1.from_numpy(partial_pressure)
    arr2.from_numpy(pure_vapor_pressure)
    kget_saturation_ratio_from_pressure(arr1, arr2, result)
    np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-12)
