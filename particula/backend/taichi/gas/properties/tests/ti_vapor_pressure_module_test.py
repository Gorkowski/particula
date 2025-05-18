import taichi as ti
import numpy as np
import pytest
from numpy.testing import assert_allclose

from particula.backend.taichi.gas.properties.ti_vapor_pressure_module import (
    ti_get_antoine_vapor_pressure,
    ti_get_clausius_clapeyron_vapor_pressure,
    ti_get_buck_vapor_pressure,
    kget_antoine_vapor_pressure,
    kget_clausius_clapeyron_vapor_pressure,
    kget_buck_vapor_pressure,
)
from particula.gas.properties.vapor_pressure_module import (
    get_antoine_vapor_pressure,
    get_clausius_clapeyron_vapor_pressure,
    get_buck_vapor_pressure,
)

ti.init(arch=ti.cpu)

def test_ti_get_antoine_vapor_pressure_vs_numpy():
    # Antoine parameters for water at 100°C
    constant_a, constant_b, constant_c = 8.07131, 1730.63, 233.426
    temperature_array = np.array([373.15, 350.0])
    expected = get_antoine_vapor_pressure(constant_a, constant_b, constant_c, temperature_array)
    taichi_result = ti_get_antoine_vapor_pressure(constant_a, constant_b, constant_c, temperature_array)
    assert_allclose(taichi_result, expected, rtol=1e-10, atol=1e-8)

def test_kget_antoine_vapor_pressure_kernel():
    constant_a_array = np.array([8.07131, 8.07131])
    constant_b_array = np.array([1730.63, 1730.63])
    constant_c_array = np.array([233.426, 233.426])
    temperature_array = np.array([373.15, 350.0])
    expected = get_antoine_vapor_pressure(constant_a_array, constant_b_array, constant_c_array, temperature_array)
    n_elements = constant_a_array.size
    constant_a_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    constant_b_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    constant_c_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    temperature_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    constant_a_ti.from_numpy(constant_a_array)
    constant_b_ti.from_numpy(constant_b_array)
    constant_c_ti.from_numpy(constant_c_array)
    temperature_ti.from_numpy(temperature_array)
    kget_antoine_vapor_pressure(constant_a_ti, constant_b_ti, constant_c_ti, temperature_ti, result_ti)
    result_np = result_ti.to_numpy()
    assert_allclose(result_np, expected, rtol=1e-10, atol=1e-8)

def test_ti_get_clausius_clapeyron_vapor_pressure_vs_numpy():
    latent_heat = 40660.0
    temperature_initial = 373.15
    pressure_initial = 101325.0
    temperature_array = np.array([300.0, 350.0])
    expected = get_clausius_clapeyron_vapor_pressure(latent_heat, temperature_initial, pressure_initial, temperature_array)
    taichi_result = ti_get_clausius_clapeyron_vapor_pressure(latent_heat, temperature_initial, pressure_initial, temperature_array)
    assert_allclose(taichi_result, expected, rtol=1e-10, atol=1e-8)

def test_kget_clausius_clapeyron_vapor_pressure_kernel():
    latent_heat_array = np.array([40660.0, 40660.0])
    temperature_initial_array = np.array([373.15, 373.15])
    pressure_initial_array = np.array([101325.0, 101325.0])
    temperature_array = np.array([300.0, 350.0])
    expected = get_clausius_clapeyron_vapor_pressure(latent_heat_array, temperature_initial_array, pressure_initial_array, temperature_array)
    n_elements = latent_heat_array.size
    latent_heat_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    temperature_initial_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    pressure_initial_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    temperature_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    latent_heat_ti.from_numpy(latent_heat_array)
    temperature_initial_ti.from_numpy(temperature_initial_array)
    pressure_initial_ti.from_numpy(pressure_initial_array)
    temperature_ti.from_numpy(temperature_array)
    kget_clausius_clapeyron_vapor_pressure(
        latent_heat_ti, temperature_initial_ti, pressure_initial_ti, temperature_ti, 8.31446261815324, result_ti
    )
    result_np = result_ti.to_numpy()
    assert_allclose(result_np, expected, rtol=1e-10, atol=1e-8)

def test_ti_get_buck_vapor_pressure_vs_numpy():
    temperature_array = np.array([273.15, 300.0])
    expected = get_buck_vapor_pressure(temperature_array)
    taichi_result = ti_get_buck_vapor_pressure(temperature_array)
    assert_allclose(taichi_result, expected, rtol=1e-6, atol=1e-8)

def test_kget_buck_vapor_pressure_kernel():
    temperature_array = np.array([273.15, 300.0])
    expected = get_buck_vapor_pressure(temperature_array)
    n_elements = temperature_array.size
    temperature_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    temperature_ti.from_numpy(temperature_array)
    kget_buck_vapor_pressure(temperature_ti, result_ti)
    result_np = result_ti.to_numpy()
    assert_allclose(result_np, expected, rtol=1e-6, atol=1e-8)
