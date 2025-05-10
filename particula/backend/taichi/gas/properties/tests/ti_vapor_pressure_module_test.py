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
    a, b, c = 8.07131, 1730.63, 233.426
    T = np.array([373.15, 350.0])
    ref = get_antoine_vapor_pressure(a, b, c, T)
    taichi_result = ti_get_antoine_vapor_pressure(a, b, c, T)
    assert_allclose(taichi_result, ref, rtol=1e-10, atol=1e-8)

def test_kget_antoine_vapor_pressure_kernel():
    a = np.array([8.07131, 8.07131])
    b = np.array([1730.63, 1730.63])
    c = np.array([233.426, 233.426])
    T = np.array([373.15, 350.0])
    expected = get_antoine_vapor_pressure(a, b, c, T)
    n = a.size
    a_ti = ti.ndarray(dtype=ti.f64, shape=n)
    b_ti = ti.ndarray(dtype=ti.f64, shape=n)
    c_ti = ti.ndarray(dtype=ti.f64, shape=n)
    T_ti = ti.ndarray(dtype=ti.f64, shape=n)
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)
    a_ti.from_numpy(a)
    b_ti.from_numpy(b)
    c_ti.from_numpy(c)
    T_ti.from_numpy(T)
    kget_antoine_vapor_pressure(a_ti, b_ti, c_ti, T_ti, res_ti)
    result = res_ti.to_numpy()
    assert_allclose(result, expected, rtol=1e-10, atol=1e-8)

def test_ti_get_clausius_clapeyron_vapor_pressure_vs_numpy():
    latent_heat = 40660.0
    T0 = 373.15
    P0 = 101325.0
    T = np.array([300.0, 350.0])
    ref = get_clausius_clapeyron_vapor_pressure(latent_heat, T0, P0, T)
    taichi_result = ti_get_clausius_clapeyron_vapor_pressure(latent_heat, T0, P0, T)
    assert_allclose(taichi_result, ref, rtol=1e-10, atol=1e-8)

def test_kget_clausius_clapeyron_vapor_pressure_kernel():
    latent_heat = np.array([40660.0, 40660.0])
    T0 = np.array([373.15, 373.15])
    P0 = np.array([101325.0, 101325.0])
    T = np.array([300.0, 350.0])
    expected = get_clausius_clapeyron_vapor_pressure(latent_heat, T0, P0, T)
    n = latent_heat.size
    lh_ti = ti.ndarray(dtype=ti.f64, shape=n)
    T0_ti = ti.ndarray(dtype=ti.f64, shape=n)
    P0_ti = ti.ndarray(dtype=ti.f64, shape=n)
    T_ti = ti.ndarray(dtype=ti.f64, shape=n)
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)
    lh_ti.from_numpy(latent_heat)
    T0_ti.from_numpy(T0)
    P0_ti.from_numpy(P0)
    T_ti.from_numpy(T)
    kget_clausius_clapeyron_vapor_pressure(
        lh_ti, T0_ti, P0_ti, T_ti, 8.31446261815324, res_ti
    )
    result = res_ti.to_numpy()
    assert_allclose(result, expected, rtol=1e-10, atol=1e-8)

def test_ti_get_buck_vapor_pressure_vs_numpy():
    T = np.array([273.15, 300.0])
    ref = get_buck_vapor_pressure(T)
    taichi_result = ti_get_buck_vapor_pressure(T)
    assert_allclose(taichi_result, ref, rtol=1e-10, atol=1e-8)

def test_kget_buck_vapor_pressure_kernel():
    T = np.array([273.15, 300.0])
    expected = get_buck_vapor_pressure(T)
    n = T.size
    T_ti = ti.ndarray(dtype=ti.f64, shape=n)
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)
    T_ti.from_numpy(T)
    kget_buck_vapor_pressure(T_ti, res_ti)
    result = res_ti.to_numpy()
    assert_allclose(result, expected, rtol=1e-10, atol=1e-8)
