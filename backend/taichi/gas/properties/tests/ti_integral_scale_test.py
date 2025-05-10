import taichi as ti
import numpy as np
import numpy.testing as npt

from particula.gas.properties import (
    get_lagrangian_integral_time,
    get_eulerian_integral_length,
)
from particula.backend.taichi.gas.properties.ti_integral_scale_module import (
    get_lagrangian_integral_time_taichi,
    get_eulerian_integral_length_taichi,
    kget_lagrangian_integral_time,
    kget_eulerian_integral_length,
)

ti.init(arch=ti.cpu)

def test_lagrangian_integral_time_wrapper():
    u = np.array([0.3, 0.5, 1.0])
    eps = np.array([1e-4, 2e-4, 4e-4])
    expected = get_lagrangian_integral_time(u, eps)
    result = get_lagrangian_integral_time_taichi(u, eps)
    npt.assert_allclose(result, expected, rtol=1e-12, atol=0)

def test_eulerian_integral_length_wrapper():
    u = np.array([0.3, 0.5, 1.0])
    eps = np.array([1e-4, 2e-4, 4e-4])
    expected = get_eulerian_integral_length(u, eps)
    result = get_eulerian_integral_length_taichi(u, eps)
    npt.assert_allclose(result, expected, rtol=1e-12, atol=0)

def test_lagrangian_integral_time_kernel():
    u = np.array([0.3, 0.5, 1.0])
    eps = np.array([1e-4, 2e-4, 4e-4])
    expected = get_lagrangian_integral_time(u, eps)
    n = u.size
    u_ti = ti.ndarray(dtype=ti.f64, shape=n)
    eps_ti = ti.ndarray(dtype=ti.f64, shape=n)
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)
    u_ti.from_numpy(u)
    eps_ti.from_numpy(eps)
    kget_lagrangian_integral_time(u_ti, eps_ti, res_ti)
    result = res_ti.to_numpy()
    npt.assert_allclose(result, expected, rtol=1e-12, atol=0)

def test_eulerian_integral_length_kernel():
    u = np.array([0.3, 0.5, 1.0])
    eps = np.array([1e-4, 2e-4, 4e-4])
    expected = get_eulerian_integral_length(u, eps)
    n = u.size
    u_ti = ti.ndarray(dtype=ti.f64, shape=n)
    eps_ti = ti.ndarray(dtype=ti.f64, shape=n)
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)
    u_ti.from_numpy(u)
    eps_ti.from_numpy(eps)
    kget_eulerian_integral_length(u_ti, eps_ti, res_ti)
    result = res_ti.to_numpy()
    npt.assert_allclose(result, expected, rtol=1e-12, atol=0)
