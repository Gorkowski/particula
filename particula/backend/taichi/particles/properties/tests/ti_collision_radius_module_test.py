import taichi as ti; ti.init(arch=ti.cpu)
import numpy as np
from numpy.testing import assert_allclose

from particula.particles.properties.collision_radius_module import (
    get_collision_radius_mg1988,
    get_collision_radius_sr1992,
)
from particula.backend.taichi.particles.properties.ti_collision_radius_module import (
    ti_get_collision_radius_mg1988,
    ti_get_collision_radius_sr1992,
    kget_collision_radius_mg1988,
    kget_collision_radius_sr1992,
)

def test_mg1988_wrapper_vs_numpy():
    arr = np.random.rand(10)
    expected = get_collision_radius_mg1988(arr)
    result = ti_get_collision_radius_mg1988(arr)
    assert_allclose(result, expected, rtol=1e-12, atol=1e-12)

def test_mg1988_kernel_vs_numpy():
    arr = np.random.rand(7)
    expected = get_collision_radius_mg1988(arr)
    arr_ti = ti.ndarray(dtype=ti.f64, shape=arr.size)
    out_ti = ti.ndarray(dtype=ti.f64, shape=arr.size)
    arr_ti.from_numpy(arr)
    kget_collision_radius_mg1988(arr_ti, out_ti)
    result = out_ti.to_numpy()
    assert_allclose(result, expected, rtol=1e-12, atol=1e-12)

def test_sr1992_wrapper_vs_numpy():
    r_g = np.random.rand(8) + 1.0
    d_f = np.random.rand(8) + 1.0
    expected = get_collision_radius_sr1992(r_g, d_f)
    result = ti_get_collision_radius_sr1992(r_g, d_f)
    assert_allclose(result, expected, rtol=1e-12, atol=1e-12)

def test_sr1992_kernel_vs_numpy():
    r_g = np.random.rand(5) + 1.0
    d_f = np.random.rand(5) + 1.0
    expected = get_collision_radius_sr1992(r_g, d_f)
    r_g_ti = ti.ndarray(dtype=ti.f64, shape=r_g.size)
    d_f_ti = ti.ndarray(dtype=ti.f64, shape=d_f.size)
    out_ti = ti.ndarray(dtype=ti.f64, shape=r_g.size)
    r_g_ti.from_numpy(r_g)
    d_f_ti.from_numpy(d_f)
    kget_collision_radius_sr1992(r_g_ti, d_f_ti, out_ti)
    result = out_ti.to_numpy()
    assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
