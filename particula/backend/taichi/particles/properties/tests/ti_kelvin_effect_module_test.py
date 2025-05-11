import taichi as ti
ti.init(arch=ti.cpu)

import numpy as np
from numpy.testing import assert_allclose

from particula.particles.properties.kelvin_effect_module import (
    get_kelvin_radius as np_get_kelvin_radius,
    get_kelvin_term as np_get_kelvin_term,
)
from particula.backend.taichi.particles.properties.ti_kelvin_effect_module import (
    ti_get_kelvin_radius,
    ti_get_kelvin_term,
    kget_kelvin_term,
)

def test_wrapper_kelvin_radius():
    st = np.array([0.072, 0.073], dtype=np.float64)
    de = np.array([1000.0, 950.0], dtype=np.float64)
    mm = np.array([0.018, 0.020], dtype=np.float64)
    T = 298.15
    expected = np_get_kelvin_radius(st, de, mm, T)
    got = ti_get_kelvin_radius(st, de, mm, T)
    assert_allclose(got, expected, rtol=1e-8, atol=0)


def test_kernel_kelvin_term():
    pr = np.array([1e-7, 5e-8, 1e-7], dtype=np.float64)
    kr = np.array([2e-7, 1e-7, 5e-8], dtype=np.float64)
    expected = np_get_kelvin_term(pr, kr)

    pr_mat = np.broadcast_to(pr[:, None], (pr.size, kr.size))
    kr_mat = np.broadcast_to(kr[None, :], (pr.size, kr.size))

    pr_ti = ti.ndarray(dtype=ti.f64, shape=pr_mat.shape); pr_ti.from_numpy(pr_mat)
    kr_ti = ti.ndarray(dtype=ti.f64, shape=kr_mat.shape); kr_ti.from_numpy(kr_mat)
    res_ti = ti.ndarray(dtype=ti.f64, shape=pr_mat.shape)

    kget_kelvin_term(pr_ti, kr_ti, res_ti)
    got = res_ti.to_numpy()
    assert_allclose(got, expected, rtol=1e-8, atol=0)


def test_wrapper_kelvin_term_scalar_kr():
    pr = np.array([1e-7, 5e-8, 1e-7], dtype=np.float64)
    kr = np.array([2e-7], dtype=np.float64)
    expected = np_get_kelvin_term(pr, kr)
    got = ti_get_kelvin_term(pr, kr)
    assert_allclose(got, expected, rtol=1e-8, atol=0)
