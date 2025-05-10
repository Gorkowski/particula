import numpy as np
import numpy.testing as npt
import taichi as ti

ti.init(arch=ti.cpu)

from particula.particles.properties.friction_factor_module import get_friction_factor
from particula.backend.taichi.particles.properties.ti_friction_factor_module import (
    ti_get_friction_factor,
    kget_friction_factor,
)

def test_wrapper_vs_numpy():
    radius = np.array([1e-7, 2e-7, 5e-8])
    mu = 1.8e-5
    corr = np.array([1.1, 1.2, 1.3])

    expected = get_friction_factor(radius, mu, corr)
    actual = ti_get_friction_factor(radius, mu, corr)
    npt.assert_allclose(actual, expected, rtol=1e-14, atol=0.0)

def test_kernel_direct():
    radius = np.array([1e-7, 2e-7])
    mu = 1.8e-5
    corr = np.array([1.05, 1.25])
    n = radius.size

    radius_ti = ti.ndarray(dtype=ti.f64, shape=n)
    corr_ti = ti.ndarray(dtype=ti.f64, shape=n)
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)
    radius_ti.from_numpy(radius)
    corr_ti.from_numpy(corr)

    kget_friction_factor(radius_ti, corr_ti, mu, res_ti)
    npt.assert_allclose(res_ti.to_numpy(), get_friction_factor(radius, mu, corr),
                        rtol=1e-14, atol=0.0)
