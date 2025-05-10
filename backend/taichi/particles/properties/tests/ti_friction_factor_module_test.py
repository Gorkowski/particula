import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

from particula.particles.properties.friction_factor_module import (
    get_friction_factor,
)
from particula.backend.taichi.particles.properties import (
    ti_friction_factor_module as ti_ff,
)


def test_wrapper_matches_numpy():
    pr = np.array([1e-7, 2e-7, 5e-8])
    sc = np.array([1.1, 1.2, 1.05])
    mu = 1.8e-5
    expected = get_friction_factor(pr, mu, sc)
    actual = ti_ff.ti_get_friction_factor(pr, mu, sc)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)


def test_kernel_direct():
    pr = np.linspace(1e-8, 5e-7, 4)
    sc = np.full_like(pr, 1.1)
    mu = 2.0e-5
    n = pr.size

    pr_ti = ti.ndarray(dtype=ti.f64, shape=n)
    sc_ti = ti.ndarray(dtype=ti.f64, shape=n)
    out_ti = ti.ndarray(dtype=ti.f64, shape=n)

    pr_ti.from_numpy(pr)
    sc_ti.from_numpy(sc)

    ti_ff.kget_friction_factor(pr_ti, mu, sc_ti, out_ti)

    expected = get_friction_factor(pr, mu, sc)
    np.testing.assert_allclose(out_ti.to_numpy(), expected, rtol=1e-12, atol=0)
