import taichi as ti
import numpy as np
from numpy.testing import assert_allclose

ti.init(arch=ti.cpu)

from particula.gas.properties.integral_scale_module import (                # noqa: E402
    get_lagrangian_integral_time,
    get_eulerian_integral_length,
)
from particula.backend.taichi.gas.properties.ti_integral_scale_module import (  # noqa: E402
    ti_get_lagrangian_integral_time,
    ti_get_eulerian_integral_length,
    kget_lagrangian_integral_time,
    kget_eulerian_integral_length,
)


def test_wrapper_vs_numpy():
    u = np.array([0.1, 0.3, 1.2])
    eps = np.array([1e-5, 1e-4, 1e-3])
    assert_allclose(
        ti_get_lagrangian_integral_time(u, eps),
        get_lagrangian_integral_time(u, eps),
    )
    assert_allclose(
        ti_get_eulerian_integral_length(u, eps),
        get_eulerian_integral_length(u, eps),
    )


def test_kernel_direct():
    u = np.array([0.25], dtype=np.float64)
    eps = np.array([2e-4], dtype=np.float64)

    res = ti.ndarray(dtype=ti.f64, shape=1)
    u_ti, eps_ti = [ti.ndarray(dtype=ti.f64, shape=1) for _ in range(2)]
    u_ti.from_numpy(u)
    eps_ti.from_numpy(eps)

    kget_lagrangian_integral_time(u_ti, eps_ti, res)
    assert_allclose(res.to_numpy(), get_lagrangian_integral_time(u, eps))

    kget_eulerian_integral_length(u_ti, eps_ti, res)
    assert_allclose(res.to_numpy(), get_eulerian_integral_length(u, eps))
