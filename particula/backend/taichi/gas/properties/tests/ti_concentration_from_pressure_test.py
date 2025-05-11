import numpy as np
import taichi as ti
from numpy.testing import assert_allclose

from particula.gas.properties.concentration_function import (
    get_concentration_from_pressure,
)
from particula.backend.taichi.gas.properties.ti_concentration_from_pressure_module import (
    ti_get_concentration_from_pressure,
    kget_concentration_from_pressure,
)

ti.init(arch=ti.cpu)


def test_wrapper_matches_numpy():
    pp = np.array([101325.0, 202650.0], dtype=np.float64)
    mm = np.array([0.02897, 0.02897], dtype=np.float64)
    tt = np.array([298.15, 300.0], dtype=np.float64)

    expected = get_concentration_from_pressure(pp, mm, tt)
    result = ti_get_concentration_from_pressure(pp, mm, tt)

    assert_allclose(result, expected, rtol=1e-8, atol=0)


def test_kernel_direct_call():
    pp = np.array([101325.0, 202650.0], dtype=np.float64)
    mm = np.array([0.02897, 0.02897], dtype=np.float64)
    tt = np.array([298.15, 300.0], dtype=np.float64)

    n = pp.size
    pp_ti = ti.ndarray(dtype=ti.f64, shape=n)
    mm_ti = ti.ndarray(dtype=ti.f64, shape=n)
    tt_ti = ti.ndarray(dtype=ti.f64, shape=n)
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)
    pp_ti.from_numpy(pp)
    mm_ti.from_numpy(mm)
    tt_ti.from_numpy(tt)

    kget_concentration_from_pressure(pp_ti, mm_ti, tt_ti, res_ti)

    assert_allclose(res_ti.to_numpy(), get_concentration_from_pressure(pp, mm, tt),
                    rtol=1e-8, atol=0)
