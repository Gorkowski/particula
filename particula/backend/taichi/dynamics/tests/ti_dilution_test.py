import numpy as np
import numpy.testing as npt
import taichi as ti

from particula.dynamics.dilution import (
    get_volume_dilution_coefficient,
    get_dilution_rate,
)
from particula.backend.taichi.dynamics.ti_dilution_module import (
    ti_get_volume_dilution_coefficient,
    ti_get_dilution_rate,
    kget_volume_dilution_coefficient,
    kget_dilution_rate,
)

ti.init(arch=ti.cpu)

def test_wrapper_volume():
    v = np.array([10., 20., 30.])
    q = np.array([0.1, 0.2, 0.3])
    expected = get_volume_dilution_coefficient(v, q)
    result = ti_get_volume_dilution_coefficient(v, q)
    npt.assert_allclose(result, expected)

def test_wrapper_rate():
    a = np.array([0.01, 0.02, 0.03])
    c = np.array([100., 200., 300.])
    expected = get_dilution_rate(a, c)
    result = ti_get_dilution_rate(a, c)
    npt.assert_allclose(result, expected)

def test_kernel_volume():
    v = np.array([10.], dtype=np.float64)
    q = np.array([0.1], dtype=np.float64)
    res_ti = ti.ndarray(dtype=ti.f64, shape=v.shape)
    v_ti, q_ti = ti.ndarray(ti.f64, v.shape), ti.ndarray(ti.f64, q.shape)
    v_ti.from_numpy(v); q_ti.from_numpy(q)
    kget_volume_dilution_coefficient(v_ti, q_ti, res_ti)
    npt.assert_allclose(res_ti.to_numpy(),
                        get_volume_dilution_coefficient(v, q))

def test_kernel_rate():
    a = np.array([0.01], dtype=np.float64)
    c = np.array([100.], dtype=np.float64)
    res_ti = ti.ndarray(dtype=ti.f64, shape=a.shape)
    a_ti, c_ti = ti.ndarray(ti.f64, a.shape), ti.ndarray(ti.f64, c.shape)
    a_ti.from_numpy(a); c_ti.from_numpy(c)
    kget_dilution_rate(a_ti, c_ti, res_ti)
    npt.assert_allclose(res_ti.to_numpy(),
                        get_dilution_rate(a, c))
