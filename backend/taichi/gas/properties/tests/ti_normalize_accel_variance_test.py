import taichi as ti
import numpy as np
import numpy.testing as npt

from particula.backend.taichi.gas.properties.ti_normalize_accel_variance_module import (
    kget_normalized_accel_variance_ao2008,
    get_normalized_accel_variance_ao2008_taichi,
)
from particula.gas.properties.normalize_accel_variance import (
    get_normalized_accel_variance_ao2008,
)

ti.init(arch=ti.cpu)


def test_wrapper_matches_numpy():
    rl = np.array([10.0, 50.0, 100.0])
    eps = 1e-14
    expected = get_normalized_accel_variance_ao2008(rl, eps)
    result = get_normalized_accel_variance_ao2008_taichi(rl, eps)
    npt.assert_allclose(result, expected, rtol=1e-12, atol=0)


def test_kernel_direct():
    rl = np.array([20.0, 70.0, 500.0], dtype=np.float64)
    eps = 1e-14
    rl_ti = ti.ndarray(dtype=ti.f64, shape=rl.size)
    res_ti = ti.ndarray(dtype=ti.f64, shape=rl.size)
    rl_ti.from_numpy(rl)
    kget_normalized_accel_variance_ao2008(rl_ti, eps, res_ti)
    expected = get_normalized_accel_variance_ao2008(rl, eps)
    npt.assert_allclose(res_ti.to_numpy(), expected, rtol=1e-12, atol=0)
