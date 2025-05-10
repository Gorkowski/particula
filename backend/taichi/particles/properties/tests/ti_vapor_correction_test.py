import taichi as ti
ti.init(arch=ti.cpu)

import numpy as np
from particula.particles.properties.vapor_correction_module import (
    get_vapor_transition_correction as np_get,
)
from particula.backend.taichi.particles.properties.ti_vapor_correction_module import (
    ti_get_vapor_transition_correction,
    kget_vapor_transition_correction,
)

def test_wrapper_vs_numpy():
    kn = np.array([0.02, 0.5, 4.0], dtype=np.float64)
    ma = np.array([0.8, 1.0, 0.3], dtype=np.float64)
    np.testing.assert_allclose(
        ti_get_vapor_transition_correction(kn, ma),
        np_get(kn, ma),
    )

def test_kernel_vs_numpy():
    kn = np.array([0.1, 1.0, 10.0], dtype=np.float64)
    ma = np.array([1.0, 0.5, 0.9], dtype=np.float64)

    n = kn.size
    kn_ti = ti.ndarray(dtype=ti.f64, shape=n)
    ma_ti = ti.ndarray(dtype=ti.f64, shape=n)
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)
    kn_ti.from_numpy(kn)
    ma_ti.from_numpy(ma)

    kget_vapor_transition_correction(kn_ti, ma_ti, res_ti)
    np.testing.assert_allclose(res_ti.to_numpy(), np_get(kn, ma))
