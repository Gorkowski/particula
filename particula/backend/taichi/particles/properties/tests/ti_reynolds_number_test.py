import numpy as np
import numpy.testing as npt
import taichi as ti

ti.init(arch=ti.cpu)

from particula.particles.properties.reynolds_number import (
    get_particle_reynolds_number as ref_re,
)
from particula.backend.taichi.particles.properties.ti_reynolds_number_module import (
    ti_get_particle_reynolds_number,
    kget_particle_reynolds_number,
)

def test_wrapper_matches_reference():
    pr = np.array([1e-6, 2e-6], dtype=np.float64)
    pv = np.array([0.1, 0.2], dtype=np.float64)
    kv = np.array([1.5e-5, 1.5e-5], dtype=np.float64)
    npt.assert_allclose(
        ti_get_particle_reynolds_number(pr, pv, kv),
        ref_re(pr, pv, kv),
        rtol=1e-12,
    )

def test_kernel_direct_call():
    pr = np.array([5e-7], dtype=np.float64)
    pv = np.array([0.05], dtype=np.float64)
    kv = np.array([1.2e-5], dtype=np.float64)

    pr_ti = ti.ndarray(dtype=ti.f64, shape=1)
    pv_ti = ti.ndarray(dtype=ti.f64, shape=1)
    kv_ti = ti.ndarray(dtype=ti.f64, shape=1)
    res_ti = ti.ndarray(dtype=ti.f64, shape=1)

    pr_ti.from_numpy(pr)
    pv_ti.from_numpy(pv)
    kv_ti.from_numpy(kv)

    kget_particle_reynolds_number(pr_ti, pv_ti, kv_ti, res_ti)

    expected = (2.0 * pr * pv) / kv
    npt.assert_allclose(res_ti.to_numpy(), expected, rtol=1e-12)
