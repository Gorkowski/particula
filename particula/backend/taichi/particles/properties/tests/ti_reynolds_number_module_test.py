import numpy as np
import taichi as ti
ti.init(arch=ti.cpu)

from particula.backend.taichi.particles.properties.ti_reynolds_number_module import (
    kget_particle_reynolds_number,
    ti_get_particle_reynolds_number,
)


def reference(pr, pv, kv):
    return (2.0 * pr * pv) / kv


def test_taichi_wrapper_matches_numpy():
    pr = np.array([1e-6, 2e-6])
    pv = np.array([0.1, 0.2])
    kv = np.array([1.5e-5, 1.5e-5])
    np.testing.assert_allclose(
        ti_get_particle_reynolds_number(pr, pv, kv),
        reference(pr, pv, kv),
    )


def test_kernel_direct_call():
    pr = np.array([1e-6, 2e-6])
    pv = np.array([0.1, 0.2])
    kv = np.array([1.5e-5, 1.5e-5])
    n = pr.size
    pr_ti = ti.ndarray(dtype=ti.f64, shape=n)
    pv_ti = ti.ndarray(dtype=ti.f64, shape=n)
    kv_ti = ti.ndarray(dtype=ti.f64, shape=n)
    out_ti = ti.ndarray(dtype=ti.f64, shape=n)
    pr_ti.from_numpy(pr)
    pv_ti.from_numpy(pv)
    kv_ti.from_numpy(kv)
    kget_particle_reynolds_number(pr_ti, pv_ti, kv_ti, out_ti)
    np.testing.assert_allclose(out_ti.to_numpy(), reference(pr, pv, kv))
