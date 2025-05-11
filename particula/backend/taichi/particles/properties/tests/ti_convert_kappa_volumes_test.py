import numpy as np
import taichi as ti
ti.init(arch=ti.cpu)

from particula.particles.properties import convert_kappa_volumes as ref
from particula.backend.taichi.particles.properties.ti_convert_kappa_volumes import (
    kget_solute_volume_from_kappa,
    kget_water_volume_from_kappa,
    kget_kappa_from_volumes,
    kget_water_volume_in_mixture,
    ti_get_solute_volume_from_kappa,
    ti_get_water_volume_from_kappa,
    ti_get_kappa_from_volumes,
    ti_get_water_volume_in_mixture,
)


def _rand(n=5):
    return np.random.random(n) + 1e-3


def test_wrapper_parity():
    vt, kp, aw = _rand(), _rand(), 0.8
    vs, vw = _rand(), _rand()
    phi_w = 0.5

    np.testing.assert_allclose(
        ti_get_solute_volume_from_kappa(vt, kp, aw),
        ref.get_solute_volume_from_kappa(vt, kp, aw),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        ti_get_water_volume_from_kappa(vs, kp, aw),
        ref.get_water_volume_from_kappa(vs, kp, aw),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        ti_get_kappa_from_volumes(vs, vw, aw),
        ref.get_kappa_from_volumes(vs, vw, aw),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        ti_get_water_volume_in_mixture(vs, phi_w),
        ref.get_water_volume_in_mixture(vs, phi_w),
        rtol=1e-12,
    )


def test_kernel_direct():
    n = 4
    vt, kp, aw = _rand(n), _rand(n), 0.8
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)
    vt_t = ti.ndarray(dtype=ti.f64, shape=n); vt_t.from_numpy(vt)
    kp_t = ti.ndarray(dtype=ti.f64, shape=n); kp_t.from_numpy(kp)
    aw_t = np.float64(aw)

    kget_solute_volume_from_kappa(vt_t, kp_t, aw_t, res_ti)
    np.testing.assert_allclose(
        res_ti.to_numpy(),
        ref.get_solute_volume_from_kappa(vt, kp, aw),
        rtol=1e-12,
    )
