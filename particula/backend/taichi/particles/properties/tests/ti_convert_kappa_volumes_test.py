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

def _ref_vec_get_solute_volume_from_kappa(vt, kp, aw):
    """Reference wrapper that loops because ref code needs scalar kappa."""
    vt_arr, kp_arr = np.asarray(vt), np.asarray(kp)
    flat = [
        ref.get_solute_volume_from_kappa(v, k, aw)
        for v, k in zip(vt_arr.ravel(), kp_arr.ravel())
    ]
    return np.reshape(np.array(flat), vt_arr.shape)


def test_get_solute_volume_from_kappa_wrapper():
    vt, kp, aw = _rand(), _rand(), 0.8
    np.testing.assert_allclose(
        ti_get_solute_volume_from_kappa(vt, kp, aw),
        _ref_vec_get_solute_volume_from_kappa(vt, kp, aw),
        rtol=1e-7,
    )

def test_get_water_volume_from_kappa_wrapper():
    vs, kp, aw = _rand(), _rand(), 0.8
    np.testing.assert_allclose(
        ti_get_water_volume_from_kappa(vs, kp, aw),
        ref.get_water_volume_from_kappa(vs, kp, aw),
        rtol=1e-7,
    )

def test_get_kappa_from_volumes_wrapper():
    vs, vw, aw = _rand(), _rand(), 0.8
    np.testing.assert_allclose(
        ti_get_kappa_from_volumes(vs, vw, aw),
        ref.get_kappa_from_volumes(vs, vw, aw),
        rtol=1e-12,
    )

def test_get_water_volume_in_mixture_wrapper():
    vsd, phi_w = _rand(), 0.5
    np.testing.assert_allclose(
        ti_get_water_volume_in_mixture(vsd, phi_w),
        ref.get_water_volume_in_mixture(vsd, phi_w),
        rtol=1e-12,
    )


def test_kget_solute_volume_from_kappa_kernel():
    n = 4
    vt, kp, aw = _rand(n), _rand(n), 0.8
    res = ti.ndarray(dtype=ti.f64, shape=n)
    vt_t = ti.ndarray(dtype=ti.f64, shape=n); vt_t.from_numpy(vt)
    kp_t = ti.ndarray(dtype=ti.f64, shape=n); kp_t.from_numpy(kp)
    kget_solute_volume_from_kappa(vt_t, kp_t, np.float64(aw), res)
    np.testing.assert_allclose(
        res.to_numpy(),
        _ref_vec_get_solute_volume_from_kappa(vt, kp, aw),
        rtol=1e-7,
    )

def test_kget_water_volume_from_kappa_kernel():
    n = 4
    vs, kp, aw = _rand(n), _rand(n), 0.8
    res = ti.ndarray(dtype=ti.f64, shape=n)
    vs_t = ti.ndarray(dtype=ti.f64, shape=n); vs_t.from_numpy(vs)
    kp_t = ti.ndarray(dtype=ti.f64, shape=n); kp_t.from_numpy(kp)
    kget_water_volume_from_kappa(vs_t, kp_t, np.float64(aw), res)
    np.testing.assert_allclose(
        res.to_numpy(),
        ref.get_water_volume_from_kappa(vs, kp, aw),
        rtol=1e-7,
    )

def test_kget_kappa_from_volumes_kernel():
    n = 4
    vs, vw, aw = _rand(n), _rand(n), 0.8
    res = ti.ndarray(dtype=ti.f64, shape=n)
    vs_t = ti.ndarray(dtype=ti.f64, shape=n); vs_t.from_numpy(vs)
    vw_t = ti.ndarray(dtype=ti.f64, shape=n); vw_t.from_numpy(vw)
    kget_kappa_from_volumes(vs_t, vw_t, np.float64(aw), res)
    np.testing.assert_allclose(
        res.to_numpy(),
        ref.get_kappa_from_volumes(vs, vw, aw),
        rtol=1e-12,
    )

def test_kget_water_volume_in_mixture_kernel():
    n = 4
    vsd, phi = _rand(n), np.full(n, 0.5)
    res = ti.ndarray(dtype=ti.f64, shape=n)
    vsd_t = ti.ndarray(dtype=ti.f64, shape=n); vsd_t.from_numpy(vsd)
    phi_t = ti.ndarray(dtype=ti.f64, shape=n); phi_t.from_numpy(phi)
    kget_water_volume_in_mixture(vsd_t, phi_t, res)
    np.testing.assert_allclose(
        res.to_numpy(),
        ref.get_water_volume_in_mixture(vsd, phi),
        rtol=1e-12,
    )
