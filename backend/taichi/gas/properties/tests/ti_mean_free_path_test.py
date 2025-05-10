import taichi as ti
import numpy as np
import numpy.testing as npt

from particula.gas.properties.mean_free_path import get_molecule_mean_free_path
from particula.backend.taichi.gas.properties.ti_mean_free_path_module import (
    get_molecule_mean_free_path_taichi,
    kget_molecule_mean_free_path,
)

ti.init(arch=ti.cpu)

def test_wrapper_vs_numpy():
    T = np.array([280.0, 300.0, 320.0])
    P = np.array([90000.0, 101325.0, 120000.0])
    M = np.array([0.029, 0.029, 0.029])
    ref = get_molecule_mean_free_path(M, T, P)
    taichi_val = get_molecule_mean_free_path_taichi(M, T, P)
    npt.assert_allclose(taichi_val, ref, rtol=1e-12, atol=0)

def test_kernel_direct():
    T = np.array([298.15], dtype=np.float64)
    P = np.array([101325.0], dtype=np.float64)
    M = np.array([0.029], dtype=np.float64)
    mu = np.array([1.81e-5], dtype=np.float64)

    mm_ti = ti.ndarray(dtype=ti.f64, shape=1); mm_ti.from_numpy(M)
    T_ti  = ti.ndarray(dtype=ti.f64, shape=1); T_ti.from_numpy(T)
    P_ti  = ti.ndarray(dtype=ti.f64, shape=1); P_ti.from_numpy(P)
    mu_ti = ti.ndarray(dtype=ti.f64, shape=1); mu_ti.from_numpy(mu)
    out_ti = ti.ndarray(dtype=ti.f64, shape=1)

    kget_molecule_mean_free_path(mm_ti, T_ti, P_ti, mu_ti, out_ti)
    expected = get_molecule_mean_free_path(M[0], T[0], P[0], mu[0])
    npt.assert_allclose(out_ti.to_numpy()[0], expected, rtol=1e-12, atol=0)
