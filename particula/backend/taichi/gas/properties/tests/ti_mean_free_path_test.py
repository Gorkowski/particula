import numpy as np
import taichi as ti

ti.init(arch=ti.cpu)

from particula.backend.taichi.gas.properties.ti_mean_free_path_module import (
    get_molecule_mean_free_path_taichi as ti_get_molecule_mean_free_path,
    kget_molecule_mean_free_path,
)
from particula.gas.properties.mean_free_path import get_molecule_mean_free_path
from particula.gas.properties.dynamic_viscosity import get_dynamic_viscosity
from particula.util.constants import MOLECULAR_WEIGHT_AIR


def test_wrapper_matches_reference():
    temps = np.array([300.0])
    mm      = np.full_like(temps, MOLECULAR_WEIGHT_AIR, dtype=np.float64)
    press   = np.full_like(temps, 101325.0,          dtype=np.float64)

    np.testing.assert_allclose(
        ti_get_molecule_mean_free_path(mm, temps, press),
        get_molecule_mean_free_path(mm, temps, press),
        rtol=1e-8,
    )


def test_kernel_direct():
    temps  = np.array([280.0], dtype=np.float64)
    n      = temps.size
    mm_np  = np.full(n, MOLECULAR_WEIGHT_AIR, dtype=np.float64)
    P_np   = np.full(n, 101325.0,             dtype=np.float64)
    mu_np  = get_dynamic_viscosity(temps)

    mm_ti  = ti.ndarray(dtype=ti.f64, shape=n);  mm_ti.from_numpy(mm_np)
    T_ti   = ti.ndarray(dtype=ti.f64, shape=n);  T_ti.from_numpy(temps)
    P_ti   = ti.ndarray(dtype=ti.f64, shape=n);  P_ti.from_numpy(P_np)
    mu_ti  = ti.ndarray(dtype=ti.f64, shape=n);  mu_ti.from_numpy(mu_np)
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)

    kget_molecule_mean_free_path(mm_ti, T_ti, P_ti, mu_ti, res_ti)

    np.testing.assert_allclose(
        res_ti.to_numpy(),
        get_molecule_mean_free_path(mm_np, temps, P_np, mu_np),
        rtol=1e-8,
    )
