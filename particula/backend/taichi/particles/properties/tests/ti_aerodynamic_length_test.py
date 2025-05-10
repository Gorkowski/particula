import numpy as np
import numpy.testing as npt
import taichi as ti
ti.init(arch=ti.cpu)

from particula.particles.properties.aerodynamic_size import get_aerodynamic_length
from particula.backend.taichi.particles.properties.ti_aerodynamic_length_module import (
    ti_get_aerodynamic_length,
    kget_aerodynamic_length,
)

def test_wrapper_vs_numpy():
    pl  = np.array([1e-7, 2e-7])
    psc = np.array([1.1, 1.2])
    asc = np.array([1.0, 1.0])
    rho = np.array([1200.0, 1500.0])
    expected = get_aerodynamic_length(pl, psc, asc, rho)
    result   = ti_get_aerodynamic_length(pl, psc, asc, rho)
    npt.assert_allclose(result, expected, rtol=1e-12, atol=0)

def test_kernel_direct():
    pl  = np.array([1e-7])
    psc = np.array([1.1])
    asc = np.array([1.0])
    rho = np.array([1000.0])
    ref_rho, chi = 1000.0, 1.0

    pl_ti  = ti.ndarray(dtype=ti.f64, shape=1); pl_ti.from_numpy(pl)
    psc_ti = ti.ndarray(dtype=ti.f64, shape=1); psc_ti.from_numpy(psc)
    asc_ti = ti.ndarray(dtype=ti.f64, shape=1); asc_ti.from_numpy(asc)
    rho_ti = ti.ndarray(dtype=ti.f64, shape=1); rho_ti.from_numpy(rho)
    out_ti = ti.ndarray(dtype=ti.f64, shape=1)

    kget_aerodynamic_length(pl_ti, psc_ti, asc_ti, rho_ti, ref_rho, chi, out_ti)
    npt.assert_allclose(out_ti.to_numpy(),
                        get_aerodynamic_length(pl, psc, asc, rho))
