import numpy as np
import numpy.testing as npt
import taichi as ti

ti.init(arch=ti.cpu)

from particula.particles.properties.aerodynamic_mobility_module import (
    get_aerodynamic_mobility,
)
from particula.backend.taichi.particles.properties.ti_aerodynamic_mobility_module import (
    ti_get_aerodynamic_mobility,
    kget_aerodynamic_mobility,
)

def test_wrapper_vs_numpy():
    r   = np.array([1e-7, 2e-7, 5e-8])
    scf = np.array([1.2, 1.3, 1.1])
    mu  = np.array([1.8e-5, 1.8e-5, 1.8e-5])
    npt.assert_allclose(
        ti_get_aerodynamic_mobility(r, scf, mu),
        get_aerodynamic_mobility(r, scf, mu),
        rtol=1e-14,
        atol=0.0,
    )

def test_kernel_direct():
    r   = np.array([1e-7, 2e-7])
    scf = np.array([1.05, 1.25])
    mu  = np.array([1.8e-5, 1.8e-5])
    n   = r.size

    r_ti   = ti.ndarray(dtype=ti.f64, shape=n)
    scf_ti = ti.ndarray(dtype=ti.f64, shape=n)
    mu_ti  = ti.ndarray(dtype=ti.f64, shape=n)
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)
    r_ti.from_numpy(r)
    scf_ti.from_numpy(scf)
    mu_ti.from_numpy(mu)

    kget_aerodynamic_mobility(r_ti, scf_ti, mu_ti, res_ti)
    npt.assert_allclose(
        res_ti.to_numpy(),
        get_aerodynamic_mobility(r, scf, mu),
        rtol=1e-14,
        atol=0.0,
    )
