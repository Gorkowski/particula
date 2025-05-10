import taichi as ti
import numpy as np
import pytest
from numpy.testing import assert_allclose

ti.init(arch=ti.cpu)

from particula.backend.taichi.particles.properties.ti_inertia_time_module import (
    ti_get_particle_inertia_time,
    kget_particle_inertia_time,
)
from particula.particles.properties.inertia_time import get_particle_inertia_time

@pytest.mark.parametrize("shape", [(), (5,)])
def test_wrapper_matches_numpy(shape):
    r     = np.full(shape if shape else (), 1e-6, dtype=np.float64)
    rho_p = np.full_like(r, 1_000.0)
    rho_f = np.full_like(r, 1.2)
    nu    = np.full_like(r, 1.5e-5)

    expected = get_particle_inertia_time(r, rho_p, rho_f, nu)
    result   = ti_get_particle_inertia_time(r, rho_p, rho_f, nu)
    assert_allclose(result, expected, rtol=1e-14, atol=0.0)

def test_kernel_direct():
    n = 4
    r     = np.linspace(1e-7, 5e-6, n)
    rho_p = np.linspace(800, 1200, n)
    rho_f = np.full(n, 1.2)
    nu    = np.full(n, 1.5e-5)

    r_ti, rho_p_ti, rho_f_ti, nu_ti, res_ti = [
        ti.ndarray(dtype=ti.f64, shape=n) for _ in range(5)
    ]
    r_ti.from_numpy(r)
    rho_p_ti.from_numpy(rho_p)
    rho_f_ti.from_numpy(rho_f)
    nu_ti.from_numpy(nu)

    kget_particle_inertia_time(r_ti, rho_p_ti, rho_f_ti, nu_ti, res_ti)

    assert_allclose(
        res_ti.to_numpy(),
        (2.0 / 9.0) * (rho_p / rho_f) * (r**2 / nu),
        rtol=1e-14,
        atol=0.0,
    )
