"""
Tests for Taichi Stokes-settling-velocity implementation.
"""
import numpy as np
import numpy.testing as npt
import taichi as ti

from particula.particles.properties.settling_velocity import (
    get_particle_settling_velocity,
)
from particula.backend.taichi.particles.properties.ti_settling_velocity import (
    ti_get_particle_settling_velocity,
    kget_particle_settling_velocity,
)

ti.init(arch=ti.cpu)


def test_wrapper_scalar_and_array():
    # scalar
    npt.assert_allclose(
        ti_get_particle_settling_velocity(
            1e-6, 1500.0, 1.2, 1.8e-5, fluid_density=1.225
        ),
        get_particle_settling_velocity(
            1e-6, 1500.0, 1.2, 1.8e-5, fluid_density=1.225
        ),
        rtol=1e-12,
    )

    # array
    r   = np.array([1e-6, 2e-6, 3e-6])
    rho = np.array([1200., 1500., 1800.])
    ccf = np.array([1.1, 1.2, 1.3])
    npt.assert_allclose(
        ti_get_particle_settling_velocity(r, rho, ccf, 1.8e-5, fluid_density=1.225),
        get_particle_settling_velocity(r, rho, ccf, 1.8e-5, fluid_density=1.225),
        rtol=1e-12,
    )


def test_kernel_direct():
    r   = np.linspace(0.5e-6, 3e-6, 5)
    rho = np.linspace(1000., 2000., 5)
    ccf = np.linspace(1.0, 1.3, 5)

    r_ti   = ti.ndarray(dtype=ti.f64, shape=r.size);   r_ti.from_numpy(r)
    rho_ti = ti.ndarray(dtype=ti.f64, shape=r.size); rho_ti.from_numpy(rho)
    ccf_ti = ti.ndarray(dtype=ti.f64, shape=r.size); ccf_ti.from_numpy(ccf)
    res_ti = ti.ndarray(dtype=ti.f64, shape=r.size)

    kget_particle_settling_velocity(
        r_ti, rho_ti, ccf_ti, 1.8e-5, 9.80665, 1.225, res_ti
    )

    npt.assert_allclose(
        res_ti.to_numpy(),
        get_particle_settling_velocity(r, rho, ccf, 1.8e-5, fluid_density=1.225),
        rtol=1e-12,
    )
