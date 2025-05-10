import taichi as ti
import numpy as np
import particula as par                                 # python reference

from particula.backend.taichi.gas.properties.ti_kinematic_viscosity_module import (
    kget_kinematic_viscosity,
    ti_get_kinematic_viscosity,
    ti_get_kinematic_viscosity_via_system_state,
)

ti.init(arch=ti.cpu)


def test_wrapper_vs_numpy():
    mu  = np.array([1.8e-5, 2.0e-5])
    rho = np.array([1.2, 1.0])
    expected = par.gas.properties.kinematic_viscosity.get_kinematic_viscosity(mu, rho)
    np.testing.assert_allclose(ti_get_kinematic_viscosity(mu, rho), expected, rtol=1e-12)


def test_kernel_direct():
    mu  = np.array([1.8e-5, 2.0e-5])
    rho = np.array([1.2, 1.0])
    res_ti = ti.ndarray(dtype=ti.f64, shape=mu.size)
    mu_ti, rho_ti = (ti.ndarray(dtype=ti.f64, shape=mu.size) for _ in range(2))
    mu_ti.from_numpy(mu); rho_ti.from_numpy(rho)

    kget_kinematic_viscosity(mu_ti, rho_ti, res_ti)
    np.testing.assert_allclose(res_ti.to_numpy(), mu / rho, rtol=1e-12)


def test_via_system_state():
    T   = np.array([300.0, 320.0])
    rho = np.array([1.2, 1.1])
    expected = par.gas.properties.kinematic_viscosity.get_kinematic_viscosity_via_system_state(
        T, rho
    )
    np.testing.assert_allclose(
        ti_get_kinematic_viscosity_via_system_state(T, rho, 1.81e-5, 273.15),
        expected,
        rtol=1e-12,
    )
