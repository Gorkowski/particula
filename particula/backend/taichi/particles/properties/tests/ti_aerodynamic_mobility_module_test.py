import taichi as ti
import numpy as np
import pytest

from particula.backend.taichi.particles.properties.ti_aerodynamic_mobility_module import (
    ti_get_aerodynamic_mobility,
    kget_aerodynamic_mobility,
    fget_aerodynamic_mobility,
)

ti.init(arch=ti.cpu)

def reference_aerodynamic_mobility(particle_radius, slip_correction_factor, dynamic_viscosity):
    return slip_correction_factor / (6 * np.pi * dynamic_viscosity * particle_radius)

def test_ti_get_aerodynamic_mobility_vs_numpy():
    # Test with arrays
    r = np.array([1e-7, 2e-7, 5e-7], dtype=np.float64)
    c = np.array([1.1, 1.2, 1.3], dtype=np.float64)
    mu = np.array([1.8e-5, 1.8e-5, 1.8e-5], dtype=np.float64)
    expected = reference_aerodynamic_mobility(r, c, mu)
    result = ti_get_aerodynamic_mobility(r, c, mu)
    np.testing.assert_allclose(result, expected, rtol=1e-8)

    # Test with scalar
    r_scalar = np.array(5e-8, dtype=np.float64)
    c_scalar = np.array(1.05, dtype=np.float64)
    mu_scalar = np.array(1.9e-5, dtype=np.float64)
    expected_scalar = reference_aerodynamic_mobility(r_scalar, c_scalar, mu_scalar)
    result_scalar = ti_get_aerodynamic_mobility(r_scalar, c_scalar, mu_scalar)
    np.testing.assert_allclose(result_scalar, expected_scalar, rtol=1e-8)

def test_kget_aerodynamic_mobility_kernel_direct():
    # Prepare data
    r = np.array([1e-7, 2e-7, 5e-7], dtype=np.float64)
    c = np.array([1.1, 1.2, 1.3], dtype=np.float64)
    mu = np.array([1.8e-5, 1.8e-5, 1.8e-5], dtype=np.float64)
    expected = reference_aerodynamic_mobility(r, c, mu)

    n = r.size
    r_ti = ti.ndarray(dtype=ti.f64, shape=n)
    c_ti = ti.ndarray(dtype=ti.f64, shape=n)
    mu_ti = ti.ndarray(dtype=ti.f64, shape=n)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n)
    r_ti.from_numpy(r)
    c_ti.from_numpy(c)
    mu_ti.from_numpy(mu)

    kget_aerodynamic_mobility(r_ti, c_ti, mu_ti, result_ti)
    result = result_ti.to_numpy()
    np.testing.assert_allclose(result, expected, rtol=1e-8)
