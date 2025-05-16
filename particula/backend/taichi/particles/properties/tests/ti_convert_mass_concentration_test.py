import taichi as ti
import numpy as np
from particula.particles.properties.convert_mass_concentration import (
    get_mole_fraction_from_mass, get_volume_fraction_from_mass,
    get_mass_fraction_from_mass
)
from particula.backend.taichi.particles.properties.ti_convert_mass_concentration_module import (
    kget_mole_fraction_from_mass, kget_volume_fraction_from_mass,
    ti_get_mole_fraction_from_mass, ti_get_volume_fraction_from_mass,
    kget_mass_fraction_from_mass, ti_get_mass_fraction_from_mass
)

ti.init(arch=ti.cpu)

def _sample():
    mass = np.array([1.0, 2.0, 3.0])
    molar = np.array([0.5, 1.0, 2.0])
    dens = np.array([1000.0, 800.0, 1200.0])
    return mass, molar, dens

def _scalar_sample():
    return np.array([4.2])

def test_wrapper_mole():
    m, mm, _ = _sample()
    np.testing.assert_allclose(
        ti_get_mole_fraction_from_mass(m, mm),
        get_mole_fraction_from_mass(m, mm),
        rtol=1e-7, atol=0)

def test_wrapper_volume():
    m, _, rho = _sample()
    np.testing.assert_allclose(
        ti_get_volume_fraction_from_mass(m, rho),
        get_volume_fraction_from_mass(m, rho),
        rtol=1e-7, atol=0)

def test_kernel_mole():
    m, mm, _ = _sample()
    n = m.size
    m_ti = ti.ndarray(dtype=ti.f64, shape=n); m_ti.from_numpy(m)
    mm_ti = ti.ndarray(dtype=ti.f64, shape=n); mm_ti.from_numpy(mm)
    out = ti.ndarray(dtype=ti.f64, shape=n)
    kget_mole_fraction_from_mass(m_ti, mm_ti, out)
    np.testing.assert_allclose(out.to_numpy(),
        get_mole_fraction_from_mass(m, mm), rtol=1e-7, atol=0)

def test_kernel_volume():
    m, _, rho = _sample()
    n = m.size
    m_ti = ti.ndarray(dtype=ti.f64, shape=n); m_ti.from_numpy(m)
    rho_ti = ti.ndarray(dtype=ti.f64, shape=n); rho_ti.from_numpy(rho)
    out = ti.ndarray(dtype=ti.f64, shape=n)
    kget_volume_fraction_from_mass(m_ti, rho_ti, out)
    np.testing.assert_allclose(out.to_numpy(),
        get_volume_fraction_from_mass(m, rho), rtol=1e-7, atol=0)

def test_wrapper_mass_array():
    m, _, _ = _sample()
    np.testing.assert_allclose(
        ti_get_mass_fraction_from_mass(m),
        get_mass_fraction_from_mass(m),
        rtol=1e-7, atol=0)

def test_wrapper_mass_scalar():
    m = 4.2
    np.testing.assert_allclose(
        ti_get_mass_fraction_from_mass(m),
        get_mass_fraction_from_mass(m),
        rtol=1e-7, atol=0)

def test_kernel_mass():
    m, _, _ = _sample()
    n = m.size
    m_ti = ti.ndarray(dtype=ti.f64, shape=n); m_ti.from_numpy(m)
    out  = ti.ndarray(dtype=ti.f64, shape=n)
    kget_mass_fraction_from_mass(m_ti, out)
    np.testing.assert_allclose(out.to_numpy(),
        get_mass_fraction_from_mass(m), rtol=1e-7, atol=0)

def test_kernel_mass_scalar():
    m = _scalar_sample()
    m_ti = ti.ndarray(dtype=ti.f64, shape=m.size); m_ti.from_numpy(m)
    out  = ti.ndarray(dtype=ti.f64, shape=m.size)
    kget_mass_fraction_from_mass(m_ti, out)
    np.testing.assert_allclose(out.to_numpy(),
        get_mass_fraction_from_mass(m), rtol=1e-7, atol=0)
