import taichi as ti
import numpy as np
from numpy.testing import assert_allclose

from particula.particles.properties.activity_module import (
    get_ideal_activity_molar,
    get_ideal_activity_volume,
    get_ideal_activity_mass,
    get_kappa_activity,
    get_surface_partial_pressure,
)
from particula.backend.taichi.particles.properties.ti_activity_module import (
    ti_get_ideal_activity_molar,
    ti_get_ideal_activity_volume,
    ti_get_ideal_activity_mass,
    ti_get_kappa_activity,
    ti_get_surface_partial_pressure,
    fget_surface_partial_pressure,
    fget_ideal_activity_mass,
    fget_ideal_activity_volume,
    fget_ideal_activity_molar,
    kget_ideal_activity_mass,
    kget_surface_partial_pressure,
    kget_ideal_activity_volume,
    kget_ideal_activity_molar,
    kget_kappa_activity,
)
ti.init(arch=ti.cpu)

# sample data -------------------------------------------------------
mc = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
dens = np.array([1000.0, 1200.0, 1300.0], dtype=np.float64)
mm   = np.array([18.0, 58.44, 100.0], dtype=np.float64)
kap  = np.array([0.5, 0.1, 0.0], dtype=np.float64)
pvp  = np.array([1000.0, 2000.0], dtype=np.float64)
act  = np.array([0.95, 0.85], dtype=np.float64)

@ti.kernel
def _call_fget_surf(pvp_val: ti.f64, act_val: ti.f64) -> ti.f64:
    return fget_surface_partial_pressure(pvp_val, act_val)

@ti.kernel
def _call_fget_mass(mass_single: ti.f64, total_mass: ti.f64) -> ti.f64:
    return fget_ideal_activity_mass(mass_single, total_mass)

@ti.kernel
def _call_fget_vol(mass_single: ti.f64,
                   dens_single: ti.f64,
                   total_vol: ti.f64) -> ti.f64:
    return fget_ideal_activity_volume(mass_single, dens_single, total_vol)

@ti.kernel
def _call_fget_mol(mass_single: ti.f64,
                   mm_single: ti.f64,
                   total_mol: ti.f64) -> ti.f64:
    return fget_ideal_activity_molar(mass_single, mm_single, total_mol)

# wrapper parity ----------------------------------------------------
def test_wrapper_molar():
    assert_allclose(
        get_ideal_activity_molar(mc, mm),
        ti_get_ideal_activity_molar(mc, mm)
    )

def test_wrapper_volume():
    assert_allclose(
        get_ideal_activity_volume(mc, dens),
        ti_get_ideal_activity_volume(mc, dens)
    )

def test_wrapper_mass():
    assert_allclose(
        get_ideal_activity_mass(mc),
        ti_get_ideal_activity_mass(mc)
    )

def test_wrapper_kappa():
    assert_allclose(
        get_kappa_activity(mc, kap, dens, mm, 0),
        ti_get_kappa_activity(mc, kap, dens, mm, 0)
    )

def test_wrapper_surf_p():
    assert_allclose(
        get_surface_partial_pressure(pvp, act),
        ti_get_surface_partial_pressure(pvp, act)
    )

# one direct-kernel test (example) ----------------------------------
from particula.backend.taichi.particles.properties.ti_activity_module import (
    kget_ideal_activity_mass,
)
def test_kernel_mass_direct():
    mc_ti = ti.ndarray(dtype=ti.f64, shape=mc.shape)
    res_ti = ti.ndarray(dtype=ti.f64, shape=mc.shape)
    mc_ti.from_numpy(mc)
    kget_ideal_activity_mass(mc_ti, res_ti)
    assert_allclose(res_ti.to_numpy(), get_ideal_activity_mass(mc))

def _to_ti(arr):          # small helper for brevity
    nd = ti.ndarray(dtype=ti.f64, shape=arr.shape)
    nd.from_numpy(arr)
    return nd

# ------------------------------------------------------------------
def test_kernel_surface_partial_pressure_direct():
    res_ti = ti.ndarray(dtype=ti.f64, shape=pvp.shape)
    kget_surface_partial_pressure(
        _to_ti(pvp), _to_ti(act), res_ti
    )
    assert_allclose(res_ti.to_numpy(), get_surface_partial_pressure(pvp, act))

def test_kernel_volume_direct():
    res_ti = ti.ndarray(dtype=ti.f64, shape=mc.shape)
    kget_ideal_activity_volume(
        _to_ti(mc), _to_ti(dens), res_ti
    )
    assert_allclose(res_ti.to_numpy(), get_ideal_activity_volume(mc, dens))

def test_kernel_molar_direct():
    res_ti = ti.ndarray(dtype=ti.f64, shape=mc.shape)
    kget_ideal_activity_molar(
        _to_ti(mc), _to_ti(mm), res_ti
    )
    assert_allclose(res_ti.to_numpy(), get_ideal_activity_molar(mc, mm))

def test_kernel_kappa_direct():
    res_ti = ti.ndarray(dtype=ti.f64, shape=mc.shape)
    kget_kappa_activity(
        _to_ti(mc), _to_ti(kap), _to_ti(dens), _to_ti(mm), 0, res_ti
    )
    assert_allclose(
        res_ti.to_numpy(),
        get_kappa_activity(mc, kap, dens, mm, 0)
    )
