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
from particula.backend.dispatch_register import use_backend
use_backend("taichi")
ti.init(arch=ti.cpu)

# sample data -------------------------------------------------------
mc = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
dens = np.array([1000.0, 1200.0, 1300.0], dtype=np.float64)
mm   = np.array([18.0, 58.44, 100.0], dtype=np.float64)
kap  = np.array([0.5, 0.1, 0.0], dtype=np.float64)
pvp  = np.array([1000.0, 2000.0], dtype=np.float64)
act  = np.array([0.95, 0.85], dtype=np.float64)

# wrapper parity ----------------------------------------------------
def test_wrapper_molar():
    assert_allclose(
        get_ideal_activity_molar(mc, mm),
        use_backend.get_ideal_activity_molar(mc, mm)  # dispatch
    )

def test_wrapper_volume():
    assert_allclose(
        get_ideal_activity_volume(mc, dens),
        use_backend.get_ideal_activity_volume(mc, dens)
    )

def test_wrapper_mass():
    assert_allclose(
        get_ideal_activity_mass(mc),
        use_backend.get_ideal_activity_mass(mc)
    )

def test_wrapper_kappa():
    assert_allclose(
        get_kappa_activity(mc, kap, dens, mm, 0),
        use_backend.get_kappa_activity(mc, kap, dens, mm, 0)
    )

def test_wrapper_surf_p():
    assert_allclose(
        get_surface_partial_pressure(pvp, act),
        use_backend.get_surface_partial_pressure(pvp, act)
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
