import taichi as ti
import numpy as np
import unittest
ti.init(arch=ti.cpu, default_fp=ti.f64)

from particula.backend.taichi.particles.ti_surface_strategies import (
    SurfaceStrategyMolar, SurfaceStrategyMass, SurfaceStrategyVolume
)

def test_kelvin_functions():
    strat = SurfaceStrategyMass(surface_tension=0.072, density=1000)
    r_k = strat.kelvin_radius(0.01815, np.array([1.0]), 298.15)
    k_t = strat.kelvin_term(1e-8, 0.01815, np.array([1.0]), 298.15)
    assert r_k > 0.0 and k_t > 1.0

def test_effective_weights():
    surf = SurfaceStrategyMolar(
        surface_tension=np.array([0.072, 0.020]),
        density=np.array([1000., 2000.]),
        molar_mass=np.array([0.01815, 0.100])
    )
    mc = np.array([0.5, 0.5])
    assert np.isfinite(surf.effective_surface_tension(mc))
    assert np.isfinite(surf.effective_density(mc))
