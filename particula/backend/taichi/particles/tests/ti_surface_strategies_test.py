import unittest
import numpy as np
import taichi as ti
import particula as par

from particula.backend.taichi.particles.ti_surface_strategies import (
    TiSurfaceStrategyMolar as TiMolar,
)

# Taichi initialisation only once
ti.init(arch=ti.cpu, default_fp=ti.f64)


class TestTiSurfaceStrategyMolar(unittest.TestCase):
    """Unit tests for the Taichi molar mixing surface strategy."""

    def setUp(self):
        self.σ = np.array([0.072, 0.058], dtype=np.float64)
        self.ρ = np.array([998.0, 1100.0], dtype=np.float64)
        self.M = np.array([0.018, 0.046], dtype=np.float64)
        self.c = np.array([1.3, 2.6], dtype=np.float64)
        self.r = 5e-7
        self.T = 298.15

        self.py_strat = par.particles.SurfaceStrategyMolar(self.σ, self.ρ, self.M)
        self.ti_strat = TiMolar(self.σ, self.ρ, self.M)

    def test_effective_surface_tension(self):
        np.testing.assert_allclose(
            self.py_strat.effective_surface_tension(self.c),
            self.ti_strat.effective_surface_tension(self.c).to_numpy(),
            rtol=1e-7,
        )

    def test_effective_density(self):
        np.testing.assert_allclose(
            self.py_strat.effective_density(self.c),
            self.ti_strat.effective_density(self.c).to_numpy(),
            rtol=1e-7,
        )

    def test_kelvin_radius(self):
        np.testing.assert_allclose(
            self.py_strat.kelvin_radius(self.M, self.c, self.T),
            self.ti_strat.kelvin_radius(self.M, self.c, self.T),
            rtol=1e-7,
        )

    def test_kelvin_term(self):
        np.testing.assert_allclose(
            self.py_strat.kelvin_term(self.r, self.M, self.c, self.T),
            self.ti_strat.kelvin_term(self.r, self.M, self.c, self.T),
            rtol=1e-7,
        )

