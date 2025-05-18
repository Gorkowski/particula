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
        self.surface_tension = np.array([0.072, 0.058], dtype=np.float64)
        self.density = np.array([998.0, 1100.0], dtype=np.float64)
        self.molar_mass = np.array([0.018, 0.046], dtype=np.float64)
        self.mass_concentration = np.array([1.3, 2.6], dtype=np.float64)
        self.radius = 5e-7
        self.temperature = 298.15

        self.py_strat = par.particles.SurfaceStrategyMolar(
            self.surface_tension,
            self.density,
            self.molar_mass,
        )
        self.ti_strat = TiMolar(
            self.surface_tension,
            self.density,
            self.molar_mass,
        )

    def test_effective_surface_tension(self):
        np.testing.assert_allclose(
            self.py_strat.effective_surface_tension(self.mass_concentration),
            self.ti_strat.effective_surface_tension(self.mass_concentration),
            rtol=1e-7,
        )

    def test_effective_density(self):
        np.testing.assert_allclose(
            self.py_strat.effective_density(self.mass_concentration),
            self.ti_strat.effective_density(self.mass_concentration),
            rtol=1e-7,
        )

    def test_kelvin_radius(self):
        np.testing.assert_allclose(
            self.py_strat.kelvin_radius(
                self.molar_mass,
                self.mass_concentration,
                self.temperature,
            ),
            self.ti_strat.kelvin_radius(
                self.molar_mass,
                self.mass_concentration,
                self.temperature,
            ),
            rtol=1e-7,
        )

    def test_kelvin_term(self):
        np.testing.assert_allclose(
            self.py_strat.kelvin_term(
                self.radius,
                self.molar_mass,
                self.mass_concentration,
                self.temperature,
            ),
            self.ti_strat.kelvin_term(
                self.radius,
                self.molar_mass,
                self.mass_concentration,
                self.temperature,
            ),
            rtol=1e-7,
        )

