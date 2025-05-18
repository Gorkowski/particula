import unittest

import taichi as ti
ti.init(arch=ti.cpu, default_fp=ti.f64)

import numpy as np
import numpy.testing as npt

from particula.particles.distribution_strategies import ParticleResolvedSpeciatedMass
from particula.backend.taichi.particles import TiParticleResolvedSpeciatedMass
from particula.particles.activity_strategies import ActivityIdealMass
from particula.backend.taichi.particles.ti_activity_strategies import (
    ActivityIdealMass as TiActivityIdealMass,
)
from particula.particles.surface_strategies import SurfaceStrategyMass
from particula.backend.taichi.particles.ti_surface_strategies import (
        TiSurfaceStrategyMolar
)
from particula.particles.representation import ParticleRepresentation as PyRep
from particula.backend.taichi.particles.ti_representation import (
    TiParticleRepresentation as TiRep,
)


class TestTiParticleRepresentation(unittest.TestCase):
    """Parity tests for every public getter in ParticleRepresentation."""

    @classmethod
    def setUpClass(cls):
        # five particles, three species each
        cls.distribution = np.array(
            [
                [1e-18, 2e-18, 3e-18],
                [2e-18, 3e-18, 4e-18],
                [3e-18, 4e-18, 5e-18],
                [4e-18, 5e-18, 6e-18],
                [5e-18, 6e-18, 7e-18],
            ],
            dtype=np.float64,
        )
        cls.density = np.array([1.0e3, 1.2e3, 1.5e3], dtype=np.float64)
        cls.concentration = np.array(
            [1e6, 2e6, 1.5e6, 1.2e6, 1e6], dtype=np.float64
        )
        cls.charge = np.zeros(5, dtype=np.float64)

        ti_surface_strategy = TiSurfaceStrategyMolar(
            surface_tension=np.array([0.072, 0.058, 0.045], dtype=np.float64),
            density=cls.density,
            molar_mass=np.array([0.018, 0.046, 0.058], dtype=np.float64),
        )

        cls.py_obj = PyRep(
            ParticleResolvedSpeciatedMass(),
            ActivityIdealMass(),
            SurfaceStrategyMass(),
            cls.distribution,
            cls.density,
            cls.concentration,
            cls.charge,
        )
        cls.ti_obj = TiRep(
            TiParticleResolvedSpeciatedMass(),
            TiActivityIdealMass(),
            ti_surface_strategy,
            cls.distribution,
            cls.density,
            cls.concentration,
            cls.charge,
        )

    # ───── numeric parity checks ────────────────────────────────────────
    def test_mass_concentration(self):
        npt.assert_allclose(
            self.py_obj.get_mass_concentration(),
            self.ti_obj.get_mass_concentration(),
            rtol=1e-7,
        )

    def test_species_mass(self):
        npt.assert_allclose(
            self.py_obj.get_species_mass(),
            self.ti_obj.get_species_mass(),
            rtol=1e-7,
        )

    def test_mass(self):
        npt.assert_allclose(
            self.py_obj.get_mass(),
            self.ti_obj.get_mass().to_numpy(),
            rtol=1e-7,
        )

    def test_radius(self):
        npt.assert_allclose(
            self.py_obj.get_radius(),
            self.ti_obj.get_radius().to_numpy(),
            rtol=1e-6,
        )

    def test_effective_density(self):
        npt.assert_allclose(
            self.py_obj.get_effective_density(),
            self.ti_obj.get_effective_density(),
            rtol=1e-7,
        )

    def test_total_concentration(self):
        npt.assert_allclose(
            self.py_obj.get_total_concentration(),
            self.ti_obj.get_total_concentration(),
            rtol=1e-7,
        )
