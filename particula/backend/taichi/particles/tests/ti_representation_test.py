import taichi as ti; ti.init(arch=ti.cpu, default_fp=ti.f64)
import numpy as np, numpy.testing as npt

from particula.particles.distribution_strategies import (
    ParticleResolvedSpeciatedMass,          # NEW
)
from particula.backend.taichi.particles import (
    TiParticleResolvedSpeciatedMass,        # NEW
)
from particula.particles.activity_strategies import ActivityIdealMass
from particula.backend.taichi.particles.ti_activity_strategies import ActivityIdealMass as TiActivityIdealMass
from particula.particles.surface_strategies import SurfaceStrategyMass
from particula.backend.taichi.particles.ti_surface_strategies import SurfaceStrategyMass as TiSurfaceStrategyMass
from particula.particles.representation import ParticleRepresentation as PyRep
from particula.backend.taichi.particles.ti_representation import TiParticleRepresentation as TiRep


def test_particle_resolved_mass_concentration_parity():
    # two particles, two species each
    distribution  = np.array([[1e-18, 2e-18],
                              [2e-18, 3e-18]], dtype=np.float64)
    density       = np.array([1000.0, 1200.0], dtype=np.float64)
    concentration = np.array([1e6, 2e6], dtype=np.float64)
    charge        = np.zeros(2, dtype=np.float64)

    py_obj = PyRep(
        ParticleResolvedSpeciatedMass(), ActivityIdealMass(), SurfaceStrategyMass(),
        distribution, density, concentration, charge,
    )
    ti_obj = TiRep(
        TiParticleResolvedSpeciatedMass(), TiActivityIdealMass(), TiSurfaceStrategyMass(),
        distribution, density, concentration, charge,
    )

    npt.assert_allclose(
        py_obj.get_mass_concentration(),
        ti_obj.get_mass_concentration(),
        rtol=1e-8,
    )
    # test spcies mass
    npt.assert_allclose(
        py_obj.get_species_mass(),
        ti_obj.get_species_mass(),
        rtol=1e-8,
    )
    # test mass
    npt.assert_allclose(
        py_obj.get_mass(),
        ti_obj.get_mass().to_numpy(),
        rtol=1e-8,
    )
    # test radius
    npt.assert_allclose(
        py_obj.get_radius(),
        ti_obj.get_radius().to_numpy(),
        rtol=1e-8,
    )
    # test effective density
    npt.assert_allclose(
        py_obj.get_effective_density(),
        ti_obj.get_effective_density().to_numpy(),
        rtol=1e-8,
    )
