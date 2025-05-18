import taichi as ti; ti.init(arch=ti.cpu, default_fp=ti.f64)
import numpy as np, numpy.testing as npt

from particula.particles.distribution_strategies import (
    MassBasedMovingBin,
    ParticleResolvedSpeciatedMass,          # NEW
)
from particula.backend.taichi.particles import (
    TiMassBasedMovingBin,
    TiParticleResolvedSpeciatedMass,        # NEW
)
from particula.particles.activity_strategies import ActivityIdealMass
from particula.backend.taichi.particles.ti_activity_strategies import ActivityIdealMass as TiActivityIdealMass
from particula.particles.surface_strategies import SurfaceStrategyMass
from particula.backend.taichi.particles.ti_surface_strategies import SurfaceStrategyMass as TiSurfaceStrategyMass
from particula.particles.representation import ParticleRepresentation as PyRep
from particula.backend.taichi.particles.ti_representation import TiParticleRepresentation as TiRep

def test_mass_concentration_parity():
    distribution   = np.array([1e-18, 3e-18])
    density        = np.array([1000.0])
    concentration  = np.array([1e6, 2e6])
    charge         = np.array([0.0, 0.0])

    py_obj = PyRep(
        MassBasedMovingBin(), ActivityIdealMass(), SurfaceStrategyMass(),
        distribution, density, concentration, charge,
    )
    ti_obj = TiRep(
        TiMassBasedMovingBin(), TiActivityIdealMass(), TiSurfaceStrategyMass(),
        distribution, density, concentration, charge,
    )
    npt.assert_allclose(py_obj.get_mass_concentration(),
                        ti_obj.get_mass_concentration(),
                        rtol=1e-12)

def test_particle_resolved_mass_concentration_parity():
    # two particles, two species each
    distribution  = np.array([[1e-18, 2e-18],
                              [2e-18, 3e-18]], dtype=np.float64)
    density       = np.array([[1000.0, 1200.0],
                              [1000.0, 1200.0]], dtype=np.float64)
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
        rtol=1e-12,
    )

# ADD BELOW the existing two tests
def test_representation_method_parity():
    # single-species moving-bin
    distribution   = np.array([1e-18, 3e-18])
    density        = np.array([1000.0])
    concentration  = np.array([1e6, 2e6])
    charge         = np.zeros_like(concentration)
    added_mass     = np.array([1e-19, 2e-19])

    py = PyRep(
        MassBasedMovingBin(), ActivityIdealMass(), SurfaceStrategyMass(),
        distribution, density, concentration, charge,
    )
    ti = TiRep(
        TiMassBasedMovingBin(), TiActivityIdealMass(), TiSurfaceStrategyMass(),
        distribution, density, concentration, charge,
    )

    # ─── parity of getter methods ────────────────────────────────
    npt.assert_allclose(py.get_species_mass(),        ti.get_species_mass(),        rtol=1e-12)
    npt.assert_allclose(py.get_mass(),                ti.get_mass(),                rtol=1e-12)
    npt.assert_allclose(py.get_radius(),              ti.get_radius(),              rtol=1e-12)
    npt.assert_allclose(py.get_effective_density(),   ti.get_effective_density(),   rtol=1e-12)
    assert np.isclose(py.get_mean_effective_density(),
                      ti.get_mean_effective_density(), rtol=1e-12)
    npt.assert_allclose(py.get_total_concentration(), ti.get_total_concentration(), rtol=1e-12)

    # ─── mutate both objects identically & re-check ──────────────
    py.add_mass(added_mass)
    ti.add_mass(added_mass)

    npt.assert_allclose(py.get_distribution(),        ti.get_distribution(),        rtol=1e-12)
    npt.assert_allclose(py.get_mass_concentration(),  ti.get_mass_concentration(),  rtol=1e-12)
