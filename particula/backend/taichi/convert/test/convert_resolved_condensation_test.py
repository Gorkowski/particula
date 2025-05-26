"""
Round-trip conversion test (python → Taichi → python) for
convert_resolved_condensation utilities.
"""

# std / 3rd-party -------------------------------------------------------
import copy
import unittest

import numpy as np
import taichi as ti
from numpy.testing import assert_allclose

# particula (python backend objects) ------------------------------------
import particula as par
from particula.dynamics.condensation.condensation_strategies import (
    CondensationIsothermal,
)

# conversion helpers under test ----------------------------------------
from particula.backend.taichi.convert.convert_resolved_condensation import (
    build_ti_particle_resolved,
    update_python_aerosol_from_ti,
)

ti.init(arch=ti.cpu, default_fp=ti.f64, default_ip=ti.i32)

# ----------------------------------------------------------------------
# Local builders (identical to those used in earlier tests / benchmark)
# ----------------------------------------------------------------------
def _build_particle_and_gas_python(n_particles: int, n_species: int = 10):
    rng = np.random.default_rng(0)
    mass = np.abs(rng.standard_normal((n_particles, n_species))) * 1.0e-12

    densities = np.linspace(1_000.0, 1_500.0, n_species)
    molar_mass = np.linspace(0.018, 0.018 + 0.002 * (n_species - 1), n_species)

    # ---------- particle ----------
    activity = (
        par.particles.ActivityKappaParameterBuilder()
        .set_density(densities, "kg/m^3")
        .set_kappa(np.zeros(n_species))
        .set_molar_mass(molar_mass, "kg/mol")
        .set_water_index(0)
        .build()
    )
    surface = (
        par.particles.SurfaceStrategyVolumeBuilder()
        .set_density(densities, "kg/m^3")
        .set_surface_tension(np.full(n_species, 0.072), "N/m")
        .build()
    )
    particle = (
        par.particles.ResolvedParticleMassRepresentationBuilder()
        .set_distribution_strategy(
            par.particles.ParticleResolvedSpeciatedMass()
        )
        .set_activity_strategy(activity)
        .set_surface_strategy(surface)
        .set_mass(mass, "kg")
        .set_density(densities, "kg/m^3")
        .set_charge(0)
        .set_volume(1.0, "m^3")  # parcel volume
        .build()
    )

    # ---------- gas ----------
    vp_strategies = [
        par.gas.VaporPressureFactory().get_strategy(
            "water_buck" if i == 0 else "constant",
            (
                None
                if i == 0
                else {
                    "vapor_pressure": 100.0 + i * 50.0,
                    "vapor_pressure_units": "Pa",
                }
            ),
        )
        for i in range(n_species)
    ]
    gas_concentration = np.abs(
        rng.standard_normal(n_species)
    ) * 1.0e-6  # arbitrary concentration in kg/m^3

    gas_species = (
        par.gas.GasSpeciesBuilder()
        .set_name([f"X{i}" for i in range(n_species)])
        .set_molar_mass(molar_mass, "kg/mol")
        .set_vapor_pressure_strategy(vp_strategies)
        .set_concentration(gas_concentration, "kg/m^3")
        .set_partitioning(True)
        .build()
    )

    return particle, gas_species


# ----------------------------------------------------------------------
# Minimal container classes expected by the converter
# ----------------------------------------------------------------------
class _DummyAtmosphere:
    def __init__(self, gas_species, temperature: float, pressure: float):
        self.partitioning_species = gas_species
        self.temperature = temperature
        self.total_pressure = pressure


class _DummyAerosol:
    def __init__(self, particle, atmosphere):
        self.particles = particle
        self.atmosphere = atmosphere


# ----------------------------------------------------------------------
# Actual test-case
# ----------------------------------------------------------------------
class TestConvertResolvedCondensation(unittest.TestCase):
    """Compare one-step mass transfer between python and Taichi paths."""

    @classmethod
    def setUpClass(cls):
        # initialise Taichi once for the whole suite
        ti.init(arch=ti.cpu, default_fp=ti.f64, default_ip=ti.i32)

    # ------------------------------------------------------------------
    def setUp(self):
        self.n_particles = 100
        self.n_species = 10
        self.temperature = 298.15
        self.pressure = 101_325.0
        self.time_step = 1.0

        particle, gas_species = _build_particle_and_gas_python(
            self.n_particles, self.n_species
        )

        atmosphere = _DummyAtmosphere(
            gas_species=gas_species,
            temperature=self.temperature,
            pressure=self.pressure,
        )

        self.aerosol_ref = _DummyAerosol(
            particle=copy.deepcopy(particle), atmosphere=copy.deepcopy(atmosphere)
        )
        self.aerosol_ti = _DummyAerosol(
            particle=particle, atmosphere=atmosphere
        )

        molar_mass_vec = np.linspace(
            0.018, 0.018 + 0.002 * (self.n_species - 1), self.n_species
        )
        self.cond_py = CondensationIsothermal(
            molar_mass=molar_mass_vec,
            diffusion_coefficient=2.0e-5,
            accommodation_coefficient=1.0,
        )

    # ------------------------------------------------------------------
    def test_roundtrip_conversion(self):
        # ----- python reference step ----------------------------------
        self.cond_py.step(
            particle=self.aerosol_ref.particles,
            gas_species=self.aerosol_ref.atmosphere.partitioning_species,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=self.time_step,
        )

        # ----- Taichi step via converter ------------------------------
        ti_sim = build_ti_particle_resolved(
            aerosol=self.aerosol_ti,
            cond_py=self.cond_py,
            time_step=self.time_step,
            variant_count=1,
        )
        ti_sim.fused_step()
        update_python_aerosol_from_ti(ti_sim, self.aerosol_ti)

        # ----- comparisons -------------------------------------------
        # compare total particle mass
        total_mass_ti = np.sum(
            self.aerosol_ti.particles.get_species_mass(clone=True)
        )
        total_mass_ref = np.sum(
            self.aerosol_ref.particles.get_species_mass(clone=True)
        )
        assert_allclose(
            total_mass_ti,
            total_mass_ref,
            rtol=1e-6,
            atol=1e-12,
            err_msg="Total particle mass diverges after round-trip",
        )

        assert_allclose(
            self.aerosol_ti.particles.distribution,
            self.aerosol_ref.particles.distribution,
            rtol=1e-8,
            atol=1e-12,
            err_msg="Particle species-mass arrays diverge after round-trip",
        )

        assert_allclose(
            self.aerosol_ti.atmosphere.partitioning_species.concentration,
            self.aerosol_ref.atmosphere.partitioning_species.concentration,
            rtol=1e-8,
            atol=1e-8,
            err_msg="Gas-phase concentration arrays diverge after round-trip",
        )

