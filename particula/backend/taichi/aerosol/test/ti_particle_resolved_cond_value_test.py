"""
Test that the particle resolved condition value is computed correctly.
By comparing to the Python implementation.
"""

import taichi as ti
import unittest

import numpy as np
from numpy.testing import assert_allclose

import particula as par
from particula.backend.taichi.aerosol.ti_particle_resolved import (
    TiAerosolParticleResolved,
)

# python reference solver
from particula.dynamics.condensation.condensation_strategies import (
    CondensationIsothermal as PyCondensationIsothermal,
)


# ----------------------------------------------------------------------
# Local builders (copied from the benchmark to keep the test self-contained)
# ----------------------------------------------------------------------
def _build_ti_particle_resolved_soa(
    n_particles: int,
    n_species: int = 10,
    n_variants: int = 1,
):
    rng = np.random.default_rng(0)
    species_masses = (
        np.abs(rng.standard_normal((n_particles, n_species))) * 1e-18
    )
    density = np.linspace(1_000.0, 1_500.0, n_species)
    molar_mass = np.linspace(0.018, 0.018 + 0.002 * (n_species - 1), n_species)
    pure_vp = np.full(n_species, 50.0)
    vapor_conc = np.ones(n_species) * 1.0e-3
    kappa = np.zeros(n_species)
    surface_tension = np.full(n_species, 0.072)
    gas_mass = np.ones(n_species) * 1.0e-6
    particle_conc = np.ones(n_particles)

    sim = TiAerosolParticleResolved(
        particle_count=n_particles,
        species_count=n_species,
        variant_count=n_variants,
        time_step=1.0,
        simulation_volume=1.0,
    )
    sim.setup(
        variant_index=0,
        species_masses_np=species_masses,
        density_np=density,
        molar_mass_np=molar_mass,
        pure_vapor_pressure_np=pure_vp,
        vapor_concentration_np=vapor_conc,
        kappa_value_np=kappa,
        surface_tension_np=surface_tension,
        gas_mass_np=gas_mass,
        particle_concentration_np=particle_conc,
    )
    return sim


def _build_particle_and_gas_python(n_particles: int, n_species: int = 10):
    rng = np.random.default_rng(0)
    mass = np.abs(rng.standard_normal((n_particles, n_species))) * 1.0e-18

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
        .set_volume(1.0, "m^3")
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
    gas_species = (
        par.gas.GasSpeciesBuilder()
        .set_name([f"X{i}" for i in range(n_species)])
        .set_molar_mass(molar_mass, "kg/mol")
        .set_vapor_pressure_strategy(vp_strategies)
        .set_concentration(np.ones(n_species), "kg/m^3")
        .set_partitioning(True)
        .build()
    )

    return particle, gas_species


class TestCondensationMassEquality(unittest.TestCase):
    """Compare single-step results between the Python and Taichi back-ends."""

    @classmethod
    def setUpClass(cls):
        # we initialise Taichi only once for the whole test-suite
        ti.init(arch=ti.cpu, default_fp=ti.f64, default_ip=ti.i32)

    # ---- utility ------------------------------------------------------
    @staticmethod
    def _extract_species_masses_py(particle):
        """Return numpy array [n_particles, n_species] from python particle."""
        if hasattr(particle, "mass"):
            return np.asarray(particle.mass)
        if hasattr(particle, "_mass"):
            return np.asarray(particle._mass)
        raise AttributeError(
            "Cannot locate species-mass array in python particle"
        )

    @staticmethod
    def _extract_gas_mass_py(gas_species, simulation_volume: float = 1.0):
        """
        Compute gas-phase mass [kg] for each species as
        concentration [kg/m³] × volume [m³].
        """
        if hasattr(gas_species, "concentration"):
            conc = np.asarray(gas_species.concentration)
        elif hasattr(gas_species, "_concentration"):
            conc = np.asarray(gas_species._concentration)
        else:
            raise AttributeError(
                "Cannot locate concentration array in python gas"
            )
        return conc * simulation_volume

    # ---- actual comparison test --------------------------------------
    def test_single_step_mass_transfer(self):
        n_particles = 32
        n_species = 10

        # ---------- build python side objects -------------------------
        py_particle, py_gas = _build_particle_and_gas_python(
            n_particles, n_species
        )

        molar_mass_vec = np.linspace(
            0.018, 0.018 + 0.002 * (n_species - 1), n_species
        )
        cond_py = PyCondensationIsothermal(
            molar_mass=molar_mass_vec,
            diffusion_coefficient=2.0e-5,
            accommodation_coefficient=1.0,
        )

        # one explicit-Euler step (1 s)
        cond_py.step(
            particle=py_particle,
            gas_species=py_gas,
            temperature=298.15,
            pressure=101_325.0,
            time_step=1.0,
        )

        py_species_mass = self._extract_species_masses_py(py_particle)
        py_gas_mass = self._extract_gas_mass_py(py_gas)

        # ---------- build taichi side objects -------------------------
        ti_sim = _build_ti_particle_resolved_soa(n_particles, n_species)
        ti_sim.fused_step()  # one step

        ti_species_mass = ti_sim.get_species_masses()
        ti_gas_mass = ti_sim.get_gas_mass()

        # ---------- compare ------------------------------------------
        assert_allclose(
            ti_species_mass,
            py_species_mass,
            rtol=1e-6,
            atol=1e-12,
            err_msg="Species-mass arrays diverge between back-ends",
        )
        assert_allclose(
            ti_gas_mass,
            py_gas_mass,
            rtol=1e-6,
            atol=1e-12,
            err_msg="Gas-mass arrays diverge between back-ends",
        )
