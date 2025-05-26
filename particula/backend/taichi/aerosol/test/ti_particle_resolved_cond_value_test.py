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
from particula.backend.taichi.convert.convert_resolved_condensation import (
    build_ti_particle_resolved,
    update_python_aerosol_from_ti,
)

# python reference solver
from particula.dynamics.condensation.condensation_strategies import (
    CondensationIsothermal as PyCondensationIsothermal,
)



def _build_particle_and_gas_python(n_particles: int, n_species: int = 10):
    rng = np.random.default_rng(0)
    mass = np.abs(rng.standard_normal((n_particles, n_species))) * 1.0e-10

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
        .set_volume(1e-10, "m^3")
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


    # ---- actual comparison test --------------------------------------
    def test_single_step_mass_transfer(self):
        n_particles = 32
        n_species = 10

        # ---------- build python side objects -------------------------
        py_particle, py_gas = _build_particle_and_gas_python(
            n_particles, n_species
        )
        atmosphere = (
            par.gas.AtmosphereBuilder()
            .set_temperature(298.15, "K")
            .set_pressure(101_325.0, "Pa")
            .set_more_partitioning_species(py_gas)
            .build()
        )
        aerosol = par.Aerosol(
            atmosphere=atmosphere,
            particles=py_particle,
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
            time_step=100.0,
        )

        py_species_mass = py_particle.get_species_mass(clone=True)
        py_gas_mass = py_gas.get_concentration()

        # ---------- build taichi side objects -------------------------
        ti_sim = build_ti_particle_resolved(
            aerosol=aerosol,
            cond_py=cond_py,
            time_step=100.0,
            variant_count=1,
        )

        ti_sim.fused_step()  # one step

        # compare particle activity
        ti_activity = ti_sim.get_activity()
        py_activity = py_particle.get_activity().activity(
            mass_concentration=py_species_mass,
        )
        assert_allclose(
            ti_activity,
            py_activity,
            rtol=1e-6,
            atol=1e-12,
            err_msg="Activity diverges between back-ends",
        )


        # compare concenration
        ti_concentration = ti_sim.get_particle_concentration()
        py_concentration = py_particle.concentration

        assert_allclose(
            ti_concentration,
            py_concentration,
            rtol=1e-6,
            atol=1e-12,
            err_msg="Particle concentration diverges between back-ends",
        )




        # compare mass_transport_rate
        ti_mass_transport_rate = ti_sim.get_mass_transport_rate()
        py_mass_tranport_rate = cond_py.rate(
            particle=py_particle,
            gas_species=py_gas,
            temperature=298.15,
            pressure=101_325.0,
        )

        assert_allclose(
            ti_mass_transport_rate,
            py_mass_tranport_rate,
            rtol=1e-6,
            atol=1e-12,
            err_msg="Mass transport rate diverges between back-ends",
        )

        # ti_species_mass = ti_sim.get_species_masses()
        # ti_gas_mass = ti_sim.get_gas_mass()

        # total_mass = np.sum(ti_species_mass)
        # total_mass_py = np.sum(py_species_mass)

        # assert_allclose(
        #     total_mass,
        #     total_mass_py,
        #     rtol=1e-6,
        #     atol=1e-12,
        #     err_msg="Total mass diverges between back-ends",
        # )

        # ---------- compare ------------------------------------------
        # assert_allclose(
        #     ti_species_mass,
        #     py_species_mass,
        #     rtol=1e-6,
        #     atol=1e-12,
        #     err_msg="Species-mass arrays diverge between back-ends",
        # )
        # assert_allclose(
        #     ti_gas_mass,
        #     py_gas_mass,
        #     rtol=1e-6,
        #     atol=1e-12,
        #     err_msg="Gas-mass arrays diverge between back-ends",
        # )
