"""
Test that the particle resolved condition value is computed correctly.
By comparing to the Python implementation.
"""

import taichi as ti
import unittest

import numpy as np
from numpy.testing import assert_allclose

# helpers & reference implementations straight from the benchmark
from particula.backend.taichi.dynamics.condensation.benchmark.condensation_isothermal_benchmark import (
    _build_ti_particle_resolved_soa,
    _build_particle_and_gas_python,
)

# python reference solver
from particula.dynamics.condensation.condensation_strategies import (
    CondensationIsothermal as PyCondensationIsothermal,
)


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
        raise AttributeError("Cannot locate species-mass array in python particle")

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
            raise AttributeError("Cannot locate concentration array in python gas")
        return conc * simulation_volume

    # ---- actual comparison test --------------------------------------
    def test_single_step_mass_transfer(self):
        n_particles = 32
        n_species = 10

        # ---------- build python side objects -------------------------
        py_particle, py_gas = _build_particle_and_gas_python(n_particles, n_species)

        molar_mass_vec = np.linspace(0.018, 0.018 + 0.002 * (n_species - 1), n_species)
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


