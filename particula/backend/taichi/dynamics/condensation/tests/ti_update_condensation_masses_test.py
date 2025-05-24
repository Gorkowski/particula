"""
Test for update functions
"""

import numpy as np
import taichi as ti
import unittest

from particula.backend.taichi.dynamics.condensation.ti_update_condensation_masses import (
    update_transferable_mass,
    update_gas_mass,
    update_species_masses,
)

ti.init(arch=ti.cpu, default_fp=ti.f64)


@ti.kernel
def _k_update_transferable_mass(
    dt: float,
    mtr: ti.template(),      # mass_transport_rate
    scale: ti.template(),    # scaling_factor
    trans: ti.template(),    # transferable_mass (out)
):
    update_transferable_mass(dt, mtr, scale, trans)


@ti.kernel
def _k_update_gas_mass(
    gas: ti.template(),
    specie_mass: ti.template(),
    trans: ti.template(),
):
    update_gas_mass(gas, specie_mass, trans)


@ti.kernel
def _k_update_species_masses(
    specie_mass: ti.template(),
    particle_conc: ti.template(),
    trans: ti.template(),
):
    update_species_masses(specie_mass, particle_conc, trans)


class TestUpdateCondensationMasses(unittest.TestCase):
    def setUp(self):
        # dimensions
        self.n_particles, self.n_species = 7, 3

        # --- numpy reference data -------------------------------------------------
        rng = np.random.default_rng()
        self.mass_transport_rate_np = rng.random((self.n_particles, self.n_species))
        self.scaling_factor_np      = rng.random(self.n_species)
        self.time_step              = rng.random() * 0.1 + 0.01

        # transferable-mass reference
        self.transferable_ref = (
            self.mass_transport_rate_np
            * self.time_step
            * self.scaling_factor_np[None, :]
        )

        # gas mass (ensure strictly positive so clamping can be tested)
        self.gas_mass_np = rng.random(self.n_species) * 1e-3 + 1e-4
        self.gas_mass_ref = self.gas_mass_np - self.transferable_ref.sum(axis=0)
        self.gas_mass_ref = np.maximum(self.gas_mass_ref, 0.0)

        # particle-species masses
        self.species_masses_np = rng.random((self.n_particles, self.n_species))

        # particle concentration (allow some zeros to exercise branch)
        self.particle_conc_np = rng.random(self.n_particles)
        self.particle_conc_np[0] = 0.0

        species_masses_ref = self.species_masses_np.copy()
        for i in range(self.n_particles):
            for j in range(self.n_species):
                if self.particle_conc_np[i] > 0.0:
                    species_masses_ref[i, j] = (
                        self.species_masses_np[i, j] * self.particle_conc_np[i]
                        + self.transferable_ref[i, j]
                    ) / self.particle_conc_np[i]
                else:
                    species_masses_ref[i, j] = 0.0
                species_masses_ref[i, j] = max(species_masses_ref[i, j], 0.0)
        self.species_masses_ref = species_masses_ref

        # --- Taichi fields --------------------------------------------------------
        self.mtr = ti.field(dtype=float, shape=self.mass_transport_rate_np.shape)
        self.scale = ti.field(dtype=float, shape=self.scaling_factor_np.shape)
        self.trans = ti.field(dtype=float, shape=self.transferable_ref.shape)
        self.gas = ti.field(dtype=float, shape=self.gas_mass_np.shape)
        self.species_masses = ti.field(dtype=float, shape=self.species_masses_np.shape)
        self.particle_conc = ti.field(dtype=float, shape=self.particle_conc_np.shape)

        # load data into fields
        self.mtr.from_numpy(self.mass_transport_rate_np)
        self.scale.from_numpy(self.scaling_factor_np)
        self.gas.from_numpy(self.gas_mass_np)
        self.species_masses.from_numpy(self.species_masses_np)
        self.particle_conc.from_numpy(self.particle_conc_np)

    # -------------------------------------------------------------------------
    def test_transferable_mass(self):
        _k_update_transferable_mass(
            self.time_step, self.mtr, self.scale, self.trans
        )
        np.testing.assert_allclose(
            self.trans.to_numpy(), self.transferable_ref, rtol=1e-12, atol=0.0
        )

    def test_gas_mass(self):
        # first populate transferable_mass
        _k_update_transferable_mass(
            self.time_step, self.mtr, self.scale, self.trans
        )
        _k_update_gas_mass(self.gas, self.species_masses, self.trans)
        np.testing.assert_allclose(
            self.gas.to_numpy(), self.gas_mass_ref, rtol=1e-12, atol=0.0
        )

    def test_species_masses(self):
        _k_update_transferable_mass(
            self.time_step, self.mtr, self.scale, self.trans
        )
        _k_update_species_masses(
            self.species_masses, self.particle_conc, self.trans
        )
        np.testing.assert_allclose(
            self.species_masses.to_numpy(),
            self.species_masses_ref,
            rtol=1e-12,
            atol=0.0,
        )


if __name__ == "__main__":
    unittest.main()

