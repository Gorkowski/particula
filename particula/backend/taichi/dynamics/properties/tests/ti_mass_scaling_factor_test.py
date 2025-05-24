"""
Test for mass scaling factor in Taichi dynamics properties.
"""

import numpy as np
import taichi as ti
import unittest
from particula.backend.taichi.dynamics.properties.ti_mass_scaling_factor import (
    update_scaling_factors,
    update_scaling_factor_refactor2,
)

ti.init(arch=ti.cpu, default_fp=ti.f64)


@ti.kernel
def _k_update_scaling_factors(
    species_masses: ti.template(),
    particle_concentration: ti.template(),
    total_requested_mass: ti.template(),
    gas_mass: ti.template(),
    mass_transport_rate: ti.template(),
    dt: float,
    volume: float,
    scaling_factor: ti.template(),
):
    update_scaling_factors(
        species_masses,
        particle_concentration,
        total_requested_mass,
        gas_mass,
        mass_transport_rate,
        dt,
        volume,
        scaling_factor,
    )


@ti.kernel
def _k_update_scaling_factor_refactor2(
    mass_transport_rate: ti.template(),
    gas_mass: ti.template(),
    total_requested_mass: ti.template(),
    scaling_factor: ti.template(),
    dt: float,
    volume: float,
):
    update_scaling_factor_refactor2(
        mass_transport_rate,
        gas_mass,
        total_requested_mass,
        scaling_factor,
        dt,
        volume,
    )

class TestMassScalingFactor(unittest.TestCase):
    def setUp(self):
        # problem size
        self.n_particles, self.n_species = 12, 5

        # numpy reference data
        self.species_masses_np = np.random.rand(self.n_particles, self.n_species)
        self.mass_transport_rate_np = np.random.rand(
            self.n_particles, self.n_species
        )
        self.gas_mass_np = np.random.rand(self.n_species) + 1e-12  # avoid zeros

        # scalar params
        self.dt = 0.3
        self.volume = 2.5

        # taichi fields
        self.species_masses = ti.field(dtype=float, shape=self.species_masses_np.shape)
        self.mass_transport_rate = ti.field(
            dtype=float, shape=self.mass_transport_rate_np.shape
        )
        self.gas_mass = ti.field(dtype=float, shape=self.gas_mass_np.shape)

        self.particle_concentration = ti.field(dtype=float, shape=self.n_particles)
        self.req_mass_1 = ti.field(dtype=float, shape=self.n_species)
        self.req_mass_2 = ti.field(dtype=float, shape=self.n_species)

        self.scale_1 = ti.field(dtype=float, shape=self.n_species)
        self.scale_2 = ti.field(dtype=float, shape=self.n_species)

        # push data to fields
        self.species_masses.from_numpy(self.species_masses_np)
        self.mass_transport_rate.from_numpy(self.mass_transport_rate_np)
        self.gas_mass.from_numpy(self.gas_mass_np)

    def test_same_scaling_factors(self):
        # execute both versions
        _k_update_scaling_factors(
            self.species_masses,
            self.particle_concentration,
            self.req_mass_1,
            self.gas_mass,
            self.mass_transport_rate,
            self.dt,
            self.volume,
            self.scale_1,
        )

        _k_update_scaling_factor_refactor2(
            self.mass_transport_rate,
            self.gas_mass,
            self.req_mass_2,
            self.scale_2,
            self.dt,
            self.volume,
        )

        # compare
        np.testing.assert_allclose(
            self.scale_1.to_numpy(), self.scale_2.to_numpy(), rtol=1e-12, atol=0.0
        )


if __name__ == "__main__":
    unittest.main()
