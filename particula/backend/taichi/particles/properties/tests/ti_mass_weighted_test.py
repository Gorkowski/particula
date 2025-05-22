"""Unit tests validating the Taichi implementation of
fget_mass_weighted_density_and_surface_tension."""

import numpy as np
import taichi as ti
import unittest

from particula.backend.taichi.particles.properties.ti_mass_weighted import (
    fget_mass_weighted_density_and_surface_tension,
)

ti.init(arch=ti.cpu, default_fp=ti.f64)


@ti.kernel
def _compute_mass_weighted(
    particle_index: int,
    species_masses: ti.template(),
    density: ti.template(),
    surface_tension: ti.template(),
) -> ti.types.vector(2, ti.f64):
    """Return (sigma_eff, rho_eff) for the given particle index."""
    sigma_eff, rho_eff = fget_mass_weighted_density_and_surface_tension(
        particle_index, species_masses, density, surface_tension
    )
    return ti.Vector([sigma_eff, rho_eff])


class TestMassWeightedProperties(unittest.TestCase):
    """Validate mass-weighted effective density and surface tension."""

    def setUp(self):
        """Create random NumPy data and copy them into Taichi fields."""
        self.n_particles, self.n_species = 8, 3

        self.species_masses_np = np.random.rand(
            self.n_particles, self.n_species
        ) * 1e-3  # kg
        self.density_np = np.random.rand(self.n_species) * 1e3  # kg m⁻³
        self.surface_tension_np = np.random.rand(self.n_species) * 0.1  # N m⁻¹

        self.species_masses = ti.field(
            dtype=float, shape=(self.n_particles, self.n_species)
        )
        self.density = ti.field(dtype=float, shape=self.n_species)
        self.surface_tension = ti.field(dtype=float, shape=self.n_species)

        self.species_masses.from_numpy(self.species_masses_np)
        self.density.from_numpy(self.density_np)
        self.surface_tension.from_numpy(self.surface_tension_np)

    def test_mass_weighted_properties(self):
        """Compare Taichi results against analytical NumPy reference."""
        sigma_ti = ti.field(dtype=float, shape=self.n_particles)
        rho_ti   = ti.field(dtype=float, shape=self.n_particles)

        for p in range(self.n_particles):
            vec = _compute_mass_weighted(
                p,
                self.species_masses,
                self.density,
                self.surface_tension,
            )
            sigma_ti[p], rho_ti[p] = vec[0], vec[1]

        mass_sum = np.sum(self.species_masses_np, axis=1, keepdims=True)
        sigma_ref = (
            np.sum(
                self.species_masses_np * self.surface_tension_np,
                axis=1,
                keepdims=True,
            )
            / mass_sum
        ).squeeze()
        rho_ref = (
            np.sum(self.species_masses_np * self.density_np, axis=1, keepdims=True)
            / mass_sum
        ).squeeze()

        np.testing.assert_allclose(
            sigma_ti.to_numpy(), sigma_ref, rtol=1e-7, atol=1e-6
        )
        np.testing.assert_allclose(
            rho_ti.to_numpy(),   rho_ref, rtol=1e-7, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
