"""Unit tests validating the Taichi implementation of particle-radius
calculation (fget_particle_radius_via_masses)."""

import numpy as np
import taichi as ti
import unittest

from particula.backend.taichi.particles.properties.ti_particle_radius import (
    fget_particle_radius_via_masses,
)



ti.init(arch=ti.cpu, default_fp=ti.f64)


@ti.kernel
def _compute_radius(
    particle_index: int,
    species_masses: ti.template(),
    density: ti.template(),
) -> ti.f64:
    """Return particle radius [m] for the given index using Taichi."""
    return fget_particle_radius_via_masses(
        particle_index, species_masses, density
    )


class TestParticleRadius(unittest.TestCase):
    """Test suite for fget_particle_radius_via_masses (matrix input)."""

    def setUp(self):
        """Create random NumPy data and copy it into Taichi fields."""
        n_particles, n_species = 8, 3
        self.species_masses = ti.field(
            dtype=float, shape=(n_particles, n_species)
        )
        self.density = ti.field(dtype=float, shape=n_species)

        # simple values → particle volume = 1 m³
        self.species_masses_np = np.random.rand(n_particles, n_species) * 1e-3
        self.density_np = np.random.rand(n_species) * 1e3
        self.species_masses.from_numpy(self.species_masses_np)
        self.density.from_numpy(self.density_np)

    def test_mass_matrix(self):
        """Check Taichi radii against analytical reference for all particles."""
        radius_ti = ti.field(dtype=float, shape=self.species_masses.shape[0])
        for i in range(self.species_masses.shape[0]):
            radius_ti[i] = _compute_radius(
                i, self.species_masses, self.density
            )

        volume = np.sum(self.species_masses_np / self.density_np, axis=1)
        radius_ref = np.cbrt(3.0 * volume / (4.0 * np.pi))
        np.testing.assert_allclose(
            radius_ti.to_numpy(), radius_ref, rtol=1e-7, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
