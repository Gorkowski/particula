import math
import numpy as np
import taichi as ti
import unittest

from particula.backend.taichi.particles.properties.ti_particle_radius import (
    fget_particle_radius_via_masses,
)

ti.init(arch=ti.cpu)

@ti.kernel
def _compute_radius(
    particle_index: int,
    species_masses: ti.template(),
    density: ti.template(),
) -> ti.f64:
    return fget_particle_radius_via_masses(particle_index, species_masses, density)


class TestParticleRadiusSingle(unittest.TestCase):

    def setUp(self):
        n_particles, n_species = 5, 3
        self.species_masses = ti.field(dtype=ti.f64, shape=(n_particles, n_species))
        self.density = ti.field(dtype=ti.f64, shape=n_species)

        # simple values → particle volume = 1 m³
        self.species_masses_np = np.random.rand(n_particles, n_species) * 1e-3
        self.density_np = np.random.rand(n_species) * 1e3
        self.species_masses.from_numpy(self.species_masses_np)
        self.density.from_numpy(self.density_np)

    def test_mass_matrix(self):
        radius_ti = ti.field(dtype=ti.f64, shape=self.species_masses.shape[0])
        for i in range(self.species_masses.shape[0]):
            radius_ti[i] = _compute_radius(
                i,  # particle index
                self.species_masses,
                self.density
            )

        volume = np.sum(self.species_masses_np / self.density_np, axis=1)
        radius_ref = np.cbrt(3.0 * volume / (4.0 * np.pi))
        np.testing.assert_allclose(
            radius_ti.to_numpy(), radius_ref, rtol=1e-7, atol=1e-6
        )




if __name__ == "__main__":
    unittest.main()
