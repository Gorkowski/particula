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
    species_masses: ti.template(),
    density: ti.template(),
) -> ti.f64:
    return fget_particle_radius_via_masses(0, species_masses, density)


# --------------------------------------------------------------------------- #
#  Unit test class using unittest                                             #
# --------------------------------------------------------------------------- #
class TestParticleRadius(unittest.TestCase):

    def setUp(self):
        n_particles, n_species = 1, 1
        self.species_masses = ti.field(dtype=ti.f64, shape=(n_particles, n_species))
        self.density = ti.field(dtype=ti.f64, shape=n_species)

        # simple values → particle volume = 1 m³
        self.species_masses[0, 0] = 1.0   # kg
        self.density[0] = 1.0             # kg m⁻³

    def test_single_particle_radius(self):
        radius_ti = _compute_radius(self.species_masses, self.density)
        radius_ref = (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
        self.assertTrue(np.isclose(radius_ti, radius_ref, rtol=1e-6))



if __name__ == "__main__":
    unittest.main()
