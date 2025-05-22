import math
import numpy as np
import taichi as ti
import unittest

from particula.backend.taichi.particles.properties.ti_particle_radius import (
    fget_particle_radius_via_masses,
)

ti.init(arch=ti.cpu)

# -----------------------------  Taichi fields  ----------------------------- #
n_particles, n_species = 1, 1
species_masses = ti.field(dtype=ti.f64, shape=(n_particles, n_species))
density = ti.field(dtype=ti.f64, shape=n_species)

# initialise with simple values → volume = 1
species_masses[0, 0] = 1.0  # mass  [kg]
density[0] = 1.0  # ρ     [kg m⁻³]


@ti.kernel
def _compute_radius() -> ti.f64:
    return fget_particle_radius_via_masses(0, species_masses, density)


# --------------------------------------------------------------------------- #
#  Unit test class using unittest                                             #
# --------------------------------------------------------------------------- #
class TestParticleRadius(unittest.TestCase):

    def test_single_particle_radius(self):
        radius_ti = _compute_radius()
        radius_ref = (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
        self.assertTrue(np.isclose(radius_ti, radius_ref, rtol=1e-6))



if __name__ == "__main__":
    unittest.main()
