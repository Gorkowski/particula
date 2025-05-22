import math
import numpy as np
import taichi as ti
import unittest

from particula.backend.taichi.particles.properties.ti_particle_radius import (
    get_particle_radius_via_masses,
)

ti.init(arch=ti.cpu, default_fp=64)

# -----------------------------  Taichi fields  ----------------------------- #
n_particles, n_species = 1, 1
species_masses = ti.field(dtype=ti.f64, shape=(n_particles, n_species))
density = ti.field(dtype=ti.f64, shape=n_species)

# initialise with simple values → volume = 1
species_masses[0, 0] = 1.0  # mass  [kg]
density[0] = 1.0  # ρ     [kg m⁻³]


@ti.kernel
def _compute_radius() -> ti.f64:
    return get_particle_radius_via_masses(0, species_masses, density)


# --------------------------------------------------------------------------- #
#  Unit test class using unittest                                             #
# --------------------------------------------------------------------------- #
class TestParticleRadius(unittest.TestCase):

    def test_single_particle_radius(self):
        radius_ti = _compute_radius()
        radius_ref = (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
        self.assertTrue(np.isclose(radius_ti, radius_ref, rtol=1e-6))

    def test_matrix_particle_radii(self):
        _compute_radii()
        radii_ti = radii_field.to_numpy()
        self.assertTrue(np.allclose(radii_ti, radii_ref, rtol=1e-6))


 # --------------------------------------------------------------------------- #
 #  problem size and reference data (choose values that give easy radii)       #
 # --------------------------------------------------------------------------- #
 n_particles, n_species = 3, 2
                                                                                                                                                                                                                                                    
 # numpy helpers for reference calculations
 species_masses_np = np.array(
     [
         [1.0, 1.0],   # particle 0  → V = 1/ρ₀ + 1/ρ₁
         [8.0, 0.0],   # particle 1  → V = 8/ρ₀
         [0.0, 27.0],  # particle 2  → V = 27/ρ₁
     ],
     dtype=np.float64,
 )
 density_np = np.array([1.0, 2.0], dtype=np.float64)  # ρ₀ = 1, ρ₁ = 2
                                                                                                                                                                                                                                                    
 # expected volumes / radii
 volumes = (species_masses_np / density_np).sum(axis=1)
 radii_ref = (3.0 * volumes / (4.0 * math.pi)) ** (1.0 / 3.0)
                                                                                                                                                                                                                                                    
 # --------------------------------------------------------------------------- #
 #  Taichi fields                                                              #
 # --------------------------------------------------------------------------- #
 species_masses = ti.field(dtype=ti.f64, shape=(n_particles, n_species))
 density        = ti.field(dtype=ti.f64, shape=n_species)
 radii_field    = ti.field(dtype=ti.f64, shape=n_particles)
                                                                                                                                                                                                                                                    
 # copy data into Taichi fields (plain Python loops to avoid extra helpers)
 for i, j in np.ndindex(species_masses_np.shape):
     species_masses[i, j] = species_masses_np[i, j]
 for j in range(n_species):
     density[j] = density_np[j]
                                                                                                                                                                                                                                                    
                                                                                                                                                                                                                                                    
 @ti.kernel
 def _compute_radii():
     for p in range(n_particles):
         radii_field[p] = get_particle_radius_via_masses(p, species_masses, density)
                                                                                                                                                                                                                                                    
                                                                                                                                                                                                                                                    
 def test_get_particle_radius_via_masses_matrix():
     """Check radii for a particles × species matrix."""
     _compute_radii()
     radii_ti = radii_field.to_numpy()
     assert np.allclose(radii_ti, radii_ref, rtol=1e-6)

 if __name__ == "__main__":
     unittest.main()
