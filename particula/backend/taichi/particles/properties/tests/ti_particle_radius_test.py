import math
import numpy as np
import taichi as ti

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


def test_get_particle_radius_via_masses():
    """Check that the Taichi helper returns the analytically expected radius."""
    radius_ti = _compute_radius()

    # analytical reference: r = (3 V / 4π)^{1/3}; here V = 1
    radius_ref = (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)

    assert np.isclose(radius_ti, radius_ref, rtol=1e-6)
