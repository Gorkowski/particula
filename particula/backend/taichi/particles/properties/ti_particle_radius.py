"""
This module contains functions to calculate the radius of particles based on their properties.

"""

import taichi as ti


@ti.func
def fget_particle_radius_via_masses(
    particle_index: int,
    species_masses: ti.template(),
    density: ti.template(),
) -> float:
    """
    Calculate the radius of a particle based on its species masses and density.
    """
    volume = 0.0
    for s in range(density.shape[0]):
        volume += species_masses[particle_index, s] / density[s]
    return ti.pow(3.0 * volume / (4.0 * ti.math.pi), 1.0 / 3.0)
