"""
Mass weighted property for particles.
"""

import taichi as ti


@ti.func
def fget_mass_weighted_density_and_surface_tension(
    particle_index: int,
    species_masses: ti.template(),
    density: ti.template(),
    surface_tension: ti.template(),
) -> tuple:
    """
    Calculate the effective surface tension and density of a particle based on
    its species masses and density.

    Using the input species masses (n_particles x n_species) and the density
    and surface tension of each species (n_species), this function computes
    the effective surface tension and density of the given particle_index.
    This relies on a kernel to loop over the n_particles.
    """
    weighted_mass_sum = ti.cast(0.0, float)
    surface_tension_sum = ti.cast(0.0, float)
    density_sum = ti.cast(0.0, float)
    for species_index in range(density.shape[0]):
        mass = species_masses[particle_index, species_index]
        weighted_mass_sum += mass
        surface_tension_sum += surface_tension[species_index] * mass
        density_sum += density[species_index] * mass
    effective_surface_tension = surface_tension_sum / weighted_mass_sum
    effective_density = density_sum / weighted_mass_sum
    return (effective_surface_tension, effective_density)
