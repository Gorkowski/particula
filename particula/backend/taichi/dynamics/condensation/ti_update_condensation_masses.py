"""
This module contains functions to update the condensation masses in a
particle-resolved simulation using Taichi. It includes functions to
calculate mass transport rates, update scaling factors, and adjust
transferable mass, gas mass, and species masses based on the
mass transfer dynamics.
"""

import taichi as ti


@ti.func
def update_transferable_mass(
    time_step: float,
    mass_transport_rate: ti.template(),
    scaling_factor: ti.template(),
    transferable_mass: ti.template(),
):
    """
    Update the transferable mass field based on the mass transport rate,
    time step, and scaling factor. This function calculates the mass that
    can be transferred from the gas phase to the particles for each species
    and particle, ensuring that the mass transfer respects the scaling factor
    for each species.
    """
    for p, s in ti.ndrange(
        mass_transport_rate.shape[0], mass_transport_rate.shape[1]
    ):
        transferable_mass[p, s] = (
            mass_transport_rate[p, s] * time_step * scaling_factor[s]
        )


@ti.func
def update_gas_mass(
    gas_mass: ti.template(),
    species_masses: ti.template(),
    transferable_mass: ti.template(),
):
    """
    Update the gas mass field by subtracting the total transferable mass
    for each species from the gas mass. This ensures that the gas mass
    reflects the mass available for each species after accounting for
    the mass transferred to particles.
    """
    for j in ti.ndrange(gas_mass.shape[0]):
        species_mass = 0.0
        for i in ti.ndrange(species_masses.shape[0]):
            species_mass += transferable_mass[i, j]
        gas_mass[j] -= species_mass
        gas_mass[j] = ti.max(gas_mass[j], 0.0)


@ti.func
def update_species_masses(
    species_masses: ti.template(),
    particle_concentration: ti.template(),
    transferable_mass: ti.template(),
):
    """
    Update the species masses field based on the particle concentration
    and the transferable mass. This function ensures that the species
    masses are updated correctly for each particle and species, taking
    into account the concentration of particles and the mass that can be
    transferred from the gas phase to the particles.
    """
    for i in range(species_masses.shape[0]):
        for j in range(species_masses.shape[1]):
            if particle_concentration[i] > 0.0:
                species_masses[i, j] = (
                    species_masses[i, j] * particle_concentration[i]
                    + transferable_mass[i, j]
                ) / particle_concentration[i]
            else:
                species_masses[i, j] = 0.0
            species_masses[i, j] = ti.max(species_masses[i, j], 0.0)
