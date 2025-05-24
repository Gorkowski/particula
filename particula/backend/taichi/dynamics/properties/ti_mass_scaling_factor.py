"""
Taichi implementation of mass scaling factor update functions.
"""

import taichi as ti


@ti.func
def update_scaling_factors(
    species_masses: ti.template(),
    particle_concentration: ti.template(),
    total_requested_mass: ti.template(),
    gas_mass: ti.template(),
    mass_transport_rate: ti.template(),
    time_step: float,
    simulation_volume: float,
    scaling_factor: ti.template(),
) -> None:
    """
    Update the scaling factors based on the total requested mass and gas mass.
    This function ensures that the scaling factors are adjusted so that the
    total requested mass does not exceed the available gas mass for each
    species.
    """
    for j in range(scaling_factor.shape[0]):
        total_requested_mass[j] = 0.0
    for i, j in ti.ndrange(species_masses.shape[0], species_masses.shape[1]):
        particle_concentration[i] = 1.0 / simulation_volume
        total_requested_mass[j] += (
            mass_transport_rate[i, j] * time_step * particle_concentration[i]
        )
    for j in range(scaling_factor.shape[0]):
        scaling_factor[j] = 1.0
        if total_requested_mass[j] > gas_mass[j]:
            scaling_factor[j] = gas_mass[j] / total_requested_mass[j]


@ti.func
def update_scaling_factor_refactor2(
    mass_transport_rate: ti.template(),  # [n_particles, n_species]
    gas_mass: ti.template(),  # [n_species]
    total_requested_mass: ti.template(),  # [n_species]  (scratch, overwritten)
    scaling_factor: ti.template(),  # [n_species]  (output)
    time_step: float,
    simulation_volume: float,
) -> None:
    """
    Update per-species scaling factors so that the total mass requested by
    particles during *time_step* never exceeds the reservoir *gas_mass*.

    Arguments
    ---------
    mass_transport_rate :
        Mass-transfer rate field with shape (n_particles, n_species).
    gas_mass :
        Available gas mass for each species.
    total_requested_mass :
        Scratch buffer that will hold the integrated request per species.
    scaling_factor :
        Output scaling factor (1 → unlimited, <1 → limited by reservoir).
    time_step :
        Current Δt  [s].
    simulation_volume :
        Volume of the domain  [m³].
    """
    inv_volume = 1.0 / simulation_volume

    # ------------------------------------------------------------------ #
    # 1)  Zero the accumulator                                           #
    # ------------------------------------------------------------------ #
    for j in ti.ndrange(total_requested_mass.shape[0]):
        total_requested_mass[j] = 0.0

    # ------------------------------------------------------------------ #
    # 2)  Integrate particle demand  (parallel, atomic where needed)     #
    # ------------------------------------------------------------------ #
    for i, j in ti.ndrange(
        mass_transport_rate.shape[0], mass_transport_rate.shape[1]
    ):
        ti.atomic_add(
            total_requested_mass[j],
            mass_transport_rate[i, j] * time_step * inv_volume,
        )

    # ------------------------------------------------------------------ #
    # 3)  Limit by the available reservoir                               #
    # ------------------------------------------------------------------ #
    for j in ti.ndrange(scaling_factor.shape[0]):
        req = total_requested_mass[j]
        scaling_factor[j] = 1.0
        if req > 0.0 and req > gas_mass[j]:
            scaling_factor[j] = gas_mass[j] / req
