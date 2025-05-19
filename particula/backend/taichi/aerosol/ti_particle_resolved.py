"""
Integrated Aerosol Dynamics and Particle-Resolved Simulation Representation
"""

import taichi as ti
import numpy as np
import particula as par
import particula.backend.taichi.gas.properties as gas_properties
import particula.backend.taichi.particles.properties as particle_properties
import particula.backend.taichi.dynamics.condensation as condensation
from particula.backend.benchmark import get_function_benchmark

np_type = np.float64
ti.init(arch=ti.cpu, default_fp=ti.f64, default_ip=ti.i32, debug=True)

GAS_CONSTANT = par.util.constants.GAS_CONSTANT

# particle resolved data, 100 particles, 10 species
particle_count = 20_000
species_count = 10
input_species_masses = np.random.rand(particle_count, species_count).astype(
    np_type
)
input_density = np.random.rand(species_count).astype(np_type)
input_molar_mass = np.abs(np.random.rand(species_count).astype(np_type))
input_pure_vapor_pressure = np.abs(
    np.random.rand(species_count).astype(np_type)
)
input_vapor_concentration = np.abs(
    np.random.rand(species_count).astype(np_type)
)
input_kappa_value = np.abs(np.random.rand(species_count).astype(np_type))
input_surface_tension = np.abs(np.random.rand(species_count).astype(np_type))
input_temperature = 298.15  # K
input_pressure = 101325.0  # Pa
input_mass_accommodation = 0.5
input_dynamic_viscosity = par.gas.get_dynamic_viscosity(temperature=input_temperature)
input_diffusion_coefficient = 2.0e-5  # m^2/s
input_time_step = 10  # seconds
input_simulation_volume = 1.0e-6  # m^3


# available gas-phase mass [kg] per species  (positive values)
input_gas_mass = np.abs(np.random.rand(species_count).astype(np_type))
# particle number concentration [#/m³] for every particle (use 1.0 for now)
input_particle_concentration = np.ones(particle_count, dtype=np_type)


# create fiels that will be internally used
temperature = ti.static(input_temperature)
pressure = ti.static(input_pressure)
mass_accommodation = ti.static(input_mass_accommodation)
dynamic_viscosity = ti.static(input_dynamic_viscosity)
diffusion_coefficient = ti.static(input_diffusion_coefficient)
time_step = ti.static(input_time_step)
simulation_volume = ti.static(input_simulation_volume)


# apply input species masses to the field
species_masses = ti.field(
    float, shape=(particle_count, species_count), name="species_masses"
)
species_masses.from_numpy(input_species_masses)

# create density field
density = ti.field(float, shape=(species_count,), name="density")
density.from_numpy(input_density)
# create molar mass field
molar_mass = ti.field(float, shape=(species_count,), name="molar_mass")
molar_mass.from_numpy(input_molar_mass)
# create pure vapor pressure field
pure_vapor_pressure = ti.field(
    float, shape=(species_count,), name="pure_vapor_pressure"
)
pure_vapor_pressure.from_numpy(input_pure_vapor_pressure)
# create vapor concentration field
vapor_concentration = ti.field(
    float, shape=(species_count,), name="vapor_concentration"
)
vapor_concentration.from_numpy(input_vapor_concentration)
# create kappa value field
kappa_value = ti.field(float, shape=(species_count,), name="kappa_value")
kappa_value.from_numpy(input_kappa_value)
# create surface tension field
surface_tension = ti.field(
    float, shape=(species_count,), name="surface_tension"
)
surface_tension.from_numpy(input_surface_tension)

# create gas mass field
gas_mass = ti.field(float, shape=(species_count,), name="gas_mass")
gas_mass.from_numpy(input_gas_mass)

# create particle concentration field
particle_concentration = ti.field(
    float, shape=(particle_count,), name="particle_concentration"
)
particle_concentration.from_numpy(input_particle_concentration)


# temperay fields
radius = ti.field(float, shape=(particle_count,), name="radii")
mass_transport_rate = ti.field(
    float, shape=(particle_count, species_count), name="mass_transport_rate"
)
first_order_coefficient = ti.field(
    float,
    shape=(particle_count, species_count),
    name="first_order_coefficient",
)
partial_pressure = ti.field(
    float, shape=(particle_count, species_count), name="partial_pressure"
)
pressure_delta = ti.field(
    float, shape=(particle_count, species_count), name="pressure_delta"
)
kelvin_term = ti.field(
    float, shape=(particle_count, species_count), name="kelvin_term"
)
kelvin_radius = ti.field(
    float, shape=(particle_count, species_count), name="kelvin_radius"
)
transferable_mass = ti.field(
    float, shape=(particle_count, species_count), name="transferable_mass"
)
total_requested_mass = ti.field(
    float, shape=(species_count,), name="total_requested_mass"
)
scaling_factor = ti.field(float, shape=(species_count,), name="scaling_factor")


@ti.func
def update_radius(p_index: int):
    """
    Compute the radius of a single particle.

    Parameters
    ----------
    p_index : int
        Index of the particle whose radius is to be calculated.

    Returns
    -------
    float
        Particle radius (same units as implied by density and mass –
        typically metres if SI inputs are supplied).

    Notes
    -----
    The radius is obtained by first converting each species mass to its
    partial volume using the species density, summing those volumes, and
    then inverting the sphere-volume relation:

        r = (3 V / 4π)^(1/3)
    """
    volume = 0.0
    for j in ti.ndrange(species_count):
        volume += species_masses[p_index, j] / density[j]
    radius[p_index] = ti.pow(3.0 * volume / (4.0 * ti.math.pi), (1.0 / 3.0))


@ti.func
def update_first_order_coefficient(p_index: int):
    """
    Update the first-order mass transport term.

    """
    for j in ti.ndrange(species_count):
        first_order_coefficient[p_index, j] = (
            condensation.fget_first_order_mass_transport_via_system_state(
                particle_radius=radius[p_index],
                molar_mass=molar_mass[j],
                mass_accommodation=mass_accommodation,
                temperature=temperature,
                pressure=pressure,
                dynamic_viscosity=dynamic_viscosity,
                diffusion_coefficient=diffusion_coefficient,
            )
        )


@ti.func
def fget_weighted_average(
    value_field: ti.template(),  # 1-D field (density or surface_tension)
    weight_field: ti.template(),  # 2-D field (species_masses)
    p_index: int,  # particle index whose weights we use
) -> float:
    """
    Compute Σ_j value_field[j] * weight_field[p_index, j] / Σ_j weight_field[p_index, j].

    Returns 0.0 when the weight sum is zero.
    """
    w_sum = 0.0
    v_sum = 0.0
    for j in ti.ndrange(species_count):
        w = weight_field[p_index, j]
        v_sum += value_field[j] * w
        w_sum += w
    return ti.select(w_sum > 0.0, v_sum / w_sum, 0.0)


@ti.func
def update_kelvin_term(p_index: int):
    """
    Update the Kelvin term for a particle.

    This function is a placeholder for the actual implementation of
    the Kelvin term update. It currently does nothing.
    """
    effective_surface_tension = fget_weighted_average(
        surface_tension, species_masses, p_index
    )
    effective_density = fget_weighted_average(density, species_masses, p_index)
    for j in ti.ndrange(species_count):
        kelvin_radius[p_index, j] = particle_properties.fget_kelvin_radius(
            effective_surface_tension,
            effective_density,
            molar_mass[j],
            temperature,
        )
        kelvin_term[p_index, j] = particle_properties.fget_kelvin_term(
            r_p=radius[p_index],
            r_k=kelvin_radius[p_index, j],
        )


@ti.func
def update_partial_pressure(p_index: int):
    """
    Update the partial pressure for a particle.

    This function is a placeholder for the actual implementation of
    the partial pressure update. It currently does nothing.
    """
    for j in ti.ndrange(species_count):
        partial_pressure[p_index, j] = gas_properties.fget_partial_pressure(
            concentration=vapor_concentration[j],
            molar_mass=molar_mass[j],
            temperature=temperature,
        )
        pressure_delta[p_index, j] = (
            particle_properties.fget_partial_pressure_delta(
                partial_pressure_gas=partial_pressure[p_index, j],
                partial_pressure_particle=partial_pressure[p_index, j],
                kelvin_term=kelvin_term[p_index, j],
            )
        )


@ti.func
def update_mass_transport_rate(p_index: int):
    """
    Update the mass transport rate for a particle.

    This function is a placeholder for the actual implementation of
    the mass transport rate update. It currently does nothing.
    """
    for j in ti.ndrange(species_count):
        mass_transport_rate[p_index, j] = (
            first_order_coefficient[p_index, j]
            * pressure_delta[p_index, j]
            * molar_mass[j]
            / (GAS_CONSTANT * temperature)
        )


@ti.func
def update_transferable_mass(p_index: int, time_step: float):
    """
    Update `transferable_mass` for particle `p_index` following
    the numpy reference algorithm.
    """
    for j in ti.ndrange(species_count):
        # Step-1 & 4 : scaled mass_to_change
        mass_to_change = (
            mass_transport_rate[p_index, j]
            * time_step
            * particle_concentration[p_index]
            * scaling_factor[j]
        )

        # Step-5 : condensation limited by gas mass
        condensible_mass = ti.min(ti.abs(mass_to_change), gas_mass[j])

        # Step-6 : evaporation limited by particle mass
        evaporative_mass = ti.max(
            mass_to_change,
            -species_masses[p_index, j] * particle_concentration[p_index],
        )

        # Step-7 : final transferable mass
        transferable_mass[p_index, j] = ti.select(
            mass_to_change > 0.0,  # condensation vs evaporation
            condensible_mass,
            evaporative_mass,
        )


@ti.func
def update_scaling_factors(time_step: float):
    """
    Internal routine that fills `total_requested_mass` and `scaling_factor`.

    This is the exact logic that used to live inside
    `calculate_scaling_factors`.
    """
    # 1. reset accumulator
    for j in ti.ndrange(species_count):
        total_requested_mass[j] = 0.0

    # 2. total requested mass per species
    for i, j in ti.ndrange(particle_count, species_count):
        particle_concentration[i] = (
            1.0 / simulation_volume
        )  # particle-resolved
        total_requested_mass[j] += (
            mass_transport_rate[i, j] * time_step * particle_concentration[i]
        )

    # 3. build scaling factors
    for j in ti.ndrange(species_count):
        scaling_factor[j] = 1.0
        if total_requested_mass[j] > gas_mass[j]:
            scaling_factor[j] = gas_mass[j] / total_requested_mass[j]


@ti.func
def update_gas_mass():
    """
    Update the gas mass field based on the input gas mass.

    """
    for j in ti.ndrange(species_count):
        species_mass = 0.0
        for i in ti.ndrange(particle_count):
            species_mass += transferable_mass[i, j]
        # Update the gas mass for species j
        gas_mass[j] -= species_mass
        # Ensure gas mass does not go
        gas_mass[j] = ti.max(gas_mass[j], 0.0)


@ti.func
def update_species_masses():
    """
    Update the species masses field based on the input species masses.

    """
    for i in ti.ndrange(particle_count):
        for j in ti.ndrange(species_count):
            # Update the species mass for particle i and species j
            if particle_concentration[i] > 0.0:
                species_masses[i, j] = (
                    species_masses[i, j] * particle_concentration[i]
                    + transferable_mass[i, j]
                ) / particle_concentration[i]
            else:
                species_masses[i, j] = 0.0
            # Ensure species mass does not go negative
            species_masses[i, j] = ti.max(species_masses[i, j], 0.0)

@ti.func
def simulation_step():
    """
    Perform a simulation step.

    This function is a placeholder for the actual implementation of
    the simulation step. It currently does nothing.
    """
    for i in ti.ndrange(particle_count):
        update_radius(i)
        update_first_order_coefficient(i)
        update_kelvin_term(i)
        update_partial_pressure(i)
        update_mass_transport_rate(i)

    update_scaling_factors(time_step)
    for i in ti.ndrange(particle_count):
        update_transferable_mass(i, time_step)
    update_gas_mass()
    update_species_masses()


@ti.kernel
def fused_step():
    # --- species-independent prep (runs once per particle) -------------
    for p in ti.ndrange(particle_count):
        # a) radius ------------------------------------------------------
        volume = 0.0
        for s in ti.ndrange(species_count):
            volume += species_masses[p, s] / density[s]
        r_p = ti.pow(3.0 * volume / (4.0 * ti.math.pi), 1.0 / 3.0)
        radius[p] = r_p

        # b) effective bulk properties (weighted averages) ---------------
        w_mass = 0.0
        sig_sum = 0.0
        rho_sum = 0.0
        for s in ti.ndrange(species_count):
            w = species_masses[p, s]
            w_mass += w
            sig_sum += surface_tension[s] * w
            rho_sum += density[s] * w
        sig_eff = sig_sum / w_mass
        rho_eff = rho_sum / w_mass

        # ---------- per-species loop (still inside the same kernel) -----
        for s in ti.ndrange(species_count):
            # first-order coefficient
            k1 = condensation.fget_first_order_mass_transport_via_system_state(
                particle_radius=r_p,
                molar_mass=molar_mass[s],
                mass_accommodation=mass_accommodation,
                temperature=temperature,
                pressure=pressure,
                dynamic_viscosity=dynamic_viscosity,
                diffusion_coefficient=diffusion_coefficient,
            )
            first_order_coefficient[p, s] = k1

            # Kelvin term -------------------------------------------------
            r_k = particle_properties.fget_kelvin_radius(
                sig_eff, rho_eff, molar_mass[s], temperature
            )
            kel = particle_properties.fget_kelvin_term(r_p, r_k)
            kelvin_term[p, s] = kel

            # vapour & ΔP -----------------------------------------------
            p_g = gas_properties.fget_partial_pressure(
                vapor_concentration[s],
                molar_mass[s],
                temperature,
            )
            dP = particle_properties.fget_partial_pressure_delta(p_g, p_g, kel)
            pressure_delta[p, s] = dP

            # mass-flux ---------------------------------------------------
            mass_transport_rate[p, s] = (
                k1
                * dP
                * molar_mass[s]
                / (par.util.constants.GAS_CONSTANT * temperature)
            )
    update_scaling_factors(time_step)
    for i in ti.ndrange(particle_count):
        update_transferable_mass(i, time_step)
    update_gas_mass()
    update_species_masses()


## kernels for testing functionality
@ti.kernel
def calculate_radius():
    """
    Taichi kernel that populates `radii` for every particle.

    Iterates over all particles, calls `get_radius` for each, and stores
    the resulting radii in the dedicated Taichi field.
    """
    for i in ti.ndrange(particle_count):
        update_radius(i)


@ti.kernel
def calculate_mass_transport_rate():
    """
    Taichi kernel that populates `mass_transport_rate` for every particle.

    Iterates over all particles, calls `get_mass_transport_rate` for each,
    and stores the resulting mass transport rates in the dedicated Taichi field.
    """
    for i in ti.ndrange(particle_count):
        update_mass_transport_rate(i)


@ti.kernel
def calculate_kelvin_term():
    """
    Taichi kernel that populates `kelvin_term` for every particle.

    Iterates over all particles, calls `get_kelvin_term` for each,
    and stores the resulting Kelvin terms in the dedicated Taichi field.
    """
    for i in ti.ndrange(particle_count):
        update_kelvin_term(i)


@ti.kernel
def calculate_pressure_delta():
    """
    Taichi kernel that populates `pressure_delta` for every particle.

    Iterates over all particles, calls `get_pressure_delta` for each,
    and stores the resulting pressure deltas in the dedicated Taichi field.
    """
    for i in ti.ndrange(particle_count):
        update_partial_pressure(i)


@ti.kernel
def calculate_first_order_coefficient():
    """
    Taichi kernel that populates `first_order_coefficient` for every particle.

    Iterates over all particles, calls `get_first_order_mass_transport_via_system_state`
    for each, and stores the resulting first-order coefficients in the dedicated Taichi field.
    """
    for i in ti.ndrange(particle_count):
        update_first_order_coefficient(i)


@ti.kernel
def calculate_scaling_factors(time_step: float):
    """
    Populate `scaling_factor` for every gas species.

    Delegates the actual computation to `_update_scaling_factors`.
    """
    update_scaling_factors(time_step)


@ti.kernel
def calculate_transferable_mass(time_step: float):
    """
    Populates `transferable_mass` for every particle after
    scaling factors have been computed.
    """
    for i in ti.ndrange(particle_count):
        update_transferable_mass(i, time_step)

@ti.kernel
def calculate_gas_mass():
    """
    Populates `gas_mass` for every gas species.

    Delegates the actual computation to `_update_gas_mass`.
    """
    update_gas_mass()

@ti.kernel
def calculate_species_masses():
    """
    Populates `species_masses` for every particle.

    Delegates the actual computation to `_update_species_masses`.
    """
    update_species_masses()

@ti.kernel
def calculate_simulation_step():
    """
    Perform a simulation step.

    This function is a placeholder for the actual implementation of
    the simulation step. It currently does nothing.
    """
    simulation_step()


if __name__ == "__main__":
    # calculate_radius()
    # calculate_first_order_coefficient()
    # calculate_kelvin_term()
    # calculate_pressure_delta()
    # calculate_mass_transport_rate()
    # calculate_scaling_factors(time_step)  # NEW
    # calculate_transferable_mass(time_step)  # NEW
    # calculate_gas_mass()  # NEW
    # calculate_species_masses()  # NEW
    calculate_simulation_step()  # NEW
    fused_step()
    radius_np = radius.to_numpy()
    first_order_coefficient_np = first_order_coefficient.to_numpy()
    kelvin_radius_np = kelvin_radius.to_numpy()
    kelvin_term_np = kelvin_term.to_numpy()
    partial_pressure_np = partial_pressure.to_numpy()
    pressure_delta_np = pressure_delta.to_numpy()
    mass_transport_rate_np = mass_transport_rate.to_numpy()
    transferable_mass_np = transferable_mass.to_numpy()
    scaling_factor_np = scaling_factor.to_numpy()
    total_requested_mass_np = total_requested_mass.to_numpy()
    # for i, r in enumerate(radius.to_numpy()):
    #     if i % 10 == 0:
    #         # print(f"Particle {i}: Radius = {r}, Mass Transport Rate = {first_order_coefficient_np[i]}")
    #         # print(f"Kelvin Radius = {kelvin_radius_np[i]}, Kelvin Term = {kelvin_term_np[i]}")
    #         # print(f"Partial Pressure = {partial_pressure_np[i]}, Pressure Delta = {pressure_delta_np[i]}")
    #         # print(f"Mass Transport Rate = {mass_transport_rate_np[i]}")
    #         print(f"Transferable Mass = {transferable_mass_np[i]}")
    # print(f"Gas Mass = {gas_mass.to_numpy()}")
    results = get_function_benchmark(
        lambda: calculate_simulation_step(),
        ops_per_call=particle_count * species_count,
        max_run_time_s=2,
    )
    print(f"func version")
    print(results["report"])
    
    fused_results = get_function_benchmark(
        lambda: fused_step(),
        ops_per_call=particle_count * species_count,
        max_run_time_s=2,
    )
    print(f"fused version")
    print(fused_results["report"])
    # print(f"Species Masses = {species_masses.to_numpy()}")
    print("All calculations completed successfully.")
