"""
Integrated Aerosol Dynamics and Particle-Resolved Simulation Representation
"""

import taichi as ti
import numpy as np
import particula as par
import particula.backend.taichi.gas.properties as gas_properties
import particula.backend.taichi.particles.properties as particle_properties
import particula.backend.taichi.dynamics.condensation as condensation

np_type = np.float32
ti.init(arch=ti.cpu, default_fp=ti.f32, default_ip=ti.i32, debug=True)

GAS_CONSTANT = par.util.constants.GAS_CONSTANT

# particle resolved data, 100 particles, 10 species
particle_count = 100
species_count = 10
input_species_masses = np.random.rand(particle_count, species_count).astype(np_type)
input_density = np.random.rand(species_count).astype(np_type)
input_molar_mass = np.abs(np.random.rand(species_count).astype(np_type))
input_pure_vapor_pressure = np.abs(np.random.rand(species_count).astype(np_type))
input_vapor_concentration = np.abs(np.random.rand(species_count).astype(np_type))
input_kappa_value = np.abs(np.random.rand(species_count).astype(np_type))
input_surface_tension = np.abs(np.random.rand(species_count).astype(np_type))
temperature = 298.15  # K
pressure = 101325.0  # Pa
mass_accommodation = 0.5
dynamic_viscosity = par.gas.get_dynamic_viscosity(temperature=temperature)
diffusion_coefficient = 2.0e-5  # m^2/s
time_step = 0.1 # seconds
simulation_volume = 1.0e-6  # m^3

# available gas-phase mass [kg] per species  (positive values)
input_gas_mass = np.abs(np.random.rand(species_count).astype(np_type))
# particle number concentration [#/m³] for every particle (use 1.0 for now)
input_particle_concentration = np.ones(particle_count, dtype=np_type)


# create fiels that will be internally used

# apply input species masses to the field
species_masses = ti.field(float, shape=(particle_count, species_count), name="species_masses")
species_masses.from_numpy(input_species_masses)

# create density field
density = ti.field(float, shape=(species_count,), name="density")
density.from_numpy(input_density)
# create molar mass field
molar_mass = ti.field(float, shape=(species_count,), name="molar_mass")
molar_mass.from_numpy(input_molar_mass)
# create pure vapor pressure field
pure_vapor_pressure = ti.field(float, shape=(species_count,), name="pure_vapor_pressure")
pure_vapor_pressure.from_numpy(input_pure_vapor_pressure)
# create vapor concentration field
vapor_concentration = ti.field(float, shape=(species_count,), name="vapor_concentration")
vapor_concentration.from_numpy(input_vapor_concentration)
# create kappa value field
kappa_value = ti.field(float, shape=(species_count,), name="kappa_value")
kappa_value.from_numpy(input_kappa_value)
# create surface tension field
surface_tension = ti.field(float, shape=(species_count,), name="surface_tension")
surface_tension.from_numpy(input_surface_tension)

# create gas mass field
gas_mass = ti.field(float, shape=(species_count,), name="gas_mass")
gas_mass.from_numpy(input_gas_mass)

# create particle concentration field
particle_concentration = ti.field(float, shape=(particle_count,),
                                  name="particle_concentration")
particle_concentration.from_numpy(input_particle_concentration)

# temporary helpers
total_requested_mass = ti.field(float, shape=(species_count,),
                                name="total_requested_mass")
scaling_factor = ti.field(float, shape=(species_count,),
                          name="scaling_factor")

# intermediate fields
radius = ti.field(float, shape=(particle_count,), name="radii")
mass_transport_rate = ti.field(float, shape=(particle_count, species_count), name="mass_transport_rate")
first_order_coefficient = ti.field(float, shape=(particle_count, species_count), name="first_order_coefficient")
partial_pressure = ti.field(float, shape=(particle_count, species_count), name="partial_pressure")
pressure_delta = ti.field(float, shape=(particle_count, species_count), name="pressure_delta")
kelvin_term = ti.field(float, shape=(particle_count, species_count), name="kelvin_term")
kelvin_radius = ti.field(float, shape=(particle_count, species_count), name="kelvin_radius")
transferable_mass = ti.field(float, shape=(particle_count, species_count), name="transferable_mass")



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
    for j in range(species_count):
        volume += species_masses[p_index, j] / density[j]
    radius[p_index] = ti.pow(3.0 * volume / (4.0 * ti.math.pi), (1.0 / 3.0))


@ti.func
def update_first_order_coefficient(p_index: int):
    """
    Update the first-order mass transport term.

    This function is a placeholder for the actual implementation of
    first-order mass transport update. It currently does nothing.
    """
    for j in range(species_count):
        first_order_coefficient[p_index, j] = condensation.fget_first_order_mass_transport_via_system_state(
            particle_radius=radius[p_index],
            molar_mass=molar_mass[j],
            mass_accommodation=mass_accommodation,
            temperature=temperature,
            pressure=pressure,
            dynamic_viscosity=dynamic_viscosity,
            diffusion_coefficient=diffusion_coefficient,
        )


@ti.func
def fget_weighted_average(
    value_field: ti.template(),     # 1-D field (density or surface_tension)
    weight_field: ti.template(),    # 2-D field (species_masses)
    p_index: int                    # particle index whose weights we use
) -> float:
    """
    Compute Σ_j value_field[j] * weight_field[p_index, j] / Σ_j weight_field[p_index, j].

    Returns 0.0 when the weight sum is zero.
    """
    w_sum = 0.0
    v_sum = 0.0
    for j in range(species_count):
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
    effective_density = fget_weighted_average(
        density, species_masses, p_index
    )
    for j in range(species_count):
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
    for j in range(species_count):
        partial_pressure[p_index, j] = gas_properties.fget_partial_pressure(
            concentration=vapor_concentration[j],
            molar_mass=molar_mass[j],
            temperature=temperature,
        )
        pressure_delta[p_index, j] = particle_properties.fget_partial_pressure_delta(
            partial_pressure_gas=partial_pressure[p_index, j],
            partial_pressure_particle=partial_pressure[p_index, j],
            kelvin_term=kelvin_term[p_index, j],
        )

@ti.func
def update_mass_transport_rate(p_index: int):
    """
    Update the mass transport rate for a particle.

    This function is a placeholder for the actual implementation of
    the mass transport rate update. It currently does nothing.
    """
    for j in range(species_count):
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
    for j in range(species_count):
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


## kernels for testing functionality
@ti.kernel
def calculate_radius():
    """
    Taichi kernel that populates `radii` for every particle.

    Iterates over all particles, calls `get_radius` for each, and stores
    the resulting radii in the dedicated Taichi field.
    """
    for i in range(particle_count):
        update_radius(i)

@ti.kernel
def calculate_mass_transport_rate():
    """
    Taichi kernel that populates `mass_transport_rate` for every particle.

    Iterates over all particles, calls `get_mass_transport_rate` for each,
    and stores the resulting mass transport rates in the dedicated Taichi field.
    """
    for i in range(particle_count):
        update_mass_transport_rate(i)

@ti.kernel
def calculate_kelvin_term():
    """
    Taichi kernel that populates `kelvin_term` for every particle.

    Iterates over all particles, calls `get_kelvin_term` for each,
    and stores the resulting Kelvin terms in the dedicated Taichi field.
    """
    for i in range(particle_count):
        update_kelvin_term(i)

@ti.kernel
def calculate_pressure_delta():
    """
    Taichi kernel that populates `pressure_delta` for every particle.

    Iterates over all particles, calls `get_pressure_delta` for each,
    and stores the resulting pressure deltas in the dedicated Taichi field.
    """
    for i in range(particle_count):
        update_partial_pressure(i)

@ti.kernel
def calculate_first_order_coefficient():
    """
    Taichi kernel that populates `first_order_coefficient` for every particle.

    Iterates over all particles, calls `get_first_order_mass_transport_via_system_state`
    for each, and stores the resulting first-order coefficients in the dedicated Taichi field.
    """
    for i in range(particle_count):
        update_first_order_coefficient(i)


@ti.kernel
def calculate_scaling_factors(time_step: float):
    """
    1. total_requested_mass[j] = Σ_i (mass_rate[i,j] * time_step
                                     * particle_concentration[i])
    2. scaling_factor[j] = 1 if request ≤ gas_mass[j] else
                           gas_mass[j] / total_requested_mass[j]
    """
    # zero the accumulator
    for j in range(species_count):
        total_requested_mass[j] = 0.0

    # accumulate requested mass
    for i, j in ti.ndrange(particle_count, species_count):
        total_requested_mass[j] += (
            mass_transport_rate[i, j] * time_step * particle_concentration[i]
        )

    # build scaling factors
    for j in range(species_count):
        scaling_factor[j] = 1.0
        if total_requested_mass[j] > gas_mass[j]:
            scaling_factor[j] = gas_mass[j] / total_requested_mass[j]

@ti.kernel
def calculate_transferable_mass(time_step: float):
    """
    Populates `transferable_mass` for every particle after
    scaling factors have been computed.
    """
    for i in range(particle_count):
        update_transferable_mass(i, time_step)

if __name__ == "__main__":
    calculate_radius()
    calculate_first_order_coefficient()
    calculate_kelvin_term()
    calculate_pressure_delta()
    calculate_mass_transport_rate()
    calculate_scaling_factors(time_step)       # NEW
    calculate_transferable_mass(time_step)     # NEW
    radius_np = radius.to_numpy()
    first_order_coefficient_np = first_order_coefficient.to_numpy()
    kelvin_radius_np = kelvin_radius.to_numpy()
    kelvin_term_np = kelvin_term.to_numpy()
    partial_pressure_np = partial_pressure.to_numpy()
    pressure_delta_np = pressure_delta.to_numpy()
    mass_transport_rate_np = mass_transport_rate.to_numpy()
    transferable_mass_np = transferable_mass.to_numpy()
    for i, r in enumerate(radius.to_numpy()):
        if i % 10 == 0:
            # print(f"Particle {i}: Radius = {r}, Mass Transport Rate = {first_order_coefficient_np[i]}")
            # print(f"Kelvin Radius = {kelvin_radius_np[i]}, Kelvin Term = {kelvin_term_np[i]}")
            # print(f"Partial Pressure = {partial_pressure_np[i]}, Pressure Delta = {pressure_delta_np[i]}")
            print(f"Mass Transport Rate = {mass_transport_rate_np[i]}")
    print("All calculations completed successfully.")
