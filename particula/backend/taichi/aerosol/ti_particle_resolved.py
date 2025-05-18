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

# intermediate fields
radius = ti.field(float, shape=(particle_count,), name="radii")
mass_transport_rate = ti.field(float, shape=(particle_count, species_count), name="mass_transport_rate")
first_order_coefficient = ti.field(float, shape=(particle_count, species_count), name="first_order_coefficient")
pressure_delta = ti.field(float, shape=(particle_count, species_count), name="pressure_delta")
kelvin_term = ti.field(float, shape=(particle_count, species_count), name="kelvin_term")
kelvin_radius = ti.field(float, shape=(particle_count, species_count), name="kelvin_radius")


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
def update_mass_transport_rate(p_index: int):
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
def weighted_average(
    value_array,           # ti.field or ti.ndarray (1-D)
    weight_array,          # ti.field or ti.ndarray (1-D)
    number_of_elements: float,
) -> float:
    """
    Compute the weighted average of *value_array* with *weight_array*.

    Arguments:
        value_array: 1-D Taichi field or ndarray that holds the values.
        weight_array: 1-D Taichi field or ndarray that holds the weights
            corresponding to each value.
        number_of_elements: How many elements from the arrays to include in
            the computation. Must not exceed their length.

    Returns:
        The weighted average, defined as
        ``sum(value_array[i] * weight_array[i]) /
        sum(weight_array[i])``.
    """
    weighted_sum = 0.0
    total_weight_sum = 0.0
    for element_index in range(number_of_elements):
        weighted_sum += (
            value_array[element_index] * weight_array[element_index]
        )
        total_weight_sum += weight_array[element_index]

    if total_weight_sum > 0.0:
        result = weighted_sum / total_weight_sum
    else:
        result = 0.0

    return result



@ti.func
def update_kelvin_radius(p_index: int):
    """
    Update the Kelvin radius for a particle.

    This function is a placeholder for the actual implementation of
    the Kelvin radius update. It currently does nothing.
    """
    for j in range(species_count):
        effective_surface_tension = weighted_average(
            surface_tension,
            species_masses[p_index],
            species_count,
        )
        effective_density = weighted_average(
            density,
            species_masses[p_index],
            species_count,
        )
        kelvin_radius[p_index, j] = particle_properties.fget_kelvin_radius(
            effective_surface_tension,
            effective_density,
            molar_mass[j],
            temperature,
        )


@ti.func
def update_kelvin_term(p_index: int):
    """
    Update the Kelvin term for a particle.

    This function is a placeholder for the actual implementation of
    the Kelvin term update. It currently does nothing.
    """
    for j in range(species_count):
        kelvin_term[p_index, j] = particle_properties.fget_kelvin_term(
            r_p=radius[p_index],
            r_k=kelvin_radius[p_index, j],
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
def calculate_kelvin_radius():
    """
    Taichi kernel that populates `kelvin_radius` for every particle.

    Iterates over all particles, calls `get_kelvin_radius` for each,
    and stores the resulting Kelvin radii in the dedicated Taichi field.
    """
    for i in range(particle_count):
        update_kelvin_radius(i)

@ti.kernel
def calculate_kelvin_term():
    """
    Taichi kernel that populates `kelvin_term` for every particle.

    Iterates over all particles, calls `get_kelvin_term` for each,
    and stores the resulting Kelvin terms in the dedicated Taichi field.
    """
    for i in range(particle_count):
        update_kelvin_term(i)


if __name__ == "__main__":
    calculate_radius()
    calculate_mass_transport_rate()
    calculate_kelvin_radius()
    calculate_kelvin_term()
    radius_np = radius.to_numpy()
    first_order_coefficient_np = first_order_coefficient.to_numpy()
    kelvin_radius_np = kelvin_radius.to_numpy()
    kelvin_term_np = kelvin_term.to_numpy()
    for i, r in enumerate(radius.to_numpy()):
        print(f"Particle {i}: Radius = {r}, Mass Transport Rate = {first_order_coefficient_np[i]}")
        print(f"Kelvin Radius = {kelvin_radius_np[i]}, Kelvin Term = {kelvin_term_np[i]}")
