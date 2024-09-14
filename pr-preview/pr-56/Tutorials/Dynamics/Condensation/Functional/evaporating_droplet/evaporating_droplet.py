# %% Example for an evaporating droplet

import numpy as np
import matplotlib.pyplot as plt

from particula.next.dynamics.condensation import mass_transfer
from particula.next.particles import properties
from particula.next.particles.properties import (
    calculate_knudsen_number,
    vapor_transition_correction,
    partial_pressure_delta,
)
from particula.next.gas.properties import (
    molecule_mean_free_path,
    get_dynamic_viscosity,
    buck_vapor_pressure,
    calculate_partial_pressure,
    calculate_concentration,
)

from particula.util.input_handling import convert_units

# Define the parameters

radius = np.logspace(-7, -3, 100)
molar_mass = 0.0180153
temperature = 298.15
pressure = 101325
accommodation_coefficient = 1.0
dynamic_viscosity = get_dynamic_viscosity(temperature)


density_1 = 1000  # kg/m^3
density_2 = 2000  # kg/m^3
density_3 = 1500  # kg/m^3

masses_2 = 4/3 * np.pi * radius**3 * density_2
masses_1 = masses_2 * density_1 / density_2 * 0.1
masses_3 = masses_2 * density_3 / density_2

molar_mass_1 = 18.01528e-3  # g/mol
molar_mass_2 = 200e-3  # g/mol
molar_mass_3 = 150e-3  # g/mol

molar_mass = np.array([molar_mass_1, molar_mass_2])
mass_component = np.column_stack((masses_1, masses_2))
# mass_3component =np.column_stack((masses_1, masses_2, masses_3))

density_component = np.array([density_1, density_2])
# density_3component = np.array([density_1, density_2, density_3])

pure_vapor_pressure_1 = buck_vapor_pressure(temperature)
pure_vapor_pressure_2 = 1e-20
pure_vapor_pressure_3 = 1e-20

pure_vapor_pressure = np.array([pure_vapor_pressure_1, pure_vapor_pressure_2])


gas_vapor_pressure = np.array([pure_vapor_pressure_1, pure_vapor_pressure_2])
gas_vapor_pressure *= 0.8

kevlin_radius_1 = properties.kelvin_radius(
    effective_surface_tension=0.072,
    effective_density=density_1,
    molar_mass=molar_mass_1,
    temperature=temperature,
)
kevlin_radius_2 = properties.kelvin_radius(
    effective_surface_tension=0.072,
    effective_density=density_2,
    molar_mass=molar_mass_2,
    temperature=temperature,
)

kevlin_radius = np.array([kevlin_radius_1, kevlin_radius_2])

# %%
# Define the parameters

mean_free_path = molecule_mean_free_path(
    molar_mass=molar_mass,
    temperature=temperature,
    pressure=pressure,
    dynamic_viscosity=dynamic_viscosity,
)
knudsen_number = calculate_knudsen_number(
    mean_free_path=mean_free_path,
    particle_radius=radius,
)

vapor_transition = vapor_transition_correction(
    knudsen_number=knudsen_number,
    mass_accommodation=accommodation_coefficient,
)


# %%
first_order_mass_transport = mass_transfer.first_order_mass_transport_k(
    radius=radius,
    vapor_transition=vapor_transition,
    diffusion_coefficient=2e-5,
)

activity_particle = properties.ideal_activity_molar(
    mass_concentration=mass_component,
    molar_mass=molar_mass,
)

particle_partial_pressure = properties.calculate_partial_pressure(
    pure_vapor_pressure=pure_vapor_pressure,
    activity=activity_particle,
)

kelvin_term = properties.kelvin_term(
    radius=radius,
    kelvin_radius_value=kevlin_radius,
)

pressure_delta = partial_pressure_delta(
    partial_pressure_gas=gas_vapor_pressure,
    partial_pressure_particle=particle_partial_pressure,
    kelvin_term=kelvin_term,
)

mass_transfer_rate = mass_transfer.mass_transfer_rate(
    pressure_delta=pressure_delta,
    first_order_mass_transport=first_order_mass_transport,
    temperature=temperature,
    molar_mass=molar_mass,
)

# convert to volume and then radius rate
radius_rate = mass_transfer.radius_transfer_rate(
    mass_rate=mass_transfer_rate,
    radius=radius,
    density=density_component,
)


# %%
fig, ax = plt.subplots()
ax.plot(radius, first_order_mass_transport, label="First-order mass transport")

ax.set_xscale("log")
ax.set_xlabel("Radius [m]")
ax.set_ylabel("First-order mass transport [m^3/s]")
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(radius, mass_transfer_rate, label="Mass transfer rate")

ax.set_xscale("log")
ax.set_xlabel("Radius [m]")
ax.set_ylabel("Mass transfer rate [kg/s]")
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(radius, pressure_delta[:, 0], label="Pressure delta")

ax.set_xscale("log")
ax.set_xlabel("Radius [m]")
ax.set_ylabel("Pressure delta [Pa]")
ax.legend()
plt.show()


fig, ax = plt.subplots()
x = radius * convert_units('m', 'nm')
ax.plot(x , radius_rate[:, 0] * convert_units('m', 'nm'), label="Radius rate")
# ax.plot(x , radius_rate[:, 1] * convert_units('m', 'nm'), label="Radius rate")

ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xlabel("Radius [nm]")
ax.set_ylabel("Radius rate [nm/s]")
ax.legend()
plt.show()

# %%
# # pylint: disable=too-many-arguments

# # Calculate the first-order mass transport coefficient
# first_order_mass_transport = self.first_order_mass_transport(
#     radius=particle.get_radius(),
#     temperature=temperature,
#     pressure=pressure,
#     dynamic_viscosity=dynamic_viscosity,
# )
# # calculate the partial pressure
# partial_pressure_particle = particle.activity.partial_pressure(
#     pure_vapor_pressure=gas_species.get_pure_vapor_pressure(
#         temperature
#     ),
#     mass_concentration=particle.get_species_mass(),
# )
# partial_pressure_gas = gas_species.get_partial_pressure(temperature)
# # calculate the kelvin term
# kelvin_term = particle.surface.kelvin_term(
#     radius=particle.get_radius(),
#     molar_mass=self.molar_mass,
#     mass_concentration=particle.get_species_mass(),
#     temperature=temperature,
# )
# # calculate the pressure delta accounting for the kelvin term
# pressure_delta = partial_pressure_delta(
#     partial_pressure_gas=partial_pressure_gas,
#     partial_pressure_particle=partial_pressure_particle,
#     kelvin_term=kelvin_term,
# )
