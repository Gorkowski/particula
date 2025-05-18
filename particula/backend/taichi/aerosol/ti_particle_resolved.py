"""
Integrated Aerosol Dynamics and Particle-Resolved Simulation Representation
"""

import taichi as ti
import numpy as np
from particula.backend.taichi.util import FieldIO

import particula.backend.taichi.gas.properties as gas_properties
import particula.backend.taichi.particles.properties as particle_properties


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


# write a function to calculate the radius for a single particle
@ti.func
def get_radius(masses: ti.template(), density: ti.template()):
    """
    Calculate the radius of a particle given its mass and density

    Arguments:
        - mass_array: mass of the particle
        - density: density of the masses
    """
    # calculate the volume of the particle
    volume = ti.cast(0.0, float)
    for i in range(species_count):
        volume += masses[i] / density[i]
    # calculate the radius of the particle
    radius = (3 * volume / (4 * ti.pi)) ** (1 / 3)
    return radius

# write a kernel to calculate the radius for all particles
@ti.kernel
def calculate_radius():
    """
    Calculate the radius of all particles
    """
    for i in range(particle_count):
        # get the mass of the particle
        masses = species_masses[i, :]
        # calculate the radius of the particle
        radius = get_radius(masses, density)
    print(f"Particle {i}: Radius = {radius}")


if __name__ == "__main__":
    # run the kernel to calculate the radius
    calculate_radius()