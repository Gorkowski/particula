"""
Integrated Aerosol Dynamics and Particle-Resolved Simulation Representation
"""

import taichi as ti
import numpy as np
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

# create radii field to store computed radii
radii = ti.field(float, shape=(particle_count,), name="radii")


@ti.func
def get_radius(p_index: int) -> float:
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
    return (3.0 * volume / (4.0 * ti.pi)) ** (1.0 / 3.0)

@ti.kernel
def calculate_radius():
    """
    Taichi kernel that populates `radii` for every particle.

    Iterates over all particles, calls `get_radius` for each, and stores
    the resulting radii in the dedicated Taichi field.
    """
    for i in range(particle_count):
        radii[i] = get_radius(i)


if __name__ == "__main__":
    calculate_radius()
    for i, r in enumerate(radii.to_numpy()):
        print(f"Particle {i}: Radius = {r}")
