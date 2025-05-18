"""
Minimal demonstration of “dependency injection” with Taichi data-oriented
classes.  The Particle and GasSpecies objects are created in Python, injected
into PressureDeltaCalc, and their @ti.func methods are called from inside a
kernel.
"""

import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)  # use ti.cpu if you don’t have a GPU

GAS_CONSTANT = 8.314        # J mol⁻¹ K⁻¹
SURFACE_TENSION = 0.072     # N m⁻¹


# --------------------------------------------------------------------------- #
#  Supporting objects that will be “injected”
# --------------------------------------------------------------------------- #
@ti.data_oriented
class GasSpecies:
    def __init__(self, molar_mass, henry_constant):
        self.molar_mass = ti.field(ti.f64, shape=())
        self.henry_constant = ti.field(ti.f64, shape=())
        self.molar_mass[None] = molar_mass
        self.henry_constant[None] = henry_constant

    @ti.func
    def pure_vapor_pressure(self, temperature):
        # toy Antoine-like equation
        return ti.exp(10.0 - self.molar_mass[None] / temperature)

    @ti.func
    def partial_pressure(self, temperature):
        # Henry’s-law placeholder
        return self.henry_constant[None] * temperature


@ti.data_oriented
class Particle:
    def __init__(self, mass_array):
        n_particles = mass_array.size
        self.mass = ti.field(dtype=ti.f64, shape=(n_particles,))
        self.mass.from_numpy(mass_array)

    @ti.func
    def species_mass(self, index):
        return self.mass[index]

    @ti.func
    def kelvin_term(self, radius, molar_mass, mass_concentration, temperature):
        return ti.exp(
            2 * SURFACE_TENSION * molar_mass
            / (radius * GAS_CONSTANT * temperature)
        )

    @ti.func
    def partial_pressure(self, pure_vapor_pressure, mass_concentration):
        gamma = 1.0  # activity coefficient (placeholder)
        return gamma * mass_concentration * pure_vapor_pressure


# --------------------------------------------------------------------------- #
#  Calculator that “owns” the collaborators inside its kernel
# --------------------------------------------------------------------------- #
@ti.data_oriented
class PressureDeltaCalc:
    def __init__(self, particle: Particle, gas_species: GasSpecies, radii_array):
        self.particle = particle  # injected dependency
        self.gas_species = gas_species  # injected dependency
        self.n_particles = radii_array.size
        self.radius = ti.field(dtype=ti.f64, shape=(self.n_particles,))
        self.pressure_delta = ti.field(dtype=ti.f64, shape=(self.n_particles,))
        self.radius.from_numpy(radii_array)

    @ti.kernel
    def compute(self, temperature: ti.f64):
        for index in range(self.n_particles):
            mass_concentration_particle = self.particle.species_mass(index)
            pure_vapor_pressure = self.gas_species.pure_vapor_pressure(temperature)
            partial_pressure_particle = self.particle.partial_pressure(
                pure_vapor_pressure, mass_concentration_particle
            )
            partial_pressure_gas = self.gas_species.partial_pressure(temperature)
            kelvin_term_value = self.particle.kelvin_term(
                self.radius[index],
                self.gas_species.molar_mass[None],
                mass_concentration_particle,
                temperature,
            )
            self.pressure_delta[index] = (
                partial_pressure_gas
                - partial_pressure_particle * kelvin_term_value
            )


# --------------------------------------------------------------------------- #
#  Simple test driver
# --------------------------------------------------------------------------- #
def main():
    # toy data for a 4-particle system
    n_particles = 4
    mass_array = np.array([1e-18, 2e-18, 3e-18, 4e-18])  # kg (placeholder)
    radii_array = np.linspace(50e-9, 200e-9, n_particles)  # m
    temperature = 298.15  # K

    particle = Particle(mass_array)
    gas_species = GasSpecies(molar_mass=0.018, henry_constant=1.5e-3)
    pressure_delta_calc = PressureDeltaCalc(particle, gas_species, radii_array)

    pressure_delta_calc.compute(temperature)
    print("Delta P (gas - particle * Kelvin) [Pa]:")
    print(pressure_delta_calc.pressure_delta.to_numpy())


if __name__ == "__main__":
    main()
