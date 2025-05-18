"""
Minimal demonstration of “dependency injection” with Taichi data-oriented
classes.

The Particle and GasSpecies objects are created in Python, injected into
PressureDeltaCalc, and their @ti.func methods are called from inside a
kernel.

This module demonstrates how to structure Taichi data-oriented classes for
modular, testable scientific computing, using pressure delta calculations
as an example.

Examples:
    ```py title="Example Usage"
    python particula/backend/taichi/taichi_injection.py
    # Output:
    # Delta P (gas - particle * Kelvin) [Pa]:
    # [values...]
    ```

References:
    - "Dependency injection," [Wikipedia](https://en.wikipedia.org/wiki/Dependency_injection)
    - "Taichi: A Parallel Programming Language for High-Performance Numerical Computation,"
      [Taichi Docs](https://docs.taichi-lang.org/)
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
    """
    Represents a gas species with properties for vapor pressure and
    Henry's law calculations.

    Attributes:
        - molar_mass : Field containing the molar mass [kg/mol].
        - henry_constant : Field for Henry's law constant.

    Methods:
        - pure_vapor_pressure : Computes pure vapor pressure using a toy
          Antoine-like equation.
        - partial_pressure : Computes partial pressure using Henry's law.

    Examples:
        ```py title="Example Usage"
        gas_species = GasSpecies(molar_mass=0.018, henry_constant=1.5e-3)
        p_vap = gas_species.pure_vapor_pressure(298.15)
        p_part = gas_species.partial_pressure(298.15)
        ```

    References:
        - "Henry's law," [Wikipedia](https://en.wikipedia.org/wiki/Henry%27s_law)
        - "Antoine equation," [Wikipedia](https://en.wikipedia.org/wiki/Antoine_equation)
    """

    def __init__(self, molar_mass, henry_constant):
        """
        Initialize a GasSpecies object.

        Arguments:
            - molar_mass : Molar mass of the gas species [kg/mol].
            - henry_constant : Henry's law constant.

        Returns:
            - None
        """
        self.molar_mass = ti.field(ti.f64, shape=())
        self.henry_constant = ti.field(ti.f64, shape=())
        self.molar_mass[None] = molar_mass
        self.henry_constant[None] = henry_constant

    @ti.func
    def pure_vapor_pressure(self, temperature):
        """
        Compute the pure vapor pressure using a toy Antoine-like equation.

        Equation:
            p_vap = exp(10.0 - molar_mass / temperature)

        Arguments:
            - temperature : Temperature [K].

        Returns:
            - Pure vapor pressure [Pa].

        Examples:
            ```py
            gas_species.pure_vapor_pressure(298.15)
            ```
        """
        return ti.exp(10.0 - self.molar_mass[None] / temperature)

    @ti.func
    def partial_pressure(self, temperature):
        """
        Compute the partial pressure using Henry's law.

        Equation:
            p_part = henry_constant × temperature

        Arguments:
            - temperature : Temperature [K].

        Returns:
            - Partial pressure [Pa].

        Examples:
            ```py
            gas_species.partial_pressure(298.15)
            ```
        References:
            - "Henry's law," [Wikipedia](https://en.wikipedia.org/wiki/Henry%27s_law)
        """
        return self.henry_constant[None] * temperature


@ti.data_oriented
class Particle:
    """
    Represents a collection of particles with mass and provides
    methods for mass and pressure calculations.

    Attributes:
        - mass : Field containing particle masses [kg].

    Methods:
        - species_mass : Returns the mass of a particle at a given index.
        - kelvin_term : Computes the Kelvin effect term.
        - partial_pressure : Computes the partial pressure for a particle.

    Examples:
        ```py title="Example Usage"
        mass_array = np.array([1e-18, 2e-18, 3e-18, 4e-18])
        particle = Particle(mass_array)
        m = particle.species_mass(0)
        ```
    """

    def __init__(self, mass_array):
        """
        Initialize a Particle object.

        Arguments:
            - mass_array : Numpy array of particle masses [kg].

        Returns:
            - None
        """
        n_particles = mass_array.size
        self.mass = ti.field(dtype=ti.f64, shape=(n_particles,))
        self.mass.from_numpy(mass_array)

    @ti.func
    def species_mass(self, index):
        """
        Return the mass of the particle at the given index.

        Arguments:
            - index : Index of the particle.

        Returns:
            - Mass of the particle [kg].

        Examples:
            ```py
            particle.species_mass(0)
            ```
        """
        return self.mass[index]

    @ti.func
    def kelvin_term(self, radius, molar_mass, temperature):
        """
        Compute the Kelvin effect term.

        Equation:
            kelvin_term = exp(
                2 × surface_tension × molar_mass
                / (radius × gas_constant × temperature)
            )

        Arguments:
            - radius : Particle radius [m].
            - molar_mass : Molar mass [kg/mol].
            - temperature : Temperature [K].

        Returns:
            - Kelvin effect term (dimensionless).

        Examples:
            ```py
            particle.kelvin_term(1e-7, 0.018, 298.15)
            ```
        References:
            - "Kelvin equation," [Wikipedia](https://en.wikipedia.org/wiki/Kelvin_equation)
        """
        return ti.exp(
            2 * SURFACE_TENSION * molar_mass
            / (radius * GAS_CONSTANT * temperature)
        )

    @ti.func
    def partial_pressure(self, pure_vapor_pressure, mass_concentration):
        """
        Compute the partial pressure for a particle.

        Equation:
            p_part = activity_coefficient × mass_concentration × pure_vapor_pressure

        Arguments:
            - pure_vapor_pressure : Pure vapor pressure [Pa].
            - mass_concentration : Mass concentration (dimensionless or [kg/m³]).

        Returns:
            - Partial pressure [Pa].

        Examples:
            ```py
            particle.partial_pressure(100.0, 1.0)
            ```
        """
        activity_coefficient = 1.0  # placeholder
        return (
            activity_coefficient
            * mass_concentration
            * pure_vapor_pressure
        )


# --------------------------------------------------------------------------- #
#  Calculator that “owns” the collaborators inside its kernel
# --------------------------------------------------------------------------- #
@ti.data_oriented
class PressureDeltaCalc:
    """
    Calculator for the pressure delta (ΔP) between gas and particle phases,
    using injected Particle and GasSpecies objects.

    Attributes:
        - particle : Injected Particle object.
        - gas_species : Injected GasSpecies object.
        - n_particles : Number of particles.
        - radius : Field of particle radii [m].
        - pressure_delta : Field of computed pressure deltas [Pa].

    Methods:
        - compute : Computes the pressure delta for all particles.

    Examples:
        ```py title="Example Usage"
        pressure_delta_calc = PressureDeltaCalc(particle, gas_species, radii_array)
        pressure_delta_calc.compute(298.15)
        print(pressure_delta_calc.pressure_delta.to_numpy())
        ```
    """

    def __init__(self, particle: Particle, gas_species: GasSpecies, radii_array):
        """
        Initialize a PressureDeltaCalc object.

        Arguments:
            - particle : Particle object (injected dependency).
            - gas_species : GasSpecies object (injected dependency).
            - radii_array : Numpy array of particle radii [m].

        Returns:
            - None
        """
        self.particle = particle  # injected dependency
        self.gas_species = gas_species  # injected dependency
        self.n_particles = radii_array.size
        self.radius = ti.field(dtype=ti.f64, shape=(self.n_particles,))
        self.pressure_delta = ti.field(dtype=ti.f64, shape=(self.n_particles,))
        self.radius.from_numpy(radii_array)

    @ti.kernel
    def compute(self, temperature: ti.f64):
        """
        Compute the pressure delta (ΔP) for all particles.

        For each particle:
            ΔP = partial_pressure_gas
                 - partial_pressure_particle × kelvin_term

        Arguments:
            - temperature : Temperature [K].

        Returns:
            - None (results stored in self.pressure_delta).

        Examples:
            ```py
            pressure_delta_calc.compute(298.15)
            print(pressure_delta_calc.pressure_delta.to_numpy())
            ```
        References:
            - "Kelvin equation," [Wikipedia](https://en.wikipedia.org/wiki/Kelvin_equation)
            - "Henry's law," [Wikipedia](https://en.wikipedia.org/wiki/Henry%27s_law)
        """
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
    """
    Main driver for demonstrating pressure delta calculation with Taichi
    data-oriented classes.

    Sets up a toy 4-particle system, computes the pressure delta, and prints
    the results.

    Arguments:
        - None

    Returns:
        - None

    Examples:
        ```py title="Example Usage"
        python particula/backend/taichi/taichi_injection.py
        # Output:
        # Delta P (gas - particle * Kelvin) [Pa]:
        # [values...]
        ```
    """
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
