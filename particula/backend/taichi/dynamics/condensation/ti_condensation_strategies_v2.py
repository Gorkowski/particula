"""
Taichi implementation of condensation strategies (isothermal).

This module provides Taichi-accelerated routines for isothermal condensation
processes, including mass transfer and first-order mass transport calculations
for aerosol particles and gas species. The routines are designed for efficient
simulation of multiphase systems in atmospheric and aerosol science.

Examples:
    ```py title="Example Usage"
    # Example usage will be provided here.
    ```

References:
    # Add references here as needed.
"""

import taichi as ti
import numpy as np
from numpy.typing import NDArray
from particula.backend.dispatch_register import register

from particula.backend.taichi.dynamics.condensation.ti_mass_transfer import (
    fget_mass_transfer_rate,
    fget_first_order_mass_transport_via_system_state,
)
from particula.particles.representation import ParticleRepresentation
from particula.gas.species import GasSpecies
from particula.particles import get_partial_pressure_delta
from particula.dynamics.condensation.mass_transfer import get_mass_transfer
import logging

logger = logging.getLogger("particula")


@ti.data_oriented
class CondensationIsothermal:
    """
    Taichi version for CondensationIsothermal.

    This class implements isothermal condensation strategies using Taichi
    kernels for efficient computation. It provides methods to calculate
    first-order mass transport coefficients, mass transfer rates, and
    to advance the system state in time for multiphase aerosol/gas systems.

    Attributes:
        - molar_mass : Array of molar masses for each species.
        - diffusion_coefficient : Diffusion coefficient (scalar field).
        - accommodation_coefficient : Array of accommodation coefficients.
        - update_gases : Boolean flag to update gas concentrations.

    Methods:
        - first_order_mass_transport: Compute first-order mass transport
          coefficients for all particles and species.
        - mass_transfer_rate: Compute mass transfer rates for all particles
          and species.
        - rate: Compute concentration-scaled mass transfer rates.
        - step: Advance the system state by one time step.

    Examples:
        ```py title="Example Usage"
        condensation = CondensationIsothermal(
            molar_mass=ti.ndarray(dtype=ti.f64, shape=(2,)),
            diffusion_coefficient=ti.field(ti.f64, shape=()),
            accommodation_coefficient=ti.ndarray(dtype=ti.f64, shape=(2,)),
            update_gases=True,
        )
        # Use condensation.first_order_mass_transport(...) etc.
        ```
    """

    # ─────────────────────────── constructor ───────────────────────────────
    def __init__(
        self,
        molar_mass: ti.types.ndarray(dtype=ti.f64, ndim=1),
        diffusion_coefficient: ti.field(ti.f64, shape=()),
        accommodation_coefficient: ti.types.ndarray(dtype=ti.f64, ndim=1),
        update_gases: bool = False,
    ):
        self.molar_mass = molar_mass
        self.diffusion_coefficient = diffusion_coefficient
        if accommodation_coefficient.shape[0] != molar_mass.shape[0]:
            # distribute accommodation coefficient to all species
            accommodation_coefficient_new = ti.ndarray(
                dtype=ti.f64, shape=(molar_mass.shape[0],)
            )
            for i in range(molar_mass.shape[0]):
                accommodation_coefficient_new[i] = accommodation_coefficient[0]
            self.accommodation_coefficient = accommodation_coefficient_new
        else:
            self.accommodation_coefficient = accommodation_coefficient
        self.update_gases = bool(update_gases)

    # ───────────────────── helper: zero-radius guard ───────────────────────
    def _fill_zero_radius(self, radius: np.ndarray) -> np.ndarray:
        if np.max(radius) == 0.0:
            logger.warning(
                "All radius values are zero, radius set to 1 m for condensation calculations."
            )
            radius = np.where(radius == 0.0, 1.0, radius)
        return np.where(radius == 0.0, np.max(radius), radius)

    # ──────────────────────── Taichi kernels ───────────────────────────────
    @ti.kernel
    def _kget_first_order_mass_transport(
        self,
        particle_radius: ti.types.ndarray(dtype=ti.f64, ndim=1),
        temperature: ti.f64,
        pressure: ti.f64,
        dynamic_viscosity: ti.f64,
        result: ti.types.ndarray(dtype=ti.f64, ndim=2),
    ):
        """
        Compute first-order mass-transport coefficient per particle per species.

        Arguments:
            - particle_radius : Particle radii array, shape (n_particles,)
            - temperature : Temperature [K]
            - pressure : Pressure [Pa]
            - dynamic_viscosity : Dynamic viscosity [Pa·s]
            - result : Output array, shape (n_particles, n_species), units [m^3/s]

        Returns:
            - None (results written in-place to `result`)
        """
        for particle_i in range(particle_radius.shape[0]):  # particles
            for species_i in range(self.molar_mass.shape[0]):  # species
                result[particle_i, species_i] = (
                    fget_first_order_mass_transport_via_system_state(
                        particle_radius[particle_i],
                        self.molar_mass[species_i],
                        self.accommodation_coefficient[species_i],
                        temperature,
                        pressure,
                        dynamic_viscosity,
                        self.diffusion_coefficient[None],
                    )
                )

    @ti.kernel
    def _kget_mass_transfer_rate(
        self,
        particle_radius: ti.types.ndarray(dtype=ti.f64, ndim=1),
        temperature: ti.f64,
        pressure: ti.f64,
        dynamic_viscosity: ti.f64,
        pressure_delta: ti.types.ndarray(dtype=ti.f64, ndim=2),
        result: ti.types.ndarray(dtype=ti.f64, ndim=2),
    ):
        """
        Compute mass transfer rate array (dm/dt) per particle and species.

        Arguments:
            - particle_radius : Particle radii array, shape (n_particles,)
            - temperature : Temperature [K]
            - pressure : Pressure [Pa]
            - dynamic_viscosity : Dynamic viscosity [Pa·s]
            - pressure_delta : Partial pressure delta array,
                shape (n_particles, n_species)
            - result : Output array, shape (n_particles, n_species),
                units [kg/s]

        Returns:
            - None (results written in-place to `result`)
        """
        for particle_i in range(pressure_delta.shape[0]):
            for species_i in range(pressure_delta.shape[1]):
                first_order_mass_transport_k = (
                    fget_first_order_mass_transport_via_system_state(
                        particle_radius[particle_i],
                        self.molar_mass[species_i],
                        self.accommodation_coefficient[species_i],
                        temperature,
                        pressure,
                        dynamic_viscosity,
                        self.diffusion_coefficient[None],
                    )
                )

                result[particle_i, species_i] = fget_mass_transfer_rate(
                    dp=pressure_delta[particle_i, species_i],
                    k=first_order_mass_transport_k,
                    t=temperature,
                    m=self.molar_mass[species_i],
                )

    # ─────────────────────── public helpers ────────────────────────────────
    def calculate_pressure_delta(
        self,
        particle: ParticleRepresentation,
        gas_species: GasSpecies,
        temperature: float,
        radius: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Calculate the difference in partial pressure between the gas and
        particle phases.

        Arguments:
            - particle : The particle for which the partial pressure difference
                is to be calculated.
            - gas_species : The gas species with which the particle is in
                contact.
            - temperature : The temperature at which the partial pressure
                difference is to be calculated.
            - radius : The radius of the particles.

        Returns:
            - partial_pressure_delta : The difference in partial pressure
                between the gas and particle phases.
        """
        mass_concentration_in_particle = particle.get_species_mass()
        pure_vapor_pressure = gas_species.get_pure_vapor_pressure(
            temperature=temperature
        )
        partial_pressure_particle = particle.activity.partial_pressure(
            pure_vapor_pressure=pure_vapor_pressure,
            mass_concentration=mass_concentration_in_particle,
        )

        partial_pressure_gas = gas_species.get_partial_pressure(temperature)
        kelvin_term = particle.surface.kelvin_term(
            radius=radius,
            molar_mass=self.molar_mass,
            mass_concentration=mass_concentration_in_particle,
            temperature=temperature,
        )

        return get_partial_pressure_delta(
            partial_pressure_gas=partial_pressure_gas,
            partial_pressure_particle=partial_pressure_particle,
            kelvin_term=kelvin_term,
        )

    def first_order_mass_transport(
        self,
        particle_radius: np.ndarray,
        temperature: float,
        pressure: float,
        dynamic_viscosity: float,
    ) -> np.ndarray:
        """Vectorised mass_transport_coeff_array (K(r)) using Taichi kernel.

        Arguments:
            - particle_radius : Particle radii array, shape (n_particles,)
            - temperature : Temperature [K]
            - pressure : Pressure [Pa]
            - dynamic_viscosity : Dynamic viscosity [Pa·s]

        Returns:
            - mass_transport_coeff_array : Array of shape
                (n_particles, n_species)
        """
        particle_radius_np = np.ascontiguousarray(particle_radius, dtype=np.float64)
        n_particles = particle_radius_np.shape[0]
        n_species = self.molar_mass.shape[0]
        mass_transport_coeff_array = np.empty(
            (n_particles, n_species), dtype=np.float64
        )

        self._kget_first_order_mass_transport(
            particle_radius_np,
            float(temperature),
            float(pressure),
            float(dynamic_viscosity),
            mass_transport_coeff_array,
        )
        return mass_transport_coeff_array

    # ───────────────────────── API: dm/dt ───────────────────────────────────
    def mass_transfer_rate(
        self,
        particle: ParticleRepresentation,
        gas_species: GasSpecies,
        temperature: float,
        pressure: float,
        dynamic_viscosity: float | None = None,
    ) -> np.ndarray:
        """
        Compute mass transfer rate array (dm/dt) for all particles and species.

        Arguments:
            - particle : ParticleRepresentation object
            - gas_species : GasSpecies object
            - temperature : Temperature [K]
            - pressure : Pressure [Pa]
            - dynamic_viscosity : Dynamic viscosity [Pa·s] (optional)

        Returns:
            - mass_transfer_rate_array : Array of shape
                (n_particles, n_species)
        """
        radius = self._fill_zero_radius(particle.get_radius())
        if dynamic_viscosity is None:
            from particula.gas.properties.dynamic_viscosity import (
                get_dynamic_viscosity,
            )
            dynamic_viscosity = get_dynamic_viscosity(temperature)

        pressure_delta_np = np.ascontiguousarray(
            self.calculate_pressure_delta(
                particle=particle,
                gas_species=gas_species,
                temperature=temperature,
                radius=radius,
            ),
            dtype=np.float64,
        )
        mass_transfer_rate_array = np.empty_like(pressure_delta_np)
        self._kget_mass_transfer_rate(
            radius,
            float(temperature),
            float(pressure),
            float(dynamic_viscosity),
            pressure_delta_np,
            mass_transfer_rate_array,
        )
        return mass_transfer_rate_array

    # ──────────────────── API: concentration-scaled rate ───────────────────
    def rate(
        self,
        particle: ParticleRepresentation,
        gas_species: GasSpecies,
        temperature: float,
        pressure: float,
    ) -> np.ndarray:
        mass_transfer_rate_array = self.mass_transfer_rate(
            particle=particle,
            gas_species=gas_species,
            temperature=temperature,
            pressure=pressure,
        )
        concentration = particle.concentration
        if mass_transfer_rate_array.ndim == 2:
            concentration = concentration[:, np.newaxis]
        return mass_transfer_rate_array * concentration

    # ─────────────────────────── API: step ─────────────────────────────────
    def step(
        self,
        particle: ParticleRepresentation,
        gas_species: GasSpecies,
        temperature: float,
        pressure: float,
        time_step: float,
    ):
        mass_transfer_rate_array = self.mass_transfer_rate(
            particle=particle,
            gas_species=gas_species,
            temperature=temperature,
            pressure=pressure,
        )
        mass_change = get_mass_transfer(
            mass_rate=mass_transfer_rate_array,
            time_step=time_step,
            gas_mass=gas_species.get_concentration(),
            particle_mass=particle.get_species_mass(),
            particle_concentration=particle.get_concentration(),
        )
        particle.add_mass(added_mass=mass_change)
        if self.update_gases:
            gas_species.add_concentration(-mass_change.sum(axis=0))
        return particle, gas_species


# ─────────────────── backend factory registration ───────────────────────────
@register("condensation_isothermal", backend="taichi")  # factory key
def TiCondensationIsothermal(**kwargs):
    """Factory wrapper used by dispatch_register."""
    return CondensationIsothermal(**kwargs)
