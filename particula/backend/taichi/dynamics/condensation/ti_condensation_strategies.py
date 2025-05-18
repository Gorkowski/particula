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
import logging
from numpy.typing import NDArray
from particula.backend.dispatch_register import register

from particula.backend.taichi.dynamics.condensation.ti_mass_transfer import (
    fget_mass_transfer_rate,
    fget_first_order_mass_transport_via_system_state,
)
from particula.backend.taichi.particles.ti_representation import (
    TiParticleRepresentation,
)
from particula.backend.taichi.gas.ti_species import TiGasSpecies
from particula.backend.taichi.particles.properties import (
    fget_partial_pressure_delta,
)


@ti.data_oriented
class TiCondensationIsothermal:
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
        mm: ti.types.ndarray(dtype=ti.f64, ndim=1),
        alpha: ti.types.ndarray(dtype=ti.f64, ndim=1),
        result: ti.types.ndarray(dtype=ti.f64, ndim=2),
    ):
        """
        Compute first-order mass-transport coefficient per particle per species.

        Arguments:
            - particle_radius : Particle radii array, shape (n_particles,)
            - temperature : Temperature [K]
            - pressure : Pressure [Pa]
            - dynamic_viscosity : Dynamic viscosity [Pa·s]
            - mm : Molar mass array, shape (n_species,)
            - alpha : Accommodation coefficient array, shape (n_species,)
            - result : Output array, shape (n_particles, n_species), units [m^3/s]

        Returns:
            - None (results written in-place to `result`)
        """
        for particle_i in range(particle_radius.shape[0]):  # particles
            for species_i in range(mm.shape[0]):  # species
                result[particle_i, species_i] = (
                    fget_first_order_mass_transport_via_system_state(
                        particle_radius[particle_i],
                        mm[species_i],
                        alpha[species_i],
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
        mm: ti.types.ndarray(dtype=ti.f64, ndim=1),
        alpha: ti.types.ndarray(dtype=ti.f64, ndim=1),
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
            - mm : Molar mass array, shape (n_species,)
            - alpha : Accommodation coefficient array, shape (n_species,)
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
                        mm[species_i],
                        alpha[species_i],
                        temperature,
                        pressure,
                        dynamic_viscosity,
                        self.diffusion_coefficient[None],
                    )
                )

                result[particle_i, species_i] = fget_mass_transfer_rate(
                    pressure_delta=pressure_delta[particle_i, species_i],
                    first_order_mass_transport=first_order_mass_transport_k,
                    temperature=temperature,
                    molar_mass=mm[species_i],
                )

    @ti.kernel
    def _kget_pressure_delta(
        self,
        particle_mass: ti.types.ndarray(dtype=ti.f64, ndim=2),
        pure_vp:       ti.types.ndarray(dtype=ti.f64, ndim=1),
        pp_gas:        ti.types.ndarray(dtype=ti.f64, ndim=1),
        kelvin:        ti.types.ndarray(dtype=ti.f64, ndim=2),
        result:        ti.types.ndarray(dtype=ti.f64, ndim=2),
        activity:      ti.template(),        # particle.activity strategy
    ):
        for p, s in result:                  # particle, species
            pp_part = activity.fget_partial_pressure_internal(
                particle_mass[p, s], pure_vp[s]
            )
            result[p, s] = fget_partial_pressure_delta(
                pp_gas[s], pp_part, kelvin[p, s]
            )

    @ti.kernel
    def _kscale_rate(
        self,
        rate_arr: ti.types.ndarray(dtype=ti.f64, ndim=2),
        conc:     ti.template(),                   # particle.concentration field
        result:   ti.types.ndarray(dtype=ti.f64, ndim=2),
    ):
        for p, s in result:
            result[p, s] = rate_arr[p, s] * conc[p]

    @ti.kernel
    def _kapply_mass_change(
        self,
        mass_rate: ti.types.ndarray(dtype=ti.f64, ndim=2),
        dt: ti.f64,
        mass_change: ti.types.ndarray(dtype=ti.f64, ndim=2),
    ):
        for p, s in mass_change:
            mass_change[p, s] = mass_rate[p, s] * dt

    # ─────────────────────── public helpers ────────────────────────────────
    def calculate_pressure_delta(
        self,
        particle: TiParticleRepresentation,
        gas_species: TiGasSpecies,
        temperature: float,
        radius: NDArray[np.float64],
    ) -> ti.ndarray:
        # all required fields already live in Taichi
        mass_in_particle   = particle.get_species_mass()          # (N,P) ti.ndarray
        pure_vp            = gas_species.get_pure_vapor_pressure(temperature)
        pp_gas             = gas_species.get_partial_pressure(temperature)
        kelvin_term        = particle.surface.kelvin_term(
            radius=radius,
            molar_mass=self.molar_mass,
            mass_concentration=mass_in_particle,
            temperature=temperature,
        )
        delta = ti.ndarray(dtype=ti.f64, shape=mass_in_particle.shape)
        self._kget_pressure_delta(
            mass_in_particle,
            pure_vp,
            pp_gas,
            kelvin_term,
            delta,
            particle.activity,              # template arg
        )
        return delta

    def first_order_mass_transport(
        self,
        particle_radius,
        temperature: float,
        pressure: float,
        dynamic_viscosity: float,
    ):
        if isinstance(particle_radius, np.ndarray):
            r_ti = ti.ndarray(dtype=ti.f64, shape=particle_radius.shape)
            r_ti.from_numpy(particle_radius.astype(np.float64))
        else:
            r_ti = particle_radius
        coeff = ti.ndarray(
            dtype=ti.f64,
            shape=(r_ti.shape[0], self.molar_mass.shape[0]),
        )
        self._kget_first_order_mass_transport(
            r_ti, float(temperature), float(pressure),
            float(dynamic_viscosity),
            self.molar_mass,
            self.accommodation_coefficient,
            coeff
        )
        return coeff        # hand back Taichi array (convert in caller if desired)

    # ───────────────────────── API: dm/dt ───────────────────────────────────
    def mass_transfer_rate(
        self,
        particle: TiParticleRepresentation,
        gas_species: TiGasSpecies,
        temperature: float,
        pressure: float,
        dynamic_viscosity: float | None = None,
    ):
        # radius handling (still allow zero-guard in NumPy)
        radius_np = self._fill_zero_radius(particle.get_radius().to_numpy())
        radius_ti = ti.ndarray(dtype=ti.f64, shape=radius_np.shape)
        radius_ti.from_numpy(radius_np)

        if dynamic_viscosity is None:
            from particula.gas.properties.dynamic_viscosity import get_dynamic_viscosity
            dynamic_viscosity = get_dynamic_viscosity(temperature)

        pressure_delta = self.calculate_pressure_delta(
            particle, gas_species, temperature, radius_np
        )                                          # Ti array
        dm_dt = ti.ndarray(dtype=ti.f64, shape=pressure_delta.shape)
        self._kget_mass_transfer_rate(
            radius_ti, float(temperature), float(pressure),
            float(dynamic_viscosity),
            self.molar_mass,
            self.accommodation_coefficient,
            pressure_delta, dm_dt
        )
        return dm_dt

    # ──────────────────── API: concentration-scaled rate ───────────────────
    def rate(
        self,
        particle: TiParticleRepresentation,
        gas_species: TiGasSpecies,
        temperature: float,
        pressure: float,
    ):
        dm_dt = self.mass_transfer_rate(
            particle=particle,
            gas_species=gas_species,
            temperature=temperature,
            pressure=pressure,
        )
        scaled = ti.ndarray(dtype=ti.f64, shape=dm_dt.shape)
        self._kscale_rate(dm_dt, particle.concentration, scaled)
        return scaled

    # ─────────────────────────── API: step ─────────────────────────────────
    def step(
        self,
        particle: TiParticleRepresentation,
        gas_species: TiGasSpecies,
        temperature: float,
        pressure: float,
        time_step: float,
    ):
        dm_dt = self.mass_transfer_rate(
            particle=particle,
            gas_species=gas_species,
            temperature=temperature,
            pressure=pressure,
        )
        mass_change = ti.ndarray(dtype=ti.f64, shape=dm_dt.shape)
        self._kapply_mass_change(dm_dt, float(time_step), mass_change)

        particle.add_mass(added_mass=mass_change.to_numpy())   # strategy expects NumPy

        if self.update_gases:
            gas_species.add_concentration(
                -mass_change.to_numpy().sum(axis=0)
            )
        return particle, gas_species
