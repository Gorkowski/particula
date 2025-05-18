"""Taichi implementation of condensation strategies (isothermal)."""
import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

# ---- Taichi helpers already in repo -----------------------------------
from particula.backend.taichi.particles.properties import (
    fget_vapor_transition_correction,
)

# ---- Python-side utilities (unchanged, lightweight) --------------------
from particula.dynamics.condensation.mass_transfer import get_mass_transfer
from particula.gas import get_molecule_mean_free_path
from particula.particles.representation import ParticleRepresentation
from particula.gas.species import GasSpecies
from particula.particles import get_partial_pressure_delta
import logging

logger = logging.getLogger("particula")
R_GAS = 8.314462618


@ti.data_oriented
class CondensationIsothermal:
    """
    Taichi drop-in replacement for particula.dynamics.condensation.CondensationIsothermal.
    """

    # ─────────────────────────── constructor ───────────────────────────────
    def __init__(
        self,
        molar_mass: float,
        diffusion_coefficient: float = 2e-5,
        accommodation_coefficient: float = 1.0,
        update_gases: bool = True,
    ):
        # persistent scalar fields
        self.molar_mass              = ti.field(ti.f64, shape=())
        self.diffusion_coefficient   = ti.field(ti.f64, shape=())
        self.accommodation_coefficient = ti.field(ti.f64, shape=())

        self.molar_mass[None]            = molar_mass
        self.diffusion_coefficient[None] = diffusion_coefficient
        self.accommodation_coefficient[None] = accommodation_coefficient

        self.update_gases = update_gases

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
        mean_free_path: ti.f64,
        mass_transport_coefficient: ti.types.ndarray(dtype=ti.f64, ndim=1),
    ):
        """Compute first-order mass-transport coefficient per particle."""
        diffusion_coefficient = self.diffusion_coefficient[None]
        accommodation_coefficient = self.accommodation_coefficient[None]
        for i in range(particle_radius.shape[0]):
            radius = ti.max(particle_radius[i], 1e-20)
            knudsen_number = mean_free_path / radius
            correction_factor = fget_vapor_transition_correction(
                knudsen_number,
                accommodation_coefficient,
            )
            mass_transport_coefficient[i] = (
                4.0 * ti.math.pi * radius
                * diffusion_coefficient * correction_factor
            )

    @ti.kernel
    def _kget_mass_transfer_rate(
        self,
        pressure_delta: ti.types.ndarray(dtype=ti.f64, ndim=1),
        mass_transport_coefficient: ti.types.ndarray(dtype=ti.f64, ndim=1),
        temperature: ti.f64,
        mass_rate: ti.types.ndarray(dtype=ti.f64, ndim=1),
    ):
        """dm/dt per particle (isothermal)."""
        molar_mass = self.molar_mass[None]
        gas_constant = ti.static(R_GAS)
        for i in range(pressure_delta.shape[0]):
            mass_rate[i] = (
                mass_transport_coefficient[i]
                * molar_mass
                * pressure_delta[i]
                / (gas_constant * temperature)
            )

    # ─────────────────────── public helpers ────────────────────────────────
    def first_order_mass_transport(
        self,
        particle_radius: np.ndarray,
        temperature: float,
        pressure: float,
        dynamic_viscosity: float | None = None,
    ) -> np.ndarray:
        """Vectorised K(r) using Taichi kernel."""
        mean_free_path = get_molecule_mean_free_path(
            molar_mass=self.molar_mass[None],
            temperature=temperature,
            pressure=pressure,
            dynamic_viscosity=dynamic_viscosity,
        )
        radius_array = np.ascontiguousarray(particle_radius, dtype=np.float64)
        mass_transport_coefficient_array = np.empty_like(radius_array)
        self._kget_first_order_mass_transport(
            radius_array,
            mean_free_path,
            mass_transport_coefficient_array,
        )
        return mass_transport_coefficient_array

    # ───────────────────────── API: dm/dt ───────────────────────────────────
    def mass_transfer_rate(
        self,
        particle: ParticleRepresentation,
        gas_species: GasSpecies,
        temperature: float,
        pressure: float,
        dynamic_viscosity: float | None = None,
    ) -> np.ndarray:
        particle_radius = self._fill_zero_radius(particle.get_radius())
        mass_transport_coefficient = self.first_order_mass_transport(
            particle_radius=particle_radius,
            temperature=temperature,
            pressure=pressure,
            dynamic_viscosity=dynamic_viscosity,
        )
        pressure_delta = self.calculate_pressure_delta(
            particle=particle,
            gas_species=gas_species,
            temperature=temperature,
            radius=particle_radius,
        )

        pressure_delta_np = np.ascontiguousarray(pressure_delta, dtype=np.float64)
        mass_rate = np.empty_like(pressure_delta_np)
        self._kget_mass_transfer_rate(
            pressure_delta_np,
            mass_transport_coefficient,
            temperature,
            mass_rate,
        )
        return mass_rate

    # ──────────────────── API: concentration-scaled rate ───────────────────
    def rate(
        self,
        particle: ParticleRepresentation,
        gas_species: GasSpecies,
        temperature: float,
        pressure: float,
    ) -> np.ndarray:
        mass_rate = self.mass_transfer_rate(
            particle=particle,
            gas_species=gas_species,
            temperature=temperature,
            pressure=pressure,
        )
        concentration = particle.concentration
        if mass_rate.ndim == 2:
            concentration = concentration[:, np.newaxis]
        return mass_rate * concentration

    # ─────────────────────────── API: step ─────────────────────────────────
    def step(
        self,
        particle: ParticleRepresentation,
        gas_species: GasSpecies,
        temperature: float,
        pressure: float,
        time_step: float,
    ):
        mass_rate = self.mass_transfer_rate(
            particle=particle,
            gas_species=gas_species,
            temperature=temperature,
            pressure=pressure,
        )
        mass_change = get_mass_transfer(
            mass_rate=mass_rate,
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
@register("condensation_isothermal", backend="taichi")     # factory key
def TiCondensationIsothermal(**kwargs):
    """Factory wrapper used by dispatch_register."""
    return CondensationIsothermal(**kwargs)
