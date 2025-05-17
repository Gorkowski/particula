"""Taichi implementation of condensation strategies (isothermal)."""

import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

# ---- Taichi helpers already in repo -----------------------------------
from particula.backend.taichi.particles.properties import (
    fget_vapor_transition_correction,
    fget_friction_factor,
    fget_knudsen_number,
)
from particula.backend.taichi.gas.properties import (
    fget_molecule_mean_free_path,
)
from particula.backend.taichi.dynamics.condensation.ti_mass_transfer import (
    fget_mass_transfer_rate,
    fget_first_order_mass_transport_k,
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
    Taichi version for CondensationIsothermal.
    """

    # ─────────────────────────── constructor ───────────────────────────────
    def __init__(
        self,
        molar_mass: ti.types.ndarray(dtype=ti.f64, ndim=1),
        diffusion_coefficient: ti.field(ti.f64, shape=()),
        accommodation_coefficient: ti.types.ndarray(dtype=ti.f64, ndim=1),
        update_gases: ti.types.ndarray(dtype=ti.int16, ndim=1),
    ):
        # persistent scalar fields
        molar_mass = ti.types.ndarray(dtype=ti.f64, ndim=1)
        diffusion_coefficient = ti.field(ti.f64, shape=())
        accommodation_coefficient = ti.types.ndarray(dtype=ti.f64, ndim=1)
        update_gases = ti.types.ndarray(dtype=ti.int16, ndim=1)

        self.molar_mass = molar_mass
        self.diffusion_coefficient = diffusion_coefficient
        self.accommodation_coefficient = accommodation_coefficient
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
        temperature: ti.f64,
        pressure: ti.f64,
        dynamic_viscosity: ti.f64,
        result: ti.types.ndarray(dtype=ti.f64, ndim=2),
    ):
        """Compute first-order mass-transport coefficient per particle
        per species.
        """
        particle_i = ti.int16(0)
        molar_mass_i = ti.int16(1)
        for particle_i in range(particle_radius.shape[0]):
            for molar_mass_i in range(self.molar_mass.shape[0]):

                mean_free_path = fget_molecule_mean_free_path(
                    molar_mass=self.molar_mass[molar_mass_i],
                    temperature=temperature,
                    pressure=pressure,
                    dynamic_viscosity=dynamic_viscosity,
                )
                knudsen_number = fget_knudsen_number(
                    mean_free_path=mean_free_path,
                    particle_radius=particle_radius[particle_i],
                )
                vapor_transition = fget_vapor_transition_correction(
                    knudsen_number=knudsen_number,
                    mass_accommodation=self.accommodation_coefficient[
                        particle_i
                    ],
                )
                result[particle_i, molar_mass_i] = (
                    fget_first_order_mass_transport_k(
                        r=particle_radius[particle_i],
                        vt=vapor_transition,
                        d=self.diffusion_coefficient[None],
                    )
                )

    @ti.kernel
    def _kget_mass_transfer_rate(  # ← dm/dt  1-D
        self,
        delta_p: ti.types.ndarray(dtype=ti.f64, ndim=2),
        first_order_mass_transport: ti.types.ndarray(dtype=ti.f64, ndim=2),
        temperature: ti.f64,
        result: ti.types.ndarray(dtype=ti.f64, ndim=2),
    ):
        """dm/dt per particle (isothermal)."""
        for particle_i in range(delta_p.shape[0]):
            for molar_mass_i in range(delta_p.shape[1]):
                result[particle_i, molar_mass_i] = fget_mass_transfer_rate(
                    dp=delta_p[particle_i, molar_mass_i],
                    k=first_order_mass_transport[particle_i, molar_mass_i],
                    t=temperature,
                    m=self.molar_mass[molar_mass_i],
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
        mfp = get_molecule_mean_free_path(
            molar_mass=self.molar_mass[None],
            temperature=temperature,
            pressure=pressure,
            dynamic_viscosity=dynamic_viscosity,
        )
        r_np = np.ascontiguousarray(particle_radius, dtype=np.float64)
        K_np = np.empty_like(r_np)
        self._kget_first_order_mass_transport(r_np, mfp, K_np)
        return K_np

    # ───────────────────────── API: dm/dt ───────────────────────────────────
    def mass_transfer_rate(
        self,
        particle: ParticleRepresentation,
        gas_species: GasSpecies,
        temperature: float,
        pressure: float,
        dynamic_viscosity: float | None = None,
    ) -> np.ndarray:
        radius = self._fill_zero_radius(particle.get_radius())
        K = self.first_order_mass_transport(
            particle_radius=radius,
            temperature=temperature,
            pressure=pressure,
            dynamic_viscosity=dynamic_viscosity,
        )
        delta_p = self.calculate_pressure_delta(
            particle=particle,
            gas_species=gas_species,
            temperature=temperature,
            radius=radius,
        )

        delta_p_np = np.ascontiguousarray(delta_p, dtype=np.float64)
        dm_dt = np.empty_like(delta_p_np)
        self._kget_mass_transfer_rate(delta_p_np, K, temperature, dm_dt)
        return dm_dt

    # ──────────────────── API: concentration-scaled rate ───────────────────
    def rate(
        self,
        particle: ParticleRepresentation,
        gas_species: GasSpecies,
        temperature: float,
        pressure: float,
    ) -> np.ndarray:
        dm_dt = self.mass_transfer_rate(
            particle=particle,
            gas_species=gas_species,
            temperature=temperature,
            pressure=pressure,
        )
        concentration = particle.concentration
        if dm_dt.ndim == 2:
            concentration = concentration[:, np.newaxis]
        return dm_dt * concentration

    # ─────────────────────────── API: step ─────────────────────────────────
    def step(
        self,
        particle: ParticleRepresentation,
        gas_species: GasSpecies,
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
        mass_change = get_mass_transfer(
            mass_rate=dm_dt,
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
