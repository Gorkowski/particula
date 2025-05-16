"""Taichi implementation of condensation strategies (isothermal)."""
import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

# ---- Taichi helpers already in repo -----------------------------------
from particula.backend.taichi.particles.properties.ti_vapor_correction_module import (
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

    # ────────────────── Δp helper (pure-python, no Taichi) ──────────────────
    def calculate_pressure_delta(
        self,
        particle: ParticleRepresentation,
        gas_species: GasSpecies,
        temperature: float,
        radius: np.ndarray,
    ) -> np.ndarray:
        """Return Δp = p_gas – p_particle ·exp(Kelvin)."""
        mass_conc = particle.get_species_mass()
        pure_vp   = gas_species.get_pure_vapor_pressure(temperature=temperature)
        p_part    = particle.activity.partial_pressure(
                        pure_vapor_pressure=pure_vp,
                        mass_concentration=mass_conc,
                    )
        p_gas     = gas_species.get_partial_pressure(temperature)
        kelvin    = particle.surface.kelvin_term(
                        radius=radius,
                        molar_mass=self.molar_mass[None],
                        mass_concentration=mass_conc,
                        temperature=temperature,
                    )
        return get_partial_pressure_delta(
            partial_pressure_gas=p_gas,
            partial_pressure_particle=p_part,
            kelvin_term=kelvin,
        )

    # ──────────────────────── Taichi kernels ───────────────────────────────
    @ti.kernel
    def _kget_first_order_mass_transport(                 # ← K(r)   1-D
        self,
        r: ti.types.ndarray(dtype=ti.f64, ndim=1),
        mean_free_path: ti.f64,
        result: ti.types.ndarray(dtype=ti.f64, ndim=1),
    ):
        """Compute first-order mass-transport coefficient per particle."""
        D      = self.diffusion_coefficient[None]
        alpha  = self.accommodation_coefficient[None]
        for i in range(r.shape[0]):
            rad = ti.max(r[i], 1e-20)
            kn  = mean_free_path / rad
            fkn = fget_vapor_transition_correction(kn, alpha)
            result[i] = 4.0 * ti.math.pi * rad * D * fkn

    @ti.kernel
    def _kget_mass_transfer_rate(                          # ← dm/dt  1-D
        self,
        delta_p: ti.types.ndarray(dtype=ti.f64, ndim=1),
        K:       ti.types.ndarray(dtype=ti.f64, ndim=1),
        temperature: ti.f64,
        result: ti.types.ndarray(dtype=ti.f64, ndim=1),
    ):
        """dm/dt per particle (isothermal)."""
        M = self.molar_mass[None]
        R = ti.static(R_GAS)
        for i in range(delta_p.shape[0]):
            result[i] = K[i] * M * delta_p[i] / (R * temperature)

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
        K      = self.first_order_mass_transport(
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
        dm_dt      = np.empty_like(delta_p_np)
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
@register("condensation_isothermal", backend="taichi")     # factory key
def TiCondensationIsothermal(**kwargs):
    """Factory wrapper used by dispatch_register."""
    return CondensationIsothermal(**kwargs)
