"""Taichi implementation of condensation_strategies.py."""
import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

# ── Taichi helpers already available ─────────────────────────────────────
from particula.backend.taichi.particles.properties.ti_knudsen_number_module import (
    kget_knudsen_number,
)
from particula.backend.taichi.dynamics.mass_transfer.ti_vapor import (
    fget_vapor_transition_correction,
)
from particula.backend.taichi.dynamics.mass_transfer.ti_mass_transfer import (
    kget_first_order_mass_transport_k,
    kget_mass_transfer_rate,
)

# Python-side objects that stay in NumPy land
from particula.particles.representation import ParticleRepresentation
from particula.gas.species import GasSpecies
from particula.dynamics.condensation.condensation_strategies import (
    CondensationStrategy,
)

ti.init(arch=ti.cpu, default_fp=ti.f64)


@ti.data_oriented
class TiCondensationIsothermal(CondensationStrategy):
    """Taichi drop-in replacement of CondensationIsothermal."""

    def __init__(
        self,
        molar_mass,
        diffusion_coefficient=2e-5,
        accommodation_coefficient=1.0,
        update_gases=True,
    ):
        super().__init__(
            molar_mass=molar_mass,
            diffusion_coefficient=diffusion_coefficient,
            accommodation_coefficient=accommodation_coefficient,
            update_gases=update_gases,
        )

    @ti.func
    def _transition(self, kn: ti.f64) -> ti.f64:
        """Cunningham/transition correction f(Kn, α)."""
        return fget_vapor_transition_correction(kn, ti.static(self.accommodation_coefficient))

    @ti.kernel
    def _first_order_k(
        self,
        r: ti.types.ndarray(dtype=ti.f64, ndim=1),      # particle radius
        kn: ti.types.ndarray(dtype=ti.f64, ndim=1),     # knudsen number
        result: ti.types.ndarray(dtype=ti.f64, ndim=1), # out: K
    ):
        """Vectorised K = 4π r D f(Kn, α)."""
        D = ti.static(self.diffusion_coefficient)
        for i in range(r.shape[0]):
            result[i] = 4.0 * ti.math.pi * r[i] * D * self._transition(kn[i])

    @ti.kernel
    def _mass_rate_k(
        self,
        Δp:   ti.types.ndarray(dtype=ti.f64, ndim=1),   # pressure delta
        K:    ti.types.ndarray(dtype=ti.f64, ndim=1),   # first-order coef
        T:    ti.f64,
        out_: ti.types.ndarray(dtype=ti.f64, ndim=1),   # dm/dt
    ):
        """dm/dt =  K · M · Δp /(R·T)."""
        M  = ti.static(self.molar_mass)
        R  = 8.314462618
        for i in range(Δp.shape[0]):
            out_[i] = K[i] * M * Δp[i] / (R * T)

    def _fill_zero_radius(
        self, radius: np.ndarray
    ) -> np.ndarray:
        """Fill zero radius values with the maximum radius. The concentration
        value of zero will ensure that the rate of condensation is zero. The
        fill is necessary to avoid division by zero in the array operations.

        Arguments:
            - radius : The radius of the particles.

        Returns:
            - radius : The radius of the particles with zero values filled.

        Raises:
            - Warning : If all radius values are zero.
        """
        if np.max(radius) == 0:
            import logging
            message = (
                "All radius values are zero, radius set to 1 m for "
                "condensation calculations. This should be ignored as the "
                "particle concentration would also be zero."
            )
            logging.warning(message)
            radius = np.where(radius == 0, 1, radius)
        return np.where(radius == 0, np.max(radius), radius)

    def mass_transfer_rate(
        self,
        particle: ParticleRepresentation,
        gas_species: GasSpecies,
        temperature: float,
        pressure: float,
        dynamic_viscosity=None,
    ):
        radius = self._fill_zero_radius(particle.get_radius())
        r_t   = ti.ndarray(ti.f64, radius.shape)
        kn_t  = ti.ndarray(ti.f64, radius.shape)
        K_t   = ti.ndarray(ti.f64, radius.shape)
        dP_t  = ti.ndarray(ti.f64, radius.shape)
        dm_t  = ti.ndarray(ti.f64, radius.shape)
        r_t.from_numpy(radius)
        # mean free path is computed in Python, as in the original
        mean_free_path = self.mean_free_path(
            temperature=temperature,
            pressure=pressure,
            dynamic_viscosity=dynamic_viscosity,
        )
        kget_knudsen_number(mean_free_path, r_t, kn_t)
        self._first_order_k(r_t, kn_t, K_t)
        dP_t.from_numpy(self.calculate_pressure_delta(
            particle, gas_species, temperature, radius
        ))
        self._mass_rate_k(dP_t, K_t, temperature, dm_t)
        return dm_t.to_numpy()

    def rate(
        self,
        particle: ParticleRepresentation,
        gas_species: GasSpecies,
        temperature: float,
        pressure: float,
    ):
        mass_rate = self.mass_transfer_rate(
            particle=particle,
            gas_species=gas_species,
            temperature=temperature,
            pressure=pressure,
        )
        if mass_rate.ndim == 2:  # Multiple gas species  # type: ignore
            concentration = particle.concentration[:, np.newaxis]
        else:
            concentration = particle.concentration
        rates = mass_rate * concentration
        return rates

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
        from particula.dynamics.condensation.mass_transfer import get_mass_transfer
        mass_transfer = get_mass_transfer(
            mass_rate=mass_rate,  # type: ignore
            time_step=time_step,
            gas_mass=gas_species.get_concentration(),  # type: ignore
            particle_mass=particle.get_species_mass(),
            particle_concentration=particle.get_concentration(),
        )
        particle.add_mass(added_mass=mass_transfer)
        if self.update_gases:
            gas_species.add_concentration(
                added_concentration=-mass_transfer.sum(axis=0)
            )
        return particle, gas_species


@register("condensation_isothermal", backend="taichi")
def _ti_iso(**kwargs):
    """Factory wrapper for backend dispatch."""
    return TiCondensationIsothermal(**kwargs)
