"""Taichi implementation of condensation_strategies.py."""
import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

# ── Taichi helpers already available ─────────────────────────────────────
from particula.backend.taichi.particles.properties.ti_knudsen_number_module import kget_knudsen_number
from particula.backend.taichi.particles.properties import (
    fget_vapor_transition_correction,
)
from particula.backend.taichi.dynamics.condensation.ti_mass_transfer import (
    fget_first_order_mass_transport_k,
)
from particula.dynamics.condensation.mass_transfer import get_mass_transfer

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
        self.molar_mass = float(np.asarray(self.molar_mass))
        self.diffusion_coefficient = float(np.asarray(self.diffusion_coefficient))

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
            result[i] = fget_first_order_mass_transport_k(
                particle_radius=r[i],
                vapor_transition=self._transition(kn[i]),
                diffusion_coefficient=D,
            )

    @ti.kernel
    def _mass_rate_k(
        self,
        pressure_delta:   ti.types.ndarray(dtype=ti.f64, ndim=1),   # pressure delta
        K:    ti.types.ndarray(dtype=ti.f64, ndim=1),   # first-order coef
        T:    ti.f64,
        out_: ti.types.ndarray(dtype=ti.f64, ndim=1),   # dm/dt
    ):
        """dm/dt =  K · M · Δp /(R·T)."""
        M  = ti.static(self.molar_mass)
        R  = 8.314462618
        for i in range(pressure_delta.shape[0]):
            out_[i] = K[i] * M * pressure_delta[i] / (R * T)


    def mass_transfer_rate(
        self,
        particle: ParticleRepresentation,
        gas_species: GasSpecies,
        temperature: float,
        pressure: float,
        dynamic_viscosity=None,
    ):
        """Return dm/dt per particle using Taichi kernels."""
        radius = self._fill_zero_radius(particle.get_radius())
        r_t   = ti.ndarray(dtype=ti.f64, shape=radius.shape)
        kn_t  = ti.ndarray(dtype=ti.f64, shape=radius.shape)
        K_t   = ti.ndarray(dtype=ti.f64, shape=radius.shape)
        dP_t  = ti.ndarray(dtype=ti.f64, shape=radius.shape)
        dm_t  = ti.ndarray(dtype=ti.f64, shape=radius.shape)
        r_t.from_numpy(radius)
        # mean-free-path (scalar) → broadcasted Taichi array
        mean_free_path_scalar = self.mean_free_path(
            temperature=temperature,
            pressure=pressure,
            dynamic_viscosity=dynamic_viscosity,
        )
        mfp_np = np.full_like(radius, mean_free_path_scalar, dtype=np.float64)
        mfp_t  = ti.ndarray(dtype=ti.f64, shape=radius.shape)
        mfp_t.from_numpy(mfp_np)
        kget_knudsen_number(mfp_t, r_t, kn_t)
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
        """Return condensation rate per particle/bin using Taichi kernels."""
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
        """Advance the system by one time step using Taichi kernels."""
        mass_rate = self.mass_transfer_rate(
            particle=particle,
            gas_species=gas_species,
            temperature=temperature,
            pressure=pressure,
        )
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
