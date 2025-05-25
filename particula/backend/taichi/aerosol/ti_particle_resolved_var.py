"""
Structure of array approach for
Aerosol Dynamics and Particle-Resolved Simulation Representation
many variants, particles, and species.
"""

from dataclasses import dataclass
import taichi as ti
import numpy as np
import particula as par
import particula.backend.taichi.gas.properties as gas_properties
import particula.backend.taichi.particles.properties as particle_properties
import particula.backend.taichi.dynamics.condensation as condensation
import particula.backend.taichi.dynamics as dynamics
from particula.backend.taichi.aerosol.ti_species_field_builder import (
    SpeciesFieldBuilder,
)
from particula.backend.taichi.aerosol.ti_particle_resolved_field import (
    ParticleResolvedFieldBuilder,
)
from particula.backend.taichi.aerosol.ti_environmental_conditions_builder import (
    EnvironmentalConditionsBuilder,
    EnvironmentalConditions,
)


@ti.data_oriented
class TiAerosolParticleResolved_soa:

    def __init__(
        self,
        particle_count: int,
        species_count: int,
        variant_count: int = 1,
        **run_kwargs,
    ):
        # counts & global options
        self.particle_count = particle_count
        self.species_count = species_count
        self.variant_count = variant_count
        self.cfg = EnvironmentalConditions(**run_kwargs)

        # -------- field builders --------------------------------------
        self.species_builder = SpeciesFieldBuilder(
            variant_count, species_count
        )
        self.species = self.species_builder.field  # alias

        self.particle_builder = ParticleResolvedFieldBuilder(
            variant_count, particle_count, species_count
        )
        self.particle_field = self.particle_builder.field  # alias

        # -------- auxiliary 1-D fields --------------------------------
        self.radius = ti.field(ti.f32, shape=(variant_count, particle_count))
        self.total_requested_mass = ti.field(
            ti.f32, shape=(variant_count, species_count)
        )
        self.scaling_factor = ti.field(
            ti.f32, shape=(variant_count, species_count)
        )
        self.particle_concentration = ti.field(
            ti.f32, shape=(variant_count, particle_count)
        )

    # ---------------------------------------------------------------- #
    # NumPy → Taichi loader (delegates to builders)                    #
    # ---------------------------------------------------------------- #
    def setup(
        self,
        *,
        variant_index: int = 0,
        species_masses_np: np.ndarray,
        density_np: np.ndarray,
        molar_mass_np: np.ndarray,
        pure_vapor_pressure_np: np.ndarray,
        vapor_concentration_np: np.ndarray,
        kappa_value_np: np.ndarray,
        surface_tension_np: np.ndarray,
        gas_mass_np: np.ndarray,
        particle_concentration_np: np.ndarray,
    ) -> None:
        v = variant_index
        # per-species
        self.species_builder.load(
            v,
            density=density_np,
            molar_mass=molar_mass_np,
            pure_vapor_pressure=pure_vapor_pressure_np,
            vapor_concentration=vapor_concentration_np,
            kappa=kappa_value_np,
            surface_tension=surface_tension_np,
            gas_mass=gas_mass_np,
        )
        # particle × species
        self.particle_builder.load(v, species_masses=species_masses_np)
        # per-particle
        self._load_particle_concentration(v, particle_concentration_np)

    # ------------------------------------------------------------------ #
    # light-weight getters                                               #
    # ------------------------------------------------------------------ #
    def get_radius(self, v: int = 0) -> np.ndarray:
        return self.radius[v].to_numpy()

    def get_species_masses(self, v: int = 0) -> np.ndarray:
        return self.particle_field[v].mass.to_numpy()

    def get_gas_mass(self, v: int = 0) -> np.ndarray:
        return self.species[v].gas_mass.to_numpy()

    def get_transferable_mass(self, v: int = 0) -> np.ndarray:
        return self.particle_field[v].t_mass.to_numpy()

    # ------------------------------------------------------------------ #
    # core step                                                          #
    # ------------------------------------------------------------------ #
    @ti.kernel
    def _load_particle_concentration(
        self,
        v: ti.i32,
        data: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for p in range(self.particle_count):
            self.particle_concentration[v, p] = data[p]

    @ti.kernel
    def fused_step(self):
        """
        One explicit-Euler step for *all* variants.
        """
        for v, p in ti.ndrange(self.variant_count, self.particle_count):
            # --- derived single-particle props -------------------------
            r_p = particle_properties.fget_particle_radius_via_masses(
                particle_index=p,
                species_masses=self.particle_field[v].mass,
                density=self.species[v].density,
            )
            self.radius[v, p] = r_p

            sigma_eff, rho_eff = (
                particle_properties.fget_mass_weighted_density_and_surface_tension(
                    particle_index=p,
                    species_masses=self.particle_field[v].mass,
                    density=self.species[v].density,
                    surface_tension=self.species[v].sigma,
                )
            )

            for s in range(self.species_count):
                # alias common scalars
                M_i = self.species[v, s].molar_mass
                T = self.cfg.temperature
                P = self.cfg.pressure

                k1 = condensation.fget_first_order_mass_transport_via_system_state(
                    particle_radius=r_p,
                    molar_mass=M_i,
                    mass_accommodation=self.cfg.mass_accommodation,
                    temperature=T,
                    pressure=P,
                    dynamic_viscosity=self.cfg.dynamic_viscosity,
                    diffusion_coefficient=self.cfg.diffusion_coefficient,
                )

                kelvin_radius = particle_properties.fget_kelvin_radius(
                    sigma_eff, rho_eff, M_i, T
                )
                kelvin_term = particle_properties.fget_kelvin_term(
                    r_p, kelvin_radius
                )

                p_g = gas_properties.fget_partial_pressure(
                    self.species[v, s].vapor_conc, M_i, T
                )
                delta_p = particle_properties.fget_partial_pressure_delta(
                    p_g, p_g, kelvin_term
                )

                self.particle_field[v].mtr[p, s] = condensation.fget_mass_transfer_rate(
                    pressure_delta=delta_p,
                    first_order_mass_transport=k1,
                    temperature=T,
                    molar_mass=M_i,
                )

        # --- species-level updates (one loop per variant) --------------
        for v in range(self.variant_count):
            dynamics.update_scaling_factor(
                mass_transport_rate=self.particle_field[v].mtr,
                gas_mass=self.species[v].gas_mass,
                total_requested_mass=self.total_requested_mass[v],
                scaling_factor=self.scaling_factor[v],
                time_step=self.cfg.time_step,
                simulation_volume=self.cfg.simulation_volume,
            )
            condensation.update_transferable_mass(
                time_step=self.cfg.time_step,
                mass_transport_rate=self.particle_field[v].mtr,
                scaling_factor=self.scaling_factor[v],
                transferable_mass=self.particle_field[v].t_mass,
            )
            condensation.update_gas_mass(
                gas_mass=self.species[v].gas_mass,
                species_masses=self.particle_field[v].mass,
                transferable_mass=self.particle_field[v].t_mass,
            )
            condensation.update_species_masses(
                species_masses=self.particle_field[v].mass,
                particle_concentration=self.particle_concentration[v],
                transferable_mass=self.particle_field[v].t_mass,
            )
