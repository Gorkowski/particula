"""
Integrated Aerosol Dynamics and Particle-Resolved Simulation Representation
"""

from dataclasses import dataclass
import taichi as ti
import numpy as np
import particula as par
import particula.backend.taichi.gas.properties as gas_properties
import particula.backend.taichi.particles.properties as particle_properties
import particula.backend.taichi.dynamics.condensation as condensation
import particula.backend.taichi.dynamics as dynamics


# taichi data class for input conversion and kernel execution
@ti.data_oriented
class TiAerosolParticleResolved:
    """
    Aerosol particle resolved simulation class. This class is used to
    hold the data and methods for the particle-resolved simulation.
    """

    def __init__(
        self,
        particle_count: int,
        species_count: int,
        *,
        temperature: float = 298.15,
        pressure: float = 101_325.0,
        mass_accommodation: float = 0.5,
        dynamic_viscosity: float = 1.8e-5,
        diffusion_coefficient: float = 2.0e-5,
        time_step: float = 10.0,
        simulation_volume: float = 1.0e-6,
    ):
        """
        Arguments:

        """

        # input data conversion to Taichi fields --------------------------------

        # apply input species masses to the field
        self.species_masses = ti.field(
            float, shape=(particle_count, species_count), name="species_masses"
        )

        # create density field
        self.density = ti.field(float, shape=(species_count,), name="density")
        # create molar mass field
        self.molar_mass = ti.field(
            ti.float64, shape=(species_count,), name="molar_mass"
        )
        # create pure vapor pressure field
        self.pure_vapor_pressure = ti.field(
            float, shape=(species_count,), name="pure_vapor_pressure"
        )
        # create vapor concentration field
        self.vapor_concentration = ti.field(
            float, shape=(species_count,), name="vapor_concentration"
        )
        # create kappa value field
        self.kappa_value = ti.field(
            float, shape=(species_count,), name="kappa_value"
        )
        # create surface tension field
        self.surface_tension = ti.field(
            float, shape=(species_count,), name="surface_tension"
        )

        # create gas mass field
        self.gas_mass = ti.field(
            float, shape=(species_count,), name="gas_mass"
        )

        # create particle concentration field
        self.particle_concentration = ti.field(
            float, shape=(particle_count,), name="particle_concentration"
        )

        # scalar parameters as members
        self.temperature = ti.static(temperature)
        self.pressure = ti.static(pressure)
        self.mass_accommodation = ti.static(mass_accommodation)
        self.dynamic_viscosity = ti.static(dynamic_viscosity)
        self.diffusion_coefficient = ti.static(diffusion_coefficient)
        self.time_step = ti.static(time_step)
        self.simulation_volume = ti.static(simulation_volume)
        # temporary fields
        self.radius = ti.field(float, shape=(particle_count,), name="radii")
        self.mass_transport_rate = ti.field(
            float,
            shape=(particle_count, species_count),
            name="mass_transport_rate",
        )
        self.transferable_mass = ti.field(
            float,
            shape=(particle_count, species_count),
            name="transferable_mass",
        )
        self.total_requested_mass = ti.field(
            float, shape=(species_count,), name="total_requested_mass"
        )
        self.scaling_factor = ti.field(
            float, shape=(species_count,), name="scaling_factor"
        )

    # ------------------------------------------------------------------
    # I/O helper – populate fields from NumPy arrays
    # ------------------------------------------------------------------
    def setup(
        self,
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
        """Copy the supplied NumPy arrays into the internal Taichi fields."""
        self.species_masses.from_numpy(species_masses_np)
        self.density.from_numpy(density_np)
        self.molar_mass.from_numpy(molar_mass_np)
        self.pure_vapor_pressure.from_numpy(pure_vapor_pressure_np)
        self.vapor_concentration.from_numpy(vapor_concentration_np)
        self.kappa_value.from_numpy(kappa_value_np)
        self.surface_tension.from_numpy(surface_tension_np)
        self.gas_mass.from_numpy(gas_mass_np)
        self.particle_concentration.from_numpy(particle_concentration_np)

    # --------------------------------------------------------------------- #
    # Convenience wrappers that delegate to the module-level kernels
    # --------------------------------------------------------------------- #

    def get_radius(self) -> np.ndarray:
        return self.radius.to_numpy()

    def get_species_masses(self) -> np.ndarray:
        return self.species_masses.to_numpy()

    def get_gas_mass(self) -> np.ndarray:
        return self.gas_mass.to_numpy()

    def get_transferable_mass(self) -> np.ndarray:
        return self.transferable_mass.to_numpy()

    @ti.kernel
    def fused_step(self):
        """
        Perform a single step of the particle-resolved aerosol dynamics simulation.
        This method calculates the mass transport rates for each particle and species,
        updates the scaling factors, and adjusts the gas mass and species masses
        based on the mass transfer rates.
        """
        for p in range(self.species_masses.shape[0]):

            particle_radius = (
                particle_properties.fget_particle_radius_via_masses(
                    particle_index=p,
                    species_masses=self.species_masses,
                    density=self.density,
                )
            )

            effective_surface_tension, effective_density = (
                particle_properties.fget_mass_weighted_density_and_surface_tension(
                    particle_index=p,
                    species_masses=self.species_masses,
                    density=self.density,
                    surface_tension=self.surface_tension,
                )
            )

            for s in range(self.density.shape[0]):

                first_order_mass_transport_coefficient = condensation.fget_first_order_mass_transport_via_system_state(
                    particle_radius=particle_radius,
                    molar_mass=self.molar_mass[s],
                    mass_accommodation=self.mass_accommodation,
                    temperature=self.temperature,
                    pressure=self.pressure,
                    dynamic_viscosity=self.dynamic_viscosity,
                    diffusion_coefficient=self.diffusion_coefficient,
                )

                kelvin_radius = particle_properties.fget_kelvin_radius(
                    effective_surface_tension,
                    effective_density,
                    self.molar_mass[s],
                    self.temperature,
                )
                kelvin_term = particle_properties.fget_kelvin_term(
                    particle_radius, kelvin_radius
                )

                partial_pressure_gas = gas_properties.fget_partial_pressure(
                    self.vapor_concentration[s],
                    self.molar_mass[s],
                    self.temperature,
                )
                pressure_delta = (
                    particle_properties.fget_partial_pressure_delta(
                        partial_pressure_gas, partial_pressure_gas, kelvin_term
                    )
                )

                self.mass_transport_rate[p, s] = (
                    condensation.fget_mass_transfer_rate(
                        pressure_delta=pressure_delta,
                        first_order_mass_transport=first_order_mass_transport_coefficient,
                        temperature=self.temperature,
                        molar_mass=self.molar_mass[s],
                    )
                )
        dynamics.update_scaling_factor(
            mass_transport_rate=self.mass_transport_rate,
            gas_mass=self.gas_mass,
            total_requested_mass=self.total_requested_mass,
            scaling_factor=self.scaling_factor,
            time_step=self.time_step,
            simulation_volume=self.simulation_volume,
        )
        condensation.update_transferable_mass(
            time_step=self.time_step,
            mass_transport_rate=self.mass_transport_rate,
            scaling_factor=self.scaling_factor,
            transferable_mass=self.transferable_mass,
        )
        condensation.update_gas_mass(
            gas_mass=self.gas_mass,
            species_masses=self.species_masses,
            transferable_mass=self.transferable_mass,
        )
        condensation.update_species_masses(
            species_masses=self.species_masses,
            particle_concentration=self.particle_concentration,
            transferable_mass=self.transferable_mass,
        )


# --------------------------------------------------------------------- #
# 1.  Builder classes                                                   #
# --------------------------------------------------------------------- #
class SpeciesFieldBuilder:
    """Creates the per-species SoA field and offers handy load helpers."""

    def __init__(self, variant_count: int, species_count: int):
        self.variant_count = variant_count
        self.species_count = species_count

        self.Species = ti.types.struct(
            density=ti.f32,
            molar_mass=ti.f32,
            pure_vapor_pressure=ti.f32,
            vapor_concentration=ti.f32,
            kappa=ti.f32,
            surface_tension=ti.f32,
            gas_mass=ti.f32,
        )
        self.field = self.Species.field(shape=(variant_count, species_count))

    # ------------ helper: copy one variant from NumPy -----------------
    def load(
        self,
        v: int,
        *,
        density: np.ndarray,
        molar_mass: np.ndarray,
        pure_vapor_pressure: np.ndarray,
        vapor_concentration: np.ndarray,
        kappa: np.ndarray,
        surface_tension: np.ndarray,
        gas_mass: np.ndarray,
    ) -> None:
        """Fill variant `v` with NumPy arrays (1-D, length = species)."""
        self.field[v].density.from_numpy(density)
        self.field[v].molar_mass.from_numpy(molar_mass)
        self.field[v].pure_vapor_pressure.from_numpy(pure_vapor_pressure)
        self.field[v].vapor_conc.from_numpy(vapor_concentration)
        self.field[v].kappa.from_numpy(kappa)
        self.field[v].sigma.from_numpy(surface_tension)
        self.field[v].gas_mass.from_numpy(gas_mass)


class P2SFieldBuilder:
    """Creates the (variant × particle × species) state field."""

    def __init__(
        self, variant_count: int, particle_count: int, species_count: int
    ):
        self.variant_count = variant_count
        self.particle_count = particle_count
        self.species_count = species_count

        self.P2S = ti.types.struct(
            mass=ti.f32,  # species_masses
            mtr=ti.f32,  # mass_transport_rate
            t_mass=ti.f32,  # transferable_mass
        )
        self.field = self.P2S.field(
            shape=(variant_count, particle_count, species_count)
        )

    def load(self, v: int, *, species_masses: np.ndarray) -> None:
        """Copy `(particle, species)` mass matrix for one variant."""
        self.field[v].mass.from_numpy(species_masses)
        # mtr and t_mass start at zero → no load needed.


# --------------------------------------------------------------------- #
# 2.  Main solver class (uses the builders)                             #
# --------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class EnvironmentalConditions:
    temperature: float = 298.15
    pressure: float = 101_325.0
    mass_accommodation: float = 0.5
    dynamic_viscosity: float = 1.8e-5
    diffusion_coefficient: float = 2.0e-5
    time_step: float = 10.0
    simulation_volume: float = 1.0e-6


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

        self.p2s_builder = P2SFieldBuilder(
            variant_count, particle_count, species_count
        )
        self.p2s = self.p2s_builder.field  # alias

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
        self.p2s_builder.load(v, species_masses=species_masses_np)
        # per-particle
        self.particle_concentration[v].from_numpy(particle_concentration_np)

    # ------------------------------------------------------------------ #
    # light-weight getters                                               #
    # ------------------------------------------------------------------ #
    def get_radius(self, v: int = 0) -> np.ndarray:
        return self.radius[v].to_numpy()

    def get_species_masses(self, v: int = 0) -> np.ndarray:
        return self.p2s[v].mass.to_numpy()

    def get_gas_mass(self, v: int = 0) -> np.ndarray:
        return self.species[v].gas_mass.to_numpy()

    def get_transferable_mass(self, v: int = 0) -> np.ndarray:
        return self.p2s[v].t_mass.to_numpy()

    # ------------------------------------------------------------------ #
    # core step                                                          #
    # ------------------------------------------------------------------ #
    @ti.kernel
    def fused_step(self):
        """
        One explicit-Euler step for *all* variants.
        """
        for v, p in ti.ndrange(self.variant_count, self.particle_count):
            # --- derived single-particle props -------------------------
            r_p = particle_properties.fget_particle_radius_via_masses(
                particle_index=p,
                species_masses=self.p2s[v].mass,
                density=self.species[v].density,
            )
            self.radius[v, p] = r_p

            sigma_eff, rho_eff = (
                particle_properties.fget_mass_weighted_density_and_surface_tension(
                    particle_index=p,
                    species_masses=self.p2s[v].mass,
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

                self.p2s[v].mtr[p, s] = condensation.fget_mass_transfer_rate(
                    pressure_delta=delta_p,
                    first_order_mass_transport=k1,
                    temperature=T,
                    molar_mass=M_i,
                )

        # --- species-level updates (one loop per variant) --------------
        for v in range(self.variant_count):
            dynamics.update_scaling_factor(
                mass_transport_rate=self.p2s[v].mtr,
                gas_mass=self.species[v].gas_mass,
                total_requested_mass=self.total_requested_mass[v],
                scaling_factor=self.scaling_factor[v],
                time_step=self.cfg.time_step,
                simulation_volume=self.cfg.simulation_volume,
            )
            condensation.update_transferable_mass(
                time_step=self.cfg.time_step,
                mass_transport_rate=self.p2s[v].mtr,
                scaling_factor=self.scaling_factor[v],
                transferable_mass=self.p2s[v].t_mass,
            )
            condensation.update_gas_mass(
                gas_mass=self.species[v].gas_mass,
                species_masses=self.p2s[v].mass,
                transferable_mass=self.p2s[v].t_mass,
            )
            condensation.update_species_masses(
                species_masses=self.p2s[v].mass,
                particle_concentration=self.particle_concentration[v],
                transferable_mass=self.p2s[v].t_mass,
            )
