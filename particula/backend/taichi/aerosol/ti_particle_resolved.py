"""
Integrated Aerosol Dynamics and Particle-Resolved Simulation Representation
"""

import taichi as ti
import numpy as np
import particula as par
import particula.backend.taichi.gas.properties as gas_properties
import particula.backend.taichi.particles.properties as particle_properties
import particula.backend.taichi.dynamics.condensation as condensation

GAS_CONSTANT = ti.static(par.util.constants.GAS_CONSTANT)


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

    # ------------------------------------------------------------------
    # --------------------------------------------------------------------- #
    # Instance-aware Taichi helpers
    # --------------------------------------------------------------------- #

    @ti.func
    def update_transferable_mass(self, p: int, time_step: float):
        for s in ti.ndrange(self.density.shape[0]):
            self.transferable_mass[p, s] = (
                self.mass_transport_rate[p, s]
                * time_step
                * self.scaling_factor[s]
            )

    @ti.func
    def update_gas_mass(self):
        for j in ti.ndrange(self.gas_mass.shape[0]):
            species_mass = 0.0
            for i in ti.ndrange(self.species_masses.shape[0]):
                species_mass += self.transferable_mass[i, j]
            self.gas_mass[j] -= species_mass
            self.gas_mass[j] = ti.max(self.gas_mass[j], 0.0)

    @ti.func
    def update_species_masses(self):
        for i in range(self.species_masses.shape[0]):
            for j in range(self.species_masses.shape[1]):
                if self.particle_concentration[i] > 0.0:
                    self.species_masses[i, j] = (
                        self.species_masses[i, j]
                        * self.particle_concentration[i]
                        + self.transferable_mass[i, j]
                    ) / self.particle_concentration[i]
                else:
                    self.species_masses[i, j] = 0.0
                self.species_masses[i, j] = ti.max(
                    self.species_masses[i, j], 0.0
                )

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
        update_scaling_factor_refactor2(
            mass_transport_rate=self.mass_transport_rate,
            gas_mass=self.gas_mass,
            total_requested_mass=self.total_requested_mass,
            scaling_factor=self.scaling_factor,
            time_step=self.time_step,
            simulation_volume=self.simulation_volume,
        )
        for i in ti.ndrange(self.species_masses.shape[0]):
            self.update_transferable_mass(i, self.time_step)
        self.update_gas_mass()
        self.update_species_masses()

