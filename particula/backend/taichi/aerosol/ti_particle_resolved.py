"""
Integrated Aerosol Dynamics and Particle-Resolved Simulation Representation
"""

import taichi as ti
import numpy as np
import particula as par
import particula.backend.taichi.gas.properties as gas_properties
import particula.backend.taichi.particles.properties as particle_properties
import particula.backend.taichi.dynamics.condensation as condensation

np_type = np.float64
ti.init(arch=ti.cpu, default_fp=ti.f64, default_ip=ti.i32, debug=True)

GAS_CONSTANT = par.util.constants.GAS_CONSTANT

# particle resolved data, 100 particles, 10 species
particle_count = 20_000
species_count = 10
input_species_masses = np.random.rand(particle_count, species_count).astype(
    np_type
)
input_density = np.random.rand(species_count).astype(np_type)
input_molar_mass = np.abs(np.random.rand(species_count).astype(np_type))
input_pure_vapor_pressure = np.abs(
    np.random.rand(species_count).astype(np_type)
)
input_vapor_concentration = np.abs(
    np.random.rand(species_count).astype(np_type)
)
input_kappa_value = np.abs(np.random.rand(species_count).astype(np_type))
input_surface_tension = np.abs(np.random.rand(species_count).astype(np_type))
input_temperature = 298.15  # K
input_pressure = 101325.0  # Pa
input_mass_accommodation = 0.5
input_dynamic_viscosity = par.gas.get_dynamic_viscosity(
    temperature=input_temperature
)
input_diffusion_coefficient = 2.0e-5  # m^2/s
input_time_step = 10  # seconds
input_simulation_volume = 1.0e-6  # m^3


# available gas-phase mass [kg] per species  (positive values)
input_gas_mass = np.abs(np.random.rand(species_count).astype(np_type))
# particle number concentration [#/m³] for every particle (use 1.0 for now)
input_particle_concentration = np.ones(particle_count, dtype=np_type)


# taichi data class for input conversion and kernel execution
@ti.data_oriented
class TiAerosolParticleResolved():
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
        # --- basic shape checks -------------------------------------------------
        # particle_count, species_count are now direct arguments

        # input data conversion to Taichi fields --------------------------------

        # apply input species masses to the field
        self.species_masses = ti.field(
            float, shape=(particle_count, species_count), name="species_masses"
        )

        # create density field
        self.density = ti.field(float, shape=(species_count,), name="density")
        # create molar mass field
        self.molar_mass = ti.field(
            float, shape=(species_count,), name="molar_mass"
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
        self.temperature         = ti.static(temperature)
        self.pressure            = ti.static(pressure)
        self.mass_accommodation  = ti.static(mass_accommodation)
        self.dynamic_viscosity   = ti.static(dynamic_viscosity)
        self.diffusion_coefficient = ti.static(diffusion_coefficient)
        self.time_step           = ti.static(time_step)
        self.simulation_volume   = ti.static(simulation_volume)
        # temporary fields
        self.radius = ti.field(float, shape=(particle_count,), name="radii")
        self.mass_transport_rate = ti.field(
            float,
            shape=(particle_count, species_count),
            name="mass_transport_rate",
        )
        self.first_order_coefficient = ti.field(
            float,
            shape=(particle_count, species_count),
            name="first_order_coefficient",
        )
        self.partial_pressure = ti.field(
            float,
            shape=(particle_count, species_count),
            name="partial_pressure",
        )
        self.pressure_delta = ti.field(
            float, shape=(particle_count, species_count), name="pressure_delta"
        )
        self.kelvin_term = ti.field(
            float, shape=(particle_count, species_count), name="kelvin_term"
        )
        self.kelvin_radius = ti.field(
            float, shape=(particle_count, species_count), name="kelvin_radius"
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
        self.species_masses.from_numpy(species_masses_np.astype(np_type))
        self.density.from_numpy(density_np.astype(np_type))
        self.molar_mass.from_numpy(molar_mass_np.astype(np_type))
        self.pure_vapor_pressure.from_numpy(
            pure_vapor_pressure_np.astype(np_type)
        )
        self.vapor_concentration.from_numpy(
            vapor_concentration_np.astype(np_type)
        )
        self.kappa_value.from_numpy(kappa_value_np.astype(np_type))
        self.surface_tension.from_numpy(surface_tension_np.astype(np_type))
        self.gas_mass.from_numpy(gas_mass_np.astype(np_type))
        self.particle_concentration.from_numpy(
            particle_concentration_np.astype(np_type)
        )

    # --------------------------------------------------------------------- #
    # Instance-aware Taichi helpers
    # --------------------------------------------------------------------- #

    @ti.func
    def update_radius(self, p: int):
        volume = 0.0
        for s in ti.ndrange(self.density.shape[0]):
            volume += self.species_masses[p, s] / self.density[s]
        r_p = ti.pow(3.0 * volume / (4.0 * ti.math.pi), 1.0 / 3.0)
        self.radius[p] = r_p

    @ti.func
    def update_first_order_coefficient(self, p: int):
        r_p = self.radius[p]
        for s in ti.ndrange(self.molar_mass.shape[0]):
            k1 = condensation.fget_first_order_mass_transport_via_system_state(
                particle_radius=r_p,
                molar_mass=self.molar_mass[s],
                mass_accommodation=self.mass_accommodation,
                temperature=self.temperature,
                pressure=self.pressure,
                dynamic_viscosity=self.dynamic_viscosity,
                diffusion_coefficient=self.diffusion_coefficient,
            )
            self.first_order_coefficient[p, s] = k1

    @ti.func
    def update_kelvin_term(self, p: int):
        r_p = self.radius[p]
        w_mass = 0.0
        sig_sum = 0.0
        rho_sum = 0.0
        for s in ti.ndrange(self.density.shape[0]):
            w = self.species_masses[p, s]
            w_mass += w
            sig_sum += self.surface_tension[s] * w
            rho_sum += self.density[s] * w
        sig_eff = sig_sum / w_mass
        rho_eff = rho_sum / w_mass
        for s in ti.ndrange(self.density.shape[0]):
            r_k = particle_properties.fget_kelvin_radius(
                sig_eff, rho_eff, self.molar_mass[s], self.temperature
            )
            kel = particle_properties.fget_kelvin_term(r_p, r_k)
            self.kelvin_term[p, s] = kel

    @ti.func
    def update_partial_pressure(self, p: int):
        for s in ti.ndrange(self.density.shape[0]):
            p_g = gas_properties.fget_partial_pressure(
                self.vapor_concentration[s],
                self.molar_mass[s],
                self.temperature,
            )
            kel = self.kelvin_term[p, s]
            dP = particle_properties.fget_partial_pressure_delta(p_g, p_g, kel)
            self.pressure_delta[p, s] = dP

    @ti.func
    def update_mass_transport_rate(self, p: int):
        for s in ti.ndrange(self.density.shape[0]):
            k1 = self.first_order_coefficient[p, s]
            dP = self.pressure_delta[p, s]
            self.mass_transport_rate[p, s] = (
                k1
                * dP
                * self.molar_mass[s]
                / (par.util.constants.GAS_CONSTANT * self.temperature)
            )

    @ti.func
    def update_scaling_factors(self, time_step: float):
        for j in ti.ndrange(self.scaling_factor.shape[0]):
            self.total_requested_mass[j] = 0.0
        for i, j in ti.ndrange(
            self.species_masses.shape[0], self.species_masses.shape[1]
        ):
            self.particle_concentration[i] = 1.0 / self.simulation_volume
            self.total_requested_mass[j] += (
                self.mass_transport_rate[i, j]
                * time_step
                * self.particle_concentration[i]
            )
        for j in ti.ndrange(self.scaling_factor.shape[0]):
            self.scaling_factor[j] = 1.0
            if self.total_requested_mass[j] > self.gas_mass[j]:
                self.scaling_factor[j] = (
                    self.gas_mass[j] / self.total_requested_mass[j]
                )

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
        for i in ti.ndrange(self.species_masses.shape[0]):
            for j in ti.ndrange(self.species_masses.shape[1]):
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
        for p in ti.ndrange(self.species_masses.shape[0]):
            # a) radius
            volume = 0.0
            for s in ti.ndrange(self.density.shape[0]):
                volume += self.species_masses[p, s] / self.density[s]
            r_p = ti.pow(3.0 * volume / (4.0 * ti.math.pi), 1.0 / 3.0)
            self.radius[p] = r_p

            # b) effective bulk properties (weighted averages)
            w_mass = 0.0
            sig_sum = 0.0
            rho_sum = 0.0
            for s in ti.ndrange(self.density.shape[0]):
                w = self.species_masses[p, s]
                w_mass += w
                sig_sum += self.surface_tension[s] * w
                rho_sum += self.density[s] * w
            sig_eff = sig_sum / w_mass
            rho_eff = rho_sum / w_mass

            for s in ti.ndrange(self.density.shape[0]):
                k1 = condensation.fget_first_order_mass_transport_via_system_state(
                    particle_radius=r_p,
                    molar_mass=self.molar_mass[s],
                    mass_accommodation=self.mass_accommodation,
                    temperature=self.temperature,
                    pressure=self.pressure,
                    dynamic_viscosity=self.dynamic_viscosity,
                    diffusion_coefficient=self.diffusion_coefficient,
                )
                self.first_order_coefficient[p, s] = k1

                r_k = particle_properties.fget_kelvin_radius(
                    sig_eff, rho_eff, self.molar_mass[s], self.temperature
                )
                kel = particle_properties.fget_kelvin_term(r_p, r_k)
                self.kelvin_term[p, s] = kel

                p_g = gas_properties.fget_partial_pressure(
                    self.vapor_concentration[s],
                    self.molar_mass[s],
                    self.temperature,
                )
                dP = particle_properties.fget_partial_pressure_delta(
                    p_g, p_g, kel
                )
                self.pressure_delta[p, s] = dP

                self.mass_transport_rate[p, s] = (
                    k1
                    * dP
                    * self.molar_mass[s]
                    / (par.util.constants.GAS_CONSTANT * self.temperature)
                )
        self.update_scaling_factors(self.time_step)
        for i in ti.ndrange(self.species_masses.shape[0]):
            self.update_transferable_mass(i, self.time_step)
        self.update_gas_mass()
        self.update_species_masses()
