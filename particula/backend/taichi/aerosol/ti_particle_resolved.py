"""
Integrated Aerosol Dynamics and Particle-Resolved Simulation Representation
"""

import taichi as ti
import numpy as np
import particula as par
import particula.backend.taichi.gas.properties as gas_properties
import particula.backend.taichi.particles.properties as particle_properties
import particula.backend.taichi.dynamics.condensation as condensation
from particula.backend.benchmark import get_function_benchmark

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
input_dynamic_viscosity = par.gas.get_dynamic_viscosity(temperature=input_temperature)
input_diffusion_coefficient = 2.0e-5  # m^2/s
input_time_step = 10  # seconds
input_simulation_volume = 1.0e-6  # m^3


# available gas-phase mass [kg] per species  (positive values)
input_gas_mass = np.abs(np.random.rand(species_count).astype(np_type))
# particle number concentration [#/m³] for every particle (use 1.0 for now)
input_particle_concentration = np.ones(particle_count, dtype=np_type)






# taichi data class for input conversion and kernel execution
@ti.data_oriented
class TiAerosolParticleResolved:
    """
    Lightweight wrapper that converts NumPy inputs to the global Taichi
    fields defined in this module and exposes convenience methods that
    call the already-implemented kernels (`simulation_step`, `fused_step`,
    etc.).

    NOTE
    ----
    The class does **not** allocate its own private Taichi fields – it
    simply writes into the module-level fields that all kernels already
    reference.  This avoids refactoring the whole solver while still
    giving users a clean, object-based interface.
    """

    def __init__(
        self,
        input_species_masses: np.ndarray,
        input_density: np.ndarray,
        input_molar_mass: np.ndarray,
        input_pure_vapor_pressure: np.ndarray,
        input_vapor_concentration: np.ndarray,
        input_kappa_value: np.ndarray,
        input_surface_tension: np.ndarray,
        input_gas_mass: np.ndarray,
        input_particle_concentration: np.ndarray,
    ):
        """
        Parameters
        ----------
        species_masses_np : (P, S) ndarray
            Mass of each species per particle  [kg].
        density_np : (S,) ndarray
            Bulk density of each species       [kg m⁻³].
        molar_mass_np : (S,) ndarray
            Molar mass of each species         [kg mol⁻¹].
        pure_vapor_pressure_np : (S,) ndarray
            Pure vapor pressure of each species [Pa].
        vapor_concentration_np : (S,) ndarray
            Gas-phase vapor concentration      [mol m⁻³].
        kappa_value_np : (S,) ndarray
            κ-Köhler hygroscopicity parameter   [–].
        surface_tension_np : (S,) ndarray
            Surface tension of each species    [N m⁻¹].
        gas_mass_np : (S,) ndarray
            Available gas-phase mass           [kg].
        particle_concentration_np : (P,) ndarray
            Number concentration of each particle bin [# m⁻³].
        """
        # --- basic shape checks -------------------------------------------------
        p_count, s_count = species_masses_np.shape
        assert p_count == particle_count and s_count == species_count, (
            "Input array shapes must match the global `particle_count` / "
            "`species_count` that were used to allocate Taichi fields."
        )

        # input data conversion to Taichi fields --------------------------------
        
        # create fiels that will be internally used
        temperature = ti.static(input_temperature)
        pressure = ti.static(input_pressure)
        mass_accommodation = ti.static(input_mass_accommodation)
        dynamic_viscosity = ti.static(input_dynamic_viscosity)
        diffusion_coefficient = ti.static(input_diffusion_coefficient)
        time_step = ti.static(input_time_step)
        simulation_volume = ti.static(input_simulation_volume)


        # apply input species masses to the field
        species_masses = ti.field(
            float, shape=(particle_count, species_count), name="species_masses"
        )
        species_masses.from_numpy(input_species_masses)

        # create density field
        density = ti.field(float, shape=(species_count,), name="density")
        density.from_numpy(input_density)
        # create molar mass field
        molar_mass = ti.field(float, shape=(species_count,), name="molar_mass")
        molar_mass.from_numpy(input_molar_mass)
        # create pure vapor pressure field
        pure_vapor_pressure = ti.field(
            float, shape=(species_count,), name="pure_vapor_pressure"
        )
        pure_vapor_pressure.from_numpy(input_pure_vapor_pressure)
        # create vapor concentration field
        vapor_concentration = ti.field(
            float, shape=(species_count,), name="vapor_concentration"
        )
        vapor_concentration.from_numpy(input_vapor_concentration)
        # create kappa value field
        kappa_value = ti.field(float, shape=(species_count,), name="kappa_value")
        kappa_value.from_numpy(input_kappa_value)
        # create surface tension field
        surface_tension = ti.field(
            float, shape=(species_count,), name="surface_tension"
        )
        surface_tension.from_numpy(input_surface_tension)

        # create gas mass field
        gas_mass = ti.field(float, shape=(species_count,), name="gas_mass")
        gas_mass.from_numpy(input_gas_mass)

        # create particle concentration field
        particle_concentration = ti.field(
            float, shape=(particle_count,), name="particle_concentration"
        )
        particle_concentration.from_numpy(input_particle_concentration)


        # temperay fields
        radius = ti.field(float, shape=(particle_count,), name="radii")
        mass_transport_rate = ti.field(
            float, shape=(particle_count, species_count), name="mass_transport_rate"
        )
        first_order_coefficient = ti.field(
            float,
            shape=(particle_count, species_count),
            name="first_order_coefficient",
        )
        partial_pressure = ti.field(
            float, shape=(particle_count, species_count), name="partial_pressure"
        )
        pressure_delta = ti.field(
            float, shape=(particle_count, species_count), name="pressure_delta"
        )
        kelvin_term = ti.field(
            float, shape=(particle_count, species_count), name="kelvin_term"
        )
        kelvin_radius = ti.field(
            float, shape=(particle_count, species_count), name="kelvin_radius"
        )
        transferable_mass = ti.field(
            float, shape=(particle_count, species_count), name="transferable_mass"
        )
        total_requested_mass = ti.field(
            float, shape=(species_count,), name="total_requested_mass"
        )
        scaling_factor = ti.field(float, shape=(species_count,), name="scaling_factor")



    # --------------------------------------------------------------------- #
    # Convenience wrappers that delegate to the module-level kernels
    # --------------------------------------------------------------------- #

    # quick getters to pull data back to NumPy for analysis ----------------
    def get_radius(self) -> np.ndarray:
        return radius.to_numpy()

    def get_species_masses(self) -> np.ndarray:
        return species_masses.to_numpy()

    def get_gas_mass(self) -> np.ndarray:
        return gas_mass.to_numpy()

    def get_transferable_mass(self) -> np.ndarray:
        return transferable_mass.to_numpy()

    @ti.kernel
    def fused_step():
        # --- species-independent prep (runs once per particle) -------------
        for p in ti.ndrange(particle_count):
            # a) radius ------------------------------------------------------
            volume = 0.0
            for s in ti.ndrange(species_count):
                volume += species_masses[p, s] / density[s]
            r_p = ti.pow(3.0 * volume / (4.0 * ti.math.pi), 1.0 / 3.0)
            radius[p] = r_p

            # b) effective bulk properties (weighted averages) ---------------
            w_mass = 0.0
            sig_sum = 0.0
            rho_sum = 0.0
            for s in ti.ndrange(species_count):
                w = species_masses[p, s]
                w_mass += w
                sig_sum += surface_tension[s] * w
                rho_sum += density[s] * w
            sig_eff = sig_sum / w_mass
            rho_eff = rho_sum / w_mass

            # ---------- per-species loop (still inside the same kernel) -----
            for s in ti.ndrange(species_count):
                # first-order coefficient
                k1 = condensation.fget_first_order_mass_transport_via_system_state(
                    particle_radius=r_p,
                    molar_mass=molar_mass[s],
                    mass_accommodation=mass_accommodation,
                    temperature=temperature,
                    pressure=pressure,
                    dynamic_viscosity=dynamic_viscosity,
                    diffusion_coefficient=diffusion_coefficient,
                )
                first_order_coefficient[p, s] = k1

                # Kelvin term -------------------------------------------------
                r_k = particle_properties.fget_kelvin_radius(
                    sig_eff, rho_eff, molar_mass[s], temperature
                )
                kel = particle_properties.fget_kelvin_term(r_p, r_k)
                kelvin_term[p, s] = kel

                # vapour & ΔP -----------------------------------------------
                p_g = gas_properties.fget_partial_pressure(
                    vapor_concentration[s],
                    molar_mass[s],
                    temperature,
                )
                dP = particle_properties.fget_partial_pressure_delta(p_g, p_g, kel)
                pressure_delta[p, s] = dP

                # mass-flux ---------------------------------------------------
                mass_transport_rate[p, s] = (
                    k1
                    * dP
                    * molar_mass[s]
                    / (par.util.constants.GAS_CONSTANT * temperature)
                )
        update_scaling_factors(time_step)
        for i in ti.ndrange(particle_count):
            update_transferable_mass(i, time_step)
        update_gas_mass()
        update_species_masses()


    @ti.func
    def update_scaling_factors(time_step: float):
        """
        Internal routine that fills `total_requested_mass` and `scaling_factor`.

        This is the exact logic that used to live inside
        `calculate_scaling_factors`.
        """
        # 1. reset accumulator
        for j in ti.ndrange(species_count):
            total_requested_mass[j] = 0.0

        # 2. total requested mass per species
        for i, j in ti.ndrange(particle_count, species_count):
            particle_concentration[i] = (
                1.0 / simulation_volume
            )  # particle-resolved
            total_requested_mass[j] += (
                mass_transport_rate[i, j] * time_step * particle_concentration[i]
            )

        # 3. build scaling factors
        for j in ti.ndrange(species_count):
            scaling_factor[j] = 1.0
            if total_requested_mass[j] > gas_mass[j]:
                scaling_factor[j] = gas_mass[j] / total_requested_mass[j]

    @ti.func
    def update_gas_mass():
        """
        Update the gas mass field based on the input gas mass.

        """
        for j in ti.ndrange(species_count):
            species_mass = 0.0
            for i in ti.ndrange(particle_count):
                species_mass += transferable_mass[i, j]
            # Update the gas mass for species j
            gas_mass[j] -= species_mass
            # Ensure gas mass does not go
            gas_mass[j] = ti.max(gas_mass[j], 0.0)


    @ti.func
    def update_species_masses():
        """
        Update the species masses field based on the input species masses.

        """
        for i in ti.ndrange(particle_count):
            for j in ti.ndrange(species_count):
                # Update the species mass for particle i and species j
                if particle_concentration[i] > 0.0:
                    species_masses[i, j] = (
                        species_masses[i, j] * particle_concentration[i]
                        + transferable_mass[i, j]
                    ) / particle_concentration[i]
                else:
                    species_masses[i, j] = 0.0
                # Ensure species mass does not go negative
                species_masses[i, j] = ti.max(species_masses[i, j], 0.0)
