import warnings

try:
    from .ti_aerodynamic_mobility_module import (
        ti_get_aerodynamic_mobility,
        kget_aerodynamic_mobility,
        fget_aerodynamic_mobility,
    )
except ModuleNotFoundError:
    warnings.warn(
        "Skipped optional Taichi property module 'ti_aerodynamic_mobility_module'.",
        RuntimeWarning,
        stacklevel=2,
    )

try:
    from .ti_aerodynamic_length_module import (
        ti_get_aerodynamic_length,
        kget_aerodynamic_length,
        fget_aerodynamic_length,
    )
except ModuleNotFoundError:
    warnings.warn(
        "Skipped optional Taichi property module 'ti_aerodynamic_length_module'.",
        RuntimeWarning,
        stacklevel=2,
    )

try:
    from .ti_convert_mass_concentration_module import (
        ti_get_mole_fraction_from_mass,
        kget_mole_fraction_from_mass,
        fget_mole_single,
        ti_get_volume_fraction_from_mass,
        kget_volume_fraction_from_mass,
        fget_volume_single,
    )
except ModuleNotFoundError:
    warnings.warn(
        "Skipped optional Taichi property module 'ti_convert_mass_concentration_module'.",
        RuntimeWarning,
        stacklevel=2,
    )

try:
    from .ti_convert_mole_fraction_module import (
        ti_get_mass_fractions_from_moles,
        kget_mass_fractions_1d,
        kget_mass_fractions_2d,
        fget_mass_fraction,
    )
except ModuleNotFoundError:
    warnings.warn(
        "Skipped optional Taichi property module 'ti_convert_mole_fraction_module'.",
        RuntimeWarning,
        stacklevel=2,
    )

try:
    from .ti_convert_size_distribution_module import (
        ti_get_distribution_in_dn,
        kget_distribution_in_dn,
        fget_distribution_in_dn,
        ti_get_pdf_distribution_in_pmf,
        kget_pdf_distribution_in_pmf,
        fget_pdf_distribution_in_pmf,
    )
except ModuleNotFoundError:
    warnings.warn(
        "Skipped optional Taichi property module 'ti_convert_size_distribution_module'.",
        RuntimeWarning,
        stacklevel=2,
    )

try:
    from .ti_convert_kappa_volumes import (
        ti_get_solute_volume_from_kappa,
        kget_solute_volume_from_kappa,
        fget_solute_volume_from_kappa,
        ti_get_water_volume_from_kappa,
        kget_water_volume_from_kappa,
        fget_water_volume_from_kappa,
        ti_get_kappa_from_volumes,
        kget_kappa_from_volumes,
        fget_kappa_from_volumes,
        ti_get_water_volume_in_mixture,
        kget_water_volume_in_mixture,
        fget_water_volume_in_mixture,
    )
except ModuleNotFoundError:
    warnings.warn(
        "Skipped optional Taichi property module 'ti_convert_kappa_volumes'.",
        RuntimeWarning,
        stacklevel=2,
    )

try:
    from .ti_vapor_correction_module import (
        ti_get_vapor_transition_correction,
        kget_vapor_transition_correction,
        fget_vapor_transition_correction,
    )
except ModuleNotFoundError:
    warnings.warn(
        "Skipped optional Taichi property module 'ti_vapor_correction_module'.",
        RuntimeWarning,
        stacklevel=2,
    )

try:
    from .ti_stokes_number_module import (
        ti_get_stokes_number,
        kget_stokes_number,
        fget_stokes_number,
    )
except ModuleNotFoundError:
    warnings.warn(
        "Skipped optional Taichi property module 'ti_stokes_number_module'.",
        RuntimeWarning,
        stacklevel=2,
    )

try:
    from .ti_knudsen_number_module import (
        ti_get_knudsen_number,
        kget_knudsen_number,
        fget_knudsen_number,
    )
except ModuleNotFoundError:
    warnings.warn(
        "Skipped optional Taichi property module 'ti_knudsen_number_module'.",
        RuntimeWarning,
        stacklevel=2,
    )

try:
    from .ti_lognormal_size_distribution_module import (
        ti_get_lognormal_pdf_distribution,
        ti_get_lognormal_pmf_distribution,
    )
except ModuleNotFoundError:
    warnings.warn(
        "Skipped optional Taichi property module 'ti_lognormal_size_distribution_module'.",
        RuntimeWarning,
        stacklevel=2,
    )

try:
    from .ti_kelvin_effect_module import (
        ti_get_kelvin_radius,
        kget_kelvin_radius,
        fget_kelvin_radius,
        ti_get_kelvin_term,
        kget_kelvin_term,
        fget_kelvin_term,
    )
except ModuleNotFoundError:
    warnings.warn(
        "Skipped optional Taichi property module 'ti_kelvin_effect_module'.",
        RuntimeWarning,
        stacklevel=2,
    )

try:
    from .ti_special_functions import (
        ti_get_debye_function,
        kget_debye_function,
        fget_debye_function,
    )
except ModuleNotFoundError:
    warnings.warn(
        "Skipped optional Taichi property module 'ti_special_functions'.",
        RuntimeWarning,
        stacklevel=2,
    )

try:
    from .ti_slip_correction_module import (
        ti_get_cunningham_slip_correction,
        kget_cunningham_slip_correction,
        fget_cunningham_slip_correction,
    )
except ModuleNotFoundError:
    warnings.warn(
        "Skipped optional Taichi property module 'ti_slip_correction_module'.",
        RuntimeWarning,
        stacklevel=2,
    )

try:
    from .ti_friction_factor_module import (
        ti_get_friction_factor,
        kget_friction_factor,
        fget_friction_factor,
    )
except ModuleNotFoundError:
    warnings.warn(
        "Skipped optional Taichi property module 'ti_friction_factor_module'.",
        RuntimeWarning,
        stacklevel=2,
    )

try:
    from .ti_settling_velocity import (
        ti_get_particle_settling_velocity,
        kget_particle_settling_velocity,
        fget_particle_settling_velocity,
    )
except ModuleNotFoundError:
    warnings.warn(
        "Skipped optional Taichi property module 'ti_settling_velocity'.",
        RuntimeWarning,
        stacklevel=2,
    )

try:
    from .ti_reynolds_number_module import (
        ti_get_particle_reynolds_number,
        kget_particle_reynolds_number,
        fget_particle_reynolds_number,
    )
except ModuleNotFoundError:
    warnings.warn(
        "Skipped optional Taichi property module 'ti_reynolds_number_module'.",
        RuntimeWarning,
        stacklevel=2,
    )

try:
    from .ti_partial_pressure_module import (
        ti_get_partial_pressure_delta,
        kget_partial_pressure_delta,
        fget_partial_pressure_delta,
    )
except ModuleNotFoundError:
    warnings.warn(
        "Skipped optional Taichi property module 'ti_partial_pressure_module'.",
        RuntimeWarning,
        stacklevel=2,
    )

try:
    from .ti_mean_thermal_speed_module import (
        ti_get_mean_thermal_speed,
        kget_mean_thermal_speed,
        fget_mean_thermal_speed,
    )
except ModuleNotFoundError:
    warnings.warn(
        "Skipped optional Taichi property module 'ti_mean_thermal_speed_module'.",
        RuntimeWarning,
        stacklevel=2,
    )

try:
    from .ti_mixing_state_index_module import (
        ti_get_mixing_state_index,
        kget_mixing_state_index,
        fget_mixing_state_index,
    )
except ModuleNotFoundError:
    warnings.warn(
        "Skipped optional Taichi property module 'ti_mixing_state_index_module'.",
        RuntimeWarning,
        stacklevel=2,
    )

try:
    from .ti_diffusive_knudsen_module import (
        ti_get_diffusive_knudsen_number,
        kget_diffusive_knudsen_number,
        fget_diffusive_knudsen_number,
    )
except ModuleNotFoundError:
    warnings.warn(
        "Skipped optional Taichi property module 'ti_diffusive_knudsen_module'.",
        RuntimeWarning,
        stacklevel=2,
    )

try:
    from .ti_activity_module import (
        ti_get_ideal_activity_molar,
        kget_ideal_activity_molar,
        ti_get_ideal_activity_volume,
        kget_ideal_activity_volume,
        ti_get_ideal_activity_mass,
        kget_ideal_activity_mass,
        ti_get_kappa_activity,
        kget_kappa_activity,
        ti_get_surface_partial_pressure,
        kget_surface_partial_pressure,
        fget_surface_partial_pressure,
    )
except ModuleNotFoundError:
    warnings.warn(
        "Skipped optional Taichi property module 'ti_activity_module'.",
        RuntimeWarning,
        stacklevel=2,
    )

try:
    from .ti_collision_radius_module import (
        ti_get_collision_radius_mg1988,
        ti_get_collision_radius_sr1992,
        ti_get_collision_radius_mzg2002,
        ti_get_collision_radius_tt2012,
        ti_get_collision_radius_wq2022_rg,
        ti_get_collision_radius_wq2022_rg_df,
        ti_get_collision_radius_wq2022_rg_df_k0,
        ti_get_collision_radius_wq2022_rg_df_k0_a13,
    )
except ModuleNotFoundError:
    warnings.warn(
        "Skipped optional Taichi property module 'ti_collision_radius_module'.",
        RuntimeWarning,
        stacklevel=2,
    )

# ---------------------------------------------------------------------- #
#  Public export list – all symbols starting with ti_/kget_/fget_        #
# ---------------------------------------------------------------------- #
__all__: list[str] = [
    name for name in globals()
    if name.startswith(("ti_get_", "kget_", "fget_"))
]
