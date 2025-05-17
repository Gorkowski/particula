"""Taichi-accelerated gas-property functions (public re-exports)."""

from .ti_dynamic_viscosity_module import (
    fget_dynamic_viscosity,
    ti_get_dynamic_viscosity,
    kget_dynamic_viscosity,          # NEW
)
from .ti_concentration_from_pressure_module import (               # noqa: F401
    ti_get_concentration_from_pressure,
    fget_concentration_from_pressure,
    kget_concentration_from_pressure,
)
from .ti_kolmogorov_module import (
    ti_get_kolmogorov_time,
    ti_get_kolmogorov_length,
    ti_get_kolmogorov_velocity,
    fget_kolmogorov_length,
    fget_kolmogorov_time,
    fget_kolmogorov_velocity,
    kget_kolmogorov_length,
    kget_kolmogorov_time,
    kget_kolmogorov_velocity,
)
from .ti_pressure_function_module import (          # noqa: F401
    ti_get_partial_pressure,
    ti_get_saturation_ratio_from_pressure,
    fget_partial_pressure,
    fget_saturation_ratio_from_pressure,
    kget_partial_pressure,
    kget_saturation_ratio_from_pressure,
)
from .ti_taylor_microscale_module import (
    ti_get_lagrangian_taylor_microscale_time,
    ti_get_taylor_microscale,
    ti_get_taylor_microscale_reynolds_number,
    fget_lagrangian_taylor_microscale_time,
    fget_taylor_microscale,
    fget_taylor_microscale_reynolds_number,
    kget_lagrangian_taylor_microscale_time,
    kget_taylor_microscale,
    kget_taylor_microscale_reynolds_number,
)
from .ti_thermal_conductivity_module import (
    ti_get_thermal_conductivity,  # noqa: F401
    fget_thermal_conductivity,
    kget_thermal_conductivity,
)
from .ti_vapor_pressure_module import (
    ti_get_antoine_vapor_pressure,
    ti_get_clausius_clapeyron_vapor_pressure,
    ti_get_buck_vapor_pressure,
    fget_antoine_vapor_pressure,
    fget_clausius_clapeyron_vapor_pressure,
    fget_buck_vapor_pressure,
    kget_antoine_vapor_pressure,
    kget_clausius_clapeyron_vapor_pressure,
    kget_buck_vapor_pressure,
)
from .ti_mean_free_path_module import (
    fget_molecule_mean_free_path,
    kget_molecule_mean_free_path,
    get_molecule_mean_free_path_taichi,
)