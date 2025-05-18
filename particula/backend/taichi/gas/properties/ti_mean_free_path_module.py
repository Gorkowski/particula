import taichi as ti
import numpy as np
from typing import Optional, Union

from particula.backend.dispatch_register import register
from particula.util.constants import GAS_CONSTANT
from particula.gas.properties.dynamic_viscosity import get_dynamic_viscosity

"""
Taichi helpers for computing the molecular mean free path (λ).

Implements
    λ = (2 μ / p) / √(8 M / (π R T))

where μ is the dynamic viscosity, p the pressure, M the molar mass,
R the universal gas constant, and T the temperature.

All public objects follow the Particula naming guide and include
detailed docstrings in the required format.
"""

_GAS_CONSTANT = float(GAS_CONSTANT)  # J mol⁻¹ K⁻¹


@ti.func
def fget_molecule_mean_free_path(
    molar_mass: ti.f64,
    temperature: ti.f64,
    pressure: ti.f64,
    dynamic_viscosity: ti.f64,
) -> ti.f64:
    """
    Return the mean free path for a single gas state.

    Equation:
        λ = (2 μ / p) / √(8 M / (π R T))

    Arguments:
        - molar_mass : Molecular mass, M [kg mol⁻¹].
        - temperature : Temperature, T [K].
        - pressure : Pressure, p [Pa].
        - dynamic_viscosity : Dynamic viscosity, μ [Pa s].

    Returns:
        - λ : Mean free path [m].

    References:
        - “Mean free path”, Wikipedia.
    """
    return (2.0 * dynamic_viscosity / pressure) / ti.sqrt(
        8.0 * molar_mass / (ti.math.pi * _GAS_CONSTANT * temperature)
    )

@ti.kernel
def kget_molecule_mean_free_path(
    molar_mass: ti.types.ndarray(dtype=ti.f64, ndim=1),
    temperature: ti.types.ndarray(dtype=ti.f64, ndim=1),
    pressure: ti.types.ndarray(dtype=ti.f64, ndim=1),
    dynamic_viscosity: ti.types.ndarray(dtype=ti.f64, ndim=1),
    mean_free_path: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """
    Kernel version of `fget_molecule_mean_free_path`.

    Populates `mean_free_path` in-place for 1-D arrays.

    Arguments:
        - molar_mass : 1-D array of M [kg mol⁻¹].
        - temperature : 1-D array of T [K].
        - pressure : 1-D array of p [Pa].
        - dynamic_viscosity : 1-D array of μ [Pa s].
        - mean_free_path : Output array for λ [m].
    """
    for i in range(mean_free_path.shape[0]):
        mean_free_path[i] = fget_molecule_mean_free_path(
            molar_mass[i],
            temperature[i],
            pressure[i],
            dynamic_viscosity[i],
        )

@register("get_molecule_mean_free_path", backend="taichi")
def get_molecule_mean_free_path_taichi(
    molar_mass: Union[float, np.ndarray],
    temperature: Union[float, np.ndarray],
    pressure: Union[float, np.ndarray],
    dynamic_viscosity: Optional[Union[float, np.ndarray]] = None,
) -> Union[float, np.ndarray]:
    """
    Vectorised Taichi implementation of the molecular mean free path.

    If `dynamic_viscosity` is None, it is computed automatically.

    Arguments:
        - molar_mass : float | ndarray, M [kg mol⁻¹].
        - temperature : float | ndarray, T [K].
        - pressure : float | ndarray, p [Pa].
        - dynamic_viscosity : float | ndarray, μ [Pa s], optional.

    Returns:
        - float | ndarray : λ [m].

    Examples:
        ```py title="Taichi mean free path"
        from particula.backend.dispatch_register import use_backend
        from particula.backend.taichi.gas.properties.ti_mean_free_path_module \
            import get_molecule_mean_free_path_taichi

        use_backend("taichi")

        λ = get_molecule_mean_free_path_taichi(
            molar_mass=0.02897,
            temperature=298.15,
            pressure=101_325.0,
        )
        ```

    References:
        - “Mean free path”, Wikipedia.
    """
    if not all(
        isinstance(arg, (float, np.ndarray))
        for arg in (molar_mass, temperature, pressure)
    ):
        raise TypeError("Taichi backend expects float or NumPy array inputs.")

    # default μ if None
    if dynamic_viscosity is None:
        dynamic_viscosity = get_dynamic_viscosity(temperature)

    (
        molar_mass_array,
        temperature_array,
        pressure_array,
        dynamic_viscosity_array,
    ) = map(
        np.atleast_1d,
        (molar_mass, temperature, pressure, dynamic_viscosity),
    )
    n_elements = molar_mass_array.size

    molar_mass_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    molar_mass_ti.from_numpy(molar_mass_array)

    temperature_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    temperature_ti.from_numpy(temperature_array)

    pressure_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    pressure_ti.from_numpy(pressure_array)

    dynamic_viscosity_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    dynamic_viscosity_ti.from_numpy(dynamic_viscosity_array)

    mean_free_path_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)

    kget_molecule_mean_free_path(
        molar_mass_ti,
        temperature_ti,
        pressure_ti,
        dynamic_viscosity_ti,
        mean_free_path_ti,
    )

    mean_free_path_array = mean_free_path_ti.to_numpy()
    return (
        mean_free_path_array.item()
        if mean_free_path_array.size == 1
        else mean_free_path_array
    )

__all__ = [
    "fget_molecule_mean_free_path",
    "kget_molecule_mean_free_path",
    "get_molecule_mean_free_path_taichi",
]
