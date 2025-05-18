import taichi as ti
import numpy as np

from particula.backend.dispatch_register import register
from particula.util.constants import GAS_CONSTANT
from particula.gas import get_dynamic_viscosity

_GAS_CONSTANT = float(GAS_CONSTANT)  # J mol⁻¹ K⁻¹


@ti.func
def fget_molecule_mean_free_path(
    molar_mass: ti.f64,
    temperature: ti.f64,
    pressure: ti.f64,
    dynamic_viscosity: ti.f64,
) -> ti.f64:
    return (2.0 * dynamic_viscosity / pressure) / ti.sqrt(
        8.0 * molar_mass / (ti.math.pi * _GAS_CONSTANT * temperature)
    )

@ti.kernel
def kget_molecule_mean_free_path(          # noqa: N802
    molar_mass: ti.types.ndarray(dtype=ti.f64, ndim=1),
    temperature: ti.types.ndarray(dtype=ti.f64, ndim=1),
    pressure: ti.types.ndarray(dtype=ti.f64, ndim=1),
    dynamic_viscosity: ti.types.ndarray(dtype=ti.f64, ndim=1),
    mean_free_path: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(mean_free_path.shape[0]):
        mean_free_path[i] = fget_molecule_mean_free_path(
            molar_mass[i],
            temperature[i],
            pressure[i],
            dynamic_viscosity[i],
        )

@register("get_molecule_mean_free_path", backend="taichi")
def get_molecule_mean_free_path_taichi(
    molar_mass,
    temperature,
    pressure,
    dynamic_viscosity=None,
):
    if not all(
        isinstance(arg, (float, np.ndarray))
        for arg in (molar_mass, temperature, pressure)
    ):
        raise TypeError("Taichi backend expects float or NumPy array inputs.")

    # default μ if None
    if dynamic_viscosity is None:
        dynamic_viscosity = get_dynamic_viscosity(temperature)

    molar_mass_array, temperature_array, pressure_array, dynamic_viscosity_array = map(np.atleast_1d, (molar_mass, temperature, pressure, dynamic_viscosity))
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
