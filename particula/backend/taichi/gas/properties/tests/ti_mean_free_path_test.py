import numpy as np
import taichi as ti

ti.init(arch=ti.cpu)

from particula.backend.taichi.gas.properties.ti_mean_free_path_module import (
    get_molecule_mean_free_path_taichi as ti_get_molecule_mean_free_path,
    kget_molecule_mean_free_path,
)
from particula.gas.properties.mean_free_path import get_molecule_mean_free_path
from particula.gas.properties.dynamic_viscosity import get_dynamic_viscosity
from particula.util.constants import MOLECULAR_WEIGHT_AIR


def test_wrapper_matches_reference():
    temperature_array = np.array([300.0], dtype=np.float64)
    molar_mass_array  = np.full_like(temperature_array, MOLECULAR_WEIGHT_AIR, dtype=np.float64)
    pressure_array    = np.full_like(temperature_array, 101325.0, dtype=np.float64)

    np.testing.assert_allclose(
        ti_get_molecule_mean_free_path(molar_mass_array, temperature_array, pressure_array),
        get_molecule_mean_free_path(molar_mass_array, temperature_array, pressure_array),
        rtol=1e-8,
    )


def test_kernel_direct():
    temperature_array  = np.array([280.0], dtype=np.float64)
    n_samples          = temperature_array.size
    molar_mass_array   = np.full(n_samples, MOLECULAR_WEIGHT_AIR, dtype=np.float64)
    pressure_array     = np.full(n_samples, 101325.0, dtype=np.float64)
    dynamic_viscosity_array = get_dynamic_viscosity(temperature_array)

    molar_mass_field         = ti.ndarray(dtype=ti.f64, shape=n_samples);  molar_mass_field.from_numpy(molar_mass_array)
    temperature_field        = ti.ndarray(dtype=ti.f64, shape=n_samples);  temperature_field.from_numpy(temperature_array)
    pressure_field           = ti.ndarray(dtype=ti.f64, shape=n_samples);  pressure_field.from_numpy(pressure_array)
    dynamic_viscosity_field  = ti.ndarray(dtype=ti.f64, shape=n_samples);  dynamic_viscosity_field.from_numpy(dynamic_viscosity_array)
    mean_free_path_field     = ti.ndarray(dtype=ti.f64, shape=n_samples)

    kget_molecule_mean_free_path(
        molar_mass_field,
        temperature_field,
        pressure_field,
        dynamic_viscosity_field,
        mean_free_path_field
    )

    np.testing.assert_allclose(
        mean_free_path_field.to_numpy(),
        get_molecule_mean_free_path(molar_mass_array, temperature_array, pressure_array, dynamic_viscosity_array),
        rtol=1e-8,
    )
