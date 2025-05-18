import numpy as np
import taichi as ti
from numpy.testing import assert_allclose

from particula.gas.properties.concentration_function import (
    get_concentration_from_pressure,
)
from particula.backend.taichi.gas.properties.ti_concentration_from_pressure_module import (
    ti_get_concentration_from_pressure,
    kget_concentration_from_pressure,
)

ti.init(arch=ti.cpu)


def test_wrapper_matches_numpy():
    partial_pressure_array = np.array([101325.0, 202650.0], dtype=np.float64)
    molar_mass_array = np.array([0.02897, 0.02897], dtype=np.float64)
    temperature_array = np.array([298.15, 300.0], dtype=np.float64)

    expected_concentration = get_concentration_from_pressure(
        partial_pressure_array,
        molar_mass_array,
        temperature_array,
    )
    result_concentration = ti_get_concentration_from_pressure(
        partial_pressure_array,
        molar_mass_array,
        temperature_array,
    )

    assert_allclose(
        result_concentration,
        expected_concentration,
        rtol=1e-8,
        atol=0,
    )


def test_kernel_direct_call():
    partial_pressure_array = np.array([101325.0, 202650.0], dtype=np.float64)
    molar_mass_array = np.array([0.02897, 0.02897], dtype=np.float64)
    temperature_array = np.array([298.15, 300.0], dtype=np.float64)

    n_points = partial_pressure_array.size
    partial_pressure_ti = ti.ndarray(dtype=ti.f64, shape=n_points)
    molar_mass_ti = ti.ndarray(dtype=ti.f64, shape=n_points)
    temperature_ti = ti.ndarray(dtype=ti.f64, shape=n_points)
    concentration_ti = ti.ndarray(dtype=ti.f64, shape=n_points)
    partial_pressure_ti.from_numpy(partial_pressure_array)
    molar_mass_ti.from_numpy(molar_mass_array)
    temperature_ti.from_numpy(temperature_array)

    kget_concentration_from_pressure(
        partial_pressure_ti,
        molar_mass_ti,
        temperature_ti,
        concentration_ti,
    )

    assert_allclose(
        concentration_ti.to_numpy(),
        get_concentration_from_pressure(
            partial_pressure_array,
            molar_mass_array,
            temperature_array,
        ),
        rtol=1e-8,
        atol=0,
    )
