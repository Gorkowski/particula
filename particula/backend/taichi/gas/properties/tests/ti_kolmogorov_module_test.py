import taichi as ti
ti.init(arch=ti.cpu)

import numpy as np
import pytest
from particula.gas.properties.kolmogorov_module import (
    get_kolmogorov_time,
    get_kolmogorov_length,
    get_kolmogorov_velocity,
)
from particula.backend.taichi.gas.properties.ti_kolmogorov_module import (
    kget_kolmogorov_time,
    kget_kolmogorov_length,
    kget_kolmogorov_velocity,
    ti_get_kolmogorov_time,
    ti_get_kolmogorov_length,
    ti_get_kolmogorov_velocity,
)

def _sample_data():
    """Return sample kinematic-viscosity and turbulent-dissipation arrays."""
    kinematic_viscosity_array = np.array([1.5e-5, 2.0e-5])
    turbulent_dissipation_array = np.array([0.1, 0.2])
    return kinematic_viscosity_array, turbulent_dissipation_array

@pytest.mark.parametrize(
    ("ti_func", "py_func"),
    [
        (ti_get_kolmogorov_time,     get_kolmogorov_time),
        (ti_get_kolmogorov_length,   get_kolmogorov_length),
        (ti_get_kolmogorov_velocity, get_kolmogorov_velocity),
    ],
)
def test_ti_wrappers_parity(ti_func, py_func):
    kin_visc, turb_diss = _sample_data()
    np.testing.assert_allclose(
        ti_func(kin_visc, turb_diss),
        py_func(kin_visc, turb_diss),
        rtol=1e-12,
        atol=0,
    )


@pytest.mark.parametrize(
    ("kernel_func", "py_func"),
    [
        (kget_kolmogorov_time,     get_kolmogorov_time),
        (kget_kolmogorov_length,   get_kolmogorov_length),
        (kget_kolmogorov_velocity, get_kolmogorov_velocity),
    ],
)
def test_ti_kernels_parity(kernel_func, py_func):
    kin_visc_arr, turb_diss_arr = _sample_data()
    n = kin_visc_arr.size
    kin_field   = ti.ndarray(dtype=ti.f64, shape=n)
    diss_field  = ti.ndarray(dtype=ti.f64, shape=n)
    result_field = ti.ndarray(dtype=ti.f64, shape=n)
    kin_field.from_numpy(kin_visc_arr)
    diss_field.from_numpy(turb_diss_arr)

    kernel_func(kin_field, diss_field, result_field)

    np.testing.assert_allclose(
        result_field.to_numpy(),
        py_func(kin_visc_arr, turb_diss_arr),
        rtol=1e-12,
        atol=0,
    )
