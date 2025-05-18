import taichi as ti
ti.init(arch=ti.cpu)

import numpy as np
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

def test_ti_wrappers_parity():
    kinematic_viscosity_array, turbulent_dissipation_array = _sample_data()
    # Vector input
    np.testing.assert_allclose(
        ti_get_kolmogorov_time(
            kinematic_viscosity_array, turbulent_dissipation_array
        ),
        get_kolmogorov_time(
            kinematic_viscosity_array, turbulent_dissipation_array
        ),
        rtol=1e-12, atol=0
    )
    np.testing.assert_allclose(
        ti_get_kolmogorov_length(
            kinematic_viscosity_array, turbulent_dissipation_array
        ),
        get_kolmogorov_length(
            kinematic_viscosity_array, turbulent_dissipation_array
        ),
        rtol=1e-12, atol=0
    )
    np.testing.assert_allclose(
        ti_get_kolmogorov_velocity(
            kinematic_viscosity_array, turbulent_dissipation_array
        ),
        get_kolmogorov_velocity(
            kinematic_viscosity_array, turbulent_dissipation_array
        ),
        rtol=1e-12, atol=0
    )


def test_ti_kernels_parity():
    kinematic_viscosity_array, turbulent_dissipation_array = _sample_data()
    n_values = kinematic_viscosity_array.size
    kinematic_viscosity_field = ti.ndarray(dtype=ti.f64, shape=n_values)
    turbulent_dissipation_field = ti.ndarray(dtype=ti.f64, shape=n_values)
    kolmogorov_time_field = ti.ndarray(dtype=ti.f64, shape=n_values)
    kolmogorov_length_field = ti.ndarray(dtype=ti.f64, shape=n_values)
    kolmogorov_velocity_field = ti.ndarray(dtype=ti.f64, shape=n_values)
    kinematic_viscosity_field.from_numpy(kinematic_viscosity_array)
    turbulent_dissipation_field.from_numpy(turbulent_dissipation_array)
    kget_kolmogorov_time(
        kinematic_viscosity_field,
        turbulent_dissipation_field,
        kolmogorov_time_field
    )
    kget_kolmogorov_length(
        kinematic_viscosity_field,
        turbulent_dissipation_field,
        kolmogorov_length_field
    )
    kget_kolmogorov_velocity(
        kinematic_viscosity_field,
        turbulent_dissipation_field,
        kolmogorov_velocity_field
    )
    np.testing.assert_allclose(
        kolmogorov_time_field.to_numpy(),
        get_kolmogorov_time(
            kinematic_viscosity_array, turbulent_dissipation_array
        ),
        rtol=1e-12, atol=0
    )
    np.testing.assert_allclose(
        kolmogorov_length_field.to_numpy(),
        get_kolmogorov_length(
            kinematic_viscosity_array, turbulent_dissipation_array
        ),
        rtol=1e-12, atol=0
    )
    np.testing.assert_allclose(
        kolmogorov_velocity_field.to_numpy(),
        get_kolmogorov_velocity(
            kinematic_viscosity_array, turbulent_dissipation_array
        ),
        rtol=1e-12, atol=0
    )
