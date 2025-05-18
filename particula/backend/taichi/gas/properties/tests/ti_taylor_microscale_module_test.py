import numpy as np
import taichi as ti
import particula as par
from particula.backend.taichi.gas.properties.ti_taylor_microscale_module import (
    kget_lagrangian_taylor_microscale_time,
    kget_taylor_microscale,
    kget_taylor_microscale_reynolds_number,
    ti_get_lagrangian_taylor_microscale_time,
    ti_get_taylor_microscale,
    ti_get_taylor_microscale_reynolds_number,
)

ti.init(arch=ti.cpu)

def test_wrappers_vs_numpy():
    # vector inputs
    kolmogorov_time_array = np.array([0.38, 0.40])
    taylor_microscale_reynolds_number_array = np.array([400., 600.])
    acceleration_variance_array = np.array([0.05, 0.06])
    fluid_rms_velocity_array = np.array([0.35, 0.40])
    kinematic_viscosity_array = np.array([1.5e-5, 1.8e-5])
    turbulent_dissipation_array = np.array([0.10, 0.12])
    taylor_microscale_array = par.gas.get_taylor_microscale(
        fluid_rms_velocity_array, kinematic_viscosity_array, turbulent_dissipation_array
    )

    np.testing.assert_allclose(
        ti_get_lagrangian_taylor_microscale_time(
            kolmogorov_time_array,
            taylor_microscale_reynolds_number_array,
            acceleration_variance_array,
        ),
        par.gas.get_lagrangian_taylor_microscale_time(
            kolmogorov_time_array,
            taylor_microscale_reynolds_number_array,
            acceleration_variance_array,
        ),
    )
    np.testing.assert_allclose(
        ti_get_taylor_microscale(
            fluid_rms_velocity_array,
            kinematic_viscosity_array,
            turbulent_dissipation_array,
        ),
        taylor_microscale_array,
    )
    np.testing.assert_allclose(
        ti_get_taylor_microscale_reynolds_number(
            fluid_rms_velocity_array,
            taylor_microscale_array,
            kinematic_viscosity_array,
        ),
        par.gas.get_taylor_microscale_reynolds_number(
            fluid_rms_velocity_array,
            taylor_microscale_array,
            kinematic_viscosity_array,
        ),
    )

def test_kernels_direct():
    n_points = 3
    kolmogorov_time_array, taylor_microscale_reynolds_number_array, acceleration_variance_array = [
        np.full(n_points, v) for v in (0.40, 500., 0.05)
    ]
    fluid_rms_velocity_array, kinematic_viscosity_array, turbulent_dissipation_array = [
        np.full(n_points, v) for v in (0.35, 1.5e-5, 0.10)
    ]
    taylor_microscale_array = par.gas.get_taylor_microscale(
        fluid_rms_velocity_array, kinematic_viscosity_array, turbulent_dissipation_array
    )

    result_ti = ti.ndarray(dtype=ti.f64, shape=n_points)

    # lagrangian time kernel
    kolmogorov_time_ti = ti.ndarray(ti.f64, n_points)
    taylor_microscale_reynolds_number_ti = ti.ndarray(ti.f64, n_points)
    acceleration_variance_ti = ti.ndarray(ti.f64, n_points)
    kolmogorov_time_ti.from_numpy(kolmogorov_time_array)
    taylor_microscale_reynolds_number_ti.from_numpy(taylor_microscale_reynolds_number_array)
    acceleration_variance_ti.from_numpy(acceleration_variance_array)
    kget_lagrangian_taylor_microscale_time(
        kolmogorov_time_ti,
        taylor_microscale_reynolds_number_ti,
        acceleration_variance_ti,
        result_ti,
    )
    np.testing.assert_allclose(
        result_ti.to_numpy(),
        par.gas.get_lagrangian_taylor_microscale_time(
            kolmogorov_time_array,
            taylor_microscale_reynolds_number_array,
            acceleration_variance_array,
        ),
    )

    # taylor microscale kernel
    fluid_rms_velocity_ti = ti.ndarray(ti.f64, n_points)
    kinematic_viscosity_ti = ti.ndarray(ti.f64, n_points)
    turbulent_dissipation_ti = ti.ndarray(ti.f64, n_points)
    fluid_rms_velocity_ti.from_numpy(fluid_rms_velocity_array)
    kinematic_viscosity_ti.from_numpy(kinematic_viscosity_array)
    turbulent_dissipation_ti.from_numpy(turbulent_dissipation_array)
    kget_taylor_microscale(
        fluid_rms_velocity_ti,
        kinematic_viscosity_ti,
        turbulent_dissipation_ti,
        result_ti,
    )
    np.testing.assert_allclose(result_ti.to_numpy(), taylor_microscale_array)

    # Reynolds number kernel
    taylor_microscale_ti = ti.ndarray(ti.f64, n_points)
    taylor_microscale_ti.from_numpy(taylor_microscale_array)
    kget_taylor_microscale_reynolds_number(
        fluid_rms_velocity_ti,
        taylor_microscale_ti,
        kinematic_viscosity_ti,
        result_ti,
    )
    np.testing.assert_allclose(
        result_ti.to_numpy(),
        par.gas.get_taylor_microscale_reynolds_number(
            fluid_rms_velocity_array,
            taylor_microscale_array,
            kinematic_viscosity_array,
        ),
    )
