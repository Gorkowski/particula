"""Taichi-accelerated Taylor microscale helpers."""
import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

@ti.func
def fget_lagrangian_taylor_microscale_time(
    kolmogorov_time: ti.f64,
    taylor_microscale_reynolds_number: ti.f64,
    acceleration_variance: ti.f64
) -> ti.f64:
    """
    Elementwise Lagrangian Taylor microscale time (Taichi).

    Arguments:
        - kolmogorov_time : Kolmogorov time scale.
        - taylor_microscale_reynolds_number : Taylor microscale Reynolds number.
        - acceleration_variance : Acceleration variance.

    Returns:
        - Lagrangian Taylor microscale time.
    """
    return kolmogorov_time * ti.sqrt(
        (2.0 * taylor_microscale_reynolds_number) / (ti.pow(15.0, 0.5) * acceleration_variance)
    )

@ti.func
def fget_taylor_microscale(
    fluid_rms_velocity: ti.f64,
    kinematic_viscosity: ti.f64,
    turbulent_dissipation: ti.f64
) -> ti.f64:
    """Elementwise Taylor microscale (Taichi)."""
    return fluid_rms_velocity * ti.sqrt(
        (15.0 * kinematic_viscosity**2) / turbulent_dissipation
    )

@ti.func
def fget_taylor_microscale_reynolds_number(
    fluid_rms_velocity: ti.f64,
    taylor_microscale: ti.f64,
    kinematic_viscosity: ti.f64
) -> ti.f64:
    """Elementwise Taylor-microscale Reynolds number (Taichi)."""
    return (fluid_rms_velocity * taylor_microscale) / kinematic_viscosity

@ti.kernel
def kget_lagrangian_taylor_microscale_time(
    kolmogorov_time_array: ti.types.ndarray(dtype=ti.f64, ndim=1),
    taylor_microscale_reynolds_number_array: ti.types.ndarray(dtype=ti.f64, ndim=1),
    acceleration_variance_array: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result_array: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """
    Vectorized Lagrangian Taylor microscale time (Taichi).

    Arguments:
        - kolmogorov_time_array : Array of Kolmogorov time scales.
        - taylor_microscale_reynolds_number_array : Array of Taylor microscale Reynolds numbers.
        - acceleration_variance_array : Array of acceleration variances.
        - result_array : Output array for Lagrangian Taylor microscale times.

    Returns:
        - None (results stored in result_array)
    """
    for i in range(result_array.shape[0]):
        result_array[i] = fget_lagrangian_taylor_microscale_time(
            kolmogorov_time_array[i],
            taylor_microscale_reynolds_number_array[i],
            acceleration_variance_array[i]
        )

@ti.kernel
def kget_taylor_microscale(
    fluid_rms_velocity: ti.types.ndarray(dtype=ti.f64, ndim=1),
    kinematic_viscosity: ti.types.ndarray(dtype=ti.f64, ndim=1),
    turbulent_dissipation: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """Vectorized Taylor microscale (Taichi)."""
    for i in range(result.shape[0]):
        result[i] = fget_taylor_microscale(
            fluid_rms_velocity[i], kinematic_viscosity[i], turbulent_dissipation[i]
        )

@ti.kernel
def kget_taylor_microscale_reynolds_number(
    fluid_rms_velocity: ti.types.ndarray(dtype=ti.f64, ndim=1),
    taylor_microscale: ti.types.ndarray(dtype=ti.f64, ndim=1),
    kinematic_viscosity: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """Vectorized Taylor-microscale Reynolds number (Taichi)."""
    for i in range(result.shape[0]):
        result[i] = fget_taylor_microscale_reynolds_number(
            fluid_rms_velocity[i], taylor_microscale[i], kinematic_viscosity[i]
        )

@register("get_lagrangian_taylor_microscale_time", backend="taichi")
def ti_get_lagrangian_taylor_microscale_time(
    kolmogorov_time, taylor_microscale_reynolds_number, acceleration_variance
):
    """
    Taichi wrapper for Lagrangian Taylor microscale time.

    Arguments:
        - kolmogorov_time : Kolmogorov time scale (NumPy array or scalar).
        - taylor_microscale_reynolds_number : Taylor microscale Reynolds number (NumPy array or scalar).
        - acceleration_variance : Acceleration variance (NumPy array or scalar).

    Returns:
        - Lagrangian Taylor microscale time (NumPy array or scalar).
    """
    if not (
        isinstance(kolmogorov_time, np.ndarray)
        and isinstance(taylor_microscale_reynolds_number, np.ndarray)
        and isinstance(acceleration_variance, np.ndarray)
    ):
        raise TypeError("Taichi backend expects NumPy arrays for all inputs.")

    kolmogorov_time_array = np.atleast_1d(kolmogorov_time)
    taylor_microscale_reynolds_number_array = np.atleast_1d(taylor_microscale_reynolds_number)
    acceleration_variance_array = np.atleast_1d(acceleration_variance)
    n = kolmogorov_time_array.size

    kolmogorov_time_ti = ti.ndarray(dtype=ti.f64, shape=n)
    taylor_microscale_reynolds_number_ti = ti.ndarray(dtype=ti.f64, shape=n)
    acceleration_variance_ti = ti.ndarray(dtype=ti.f64, shape=n)
    result_ti_array = ti.ndarray(dtype=ti.f64, shape=n)
    kolmogorov_time_ti.from_numpy(kolmogorov_time_array)
    taylor_microscale_reynolds_number_ti.from_numpy(taylor_microscale_reynolds_number_array)
    acceleration_variance_ti.from_numpy(acceleration_variance_array)

    kget_lagrangian_taylor_microscale_time(
        kolmogorov_time_ti,
        taylor_microscale_reynolds_number_ti,
        acceleration_variance_ti,
        result_ti_array
    )
    result_np = result_ti_array.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np

@register("get_taylor_microscale", backend="taichi")
def ti_get_taylor_microscale(
    fluid_rms_velocity, kinematic_viscosity, turbulent_dissipation
):
    """
    Taichi wrapper for Taylor microscale.

    Arguments:
        - fluid_rms_velocity : Fluid root-mean-square velocity (NumPy array or scalar).
        - kinematic_viscosity : Kinematic viscosity (NumPy array or scalar).
        - turbulent_dissipation : Turbulent dissipation rate (NumPy array or scalar).

    Returns:
        - Taylor microscale (NumPy array or scalar).
    """
    if not (
        isinstance(fluid_rms_velocity, np.ndarray)
        and isinstance(kinematic_viscosity, np.ndarray)
        and isinstance(turbulent_dissipation, np.ndarray)
    ):
        raise TypeError("Taichi backend expects NumPy arrays for all inputs.")

    fluid_rms_velocity_array = np.atleast_1d(fluid_rms_velocity)
    kinematic_viscosity_array = np.atleast_1d(kinematic_viscosity)
    turbulent_dissipation_array = np.atleast_1d(turbulent_dissipation)
    n = fluid_rms_velocity_array.size

    fluid_rms_velocity_ti = ti.ndarray(dtype=ti.f64, shape=n)
    kinematic_viscosity_ti = ti.ndarray(dtype=ti.f64, shape=n)
    turbulent_dissipation_ti = ti.ndarray(dtype=ti.f64, shape=n)
    result_ti_array = ti.ndarray(dtype=ti.f64, shape=n)
    fluid_rms_velocity_ti.from_numpy(fluid_rms_velocity_array)
    kinematic_viscosity_ti.from_numpy(kinematic_viscosity_array)
    turbulent_dissipation_ti.from_numpy(turbulent_dissipation_array)

    kget_taylor_microscale(fluid_rms_velocity_ti, kinematic_viscosity_ti, turbulent_dissipation_ti, result_ti_array)
    result_np = result_ti_array.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np

@register("get_taylor_microscale_reynolds_number", backend="taichi")
def ti_get_taylor_microscale_reynolds_number(
    fluid_rms_velocity, taylor_microscale, kinematic_viscosity
):
    """
    Taichi wrapper for Taylor-microscale Reynolds number.

    Arguments:
        - fluid_rms_velocity : Fluid root-mean-square velocity (NumPy array or scalar).
        - taylor_microscale : Taylor microscale (NumPy array or scalar).
        - kinematic_viscosity : Kinematic viscosity (NumPy array or scalar).

    Returns:
        - Taylor-microscale Reynolds number (NumPy array or scalar).
    """
    if not (
        isinstance(fluid_rms_velocity, np.ndarray)
        and isinstance(taylor_microscale, np.ndarray)
        and isinstance(kinematic_viscosity, np.ndarray)
    ):
        raise TypeError("Taichi backend expects NumPy arrays for all inputs.")

    fluid_rms_velocity_array = np.atleast_1d(fluid_rms_velocity)
    taylor_microscale_array = np.atleast_1d(taylor_microscale)
    kinematic_viscosity_array = np.atleast_1d(kinematic_viscosity)
    n = fluid_rms_velocity_array.size

    fluid_rms_velocity_ti = ti.ndarray(dtype=ti.f64, shape=n)
    taylor_microscale_ti = ti.ndarray(dtype=ti.f64, shape=n)
    kinematic_viscosity_ti = ti.ndarray(dtype=ti.f64, shape=n)
    result_ti_array = ti.ndarray(dtype=ti.f64, shape=n)
    fluid_rms_velocity_ti.from_numpy(fluid_rms_velocity_array)
    taylor_microscale_ti.from_numpy(taylor_microscale_array)
    kinematic_viscosity_ti.from_numpy(kinematic_viscosity_array)

    kget_taylor_microscale_reynolds_number(
        fluid_rms_velocity_ti,
        taylor_microscale_ti,
        kinematic_viscosity_ti,
        result_ti_array
    )
    result_np = result_ti_array.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np
