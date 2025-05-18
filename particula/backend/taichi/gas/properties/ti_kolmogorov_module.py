"""Taichi-accelerated Kolmogorov scales for gas properties."""

import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register


@ti.func
def fget_kolmogorov_time(
    kinematic_viscosity: ti.f64, turbulent_dissipation: ti.f64
) -> ti.f64:
    """Elementwise Kolmogorov time: sqrt(v / eps)"""
    return ti.sqrt(kinematic_viscosity / turbulent_dissipation)


@ti.func
def fget_kolmogorov_length(
    kinematic_viscosity: ti.f64, turbulent_dissipation: ti.f64
) -> ti.f64:
    """Elementwise Kolmogorov length: sqrt(sqrt(v^3 / eps))"""
    return ti.sqrt(
        ti.sqrt(
            kinematic_viscosity
            * kinematic_viscosity
            * kinematic_viscosity
            / turbulent_dissipation
        )
    )


@ti.func
def fget_kolmogorov_velocity(
    kinematic_viscosity: ti.f64, turbulent_dissipation: ti.f64
) -> ti.f64:
    """Elementwise Kolmogorov velocity: sqrt(sqrt(v * eps))"""
    return ti.sqrt(ti.sqrt(kinematic_viscosity * turbulent_dissipation))


@ti.kernel
def kget_kolmogorov_time(
    kinematic_viscosity: ti.types.ndarray(dtype=ti.f64, ndim=1),
    turbulent_dissipation: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result_array: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """
    Vectorized Kolmogorov time kernel.

    Arguments:
        - kinematic_viscosity : Input array of kinematic viscosity values.
        - turbulent_dissipation : Input array of turbulent dissipation values.
        - result_array : Output array for Kolmogorov time results.

    Returns:
        - None. Results are written to result_array.
    """
    for index in range(result_array.shape[0]):
        result_array[index] = fget_kolmogorov_time(
            kinematic_viscosity[index], turbulent_dissipation[index]
        )


@ti.kernel
def kget_kolmogorov_length(
    kinematic_viscosity: ti.types.ndarray(dtype=ti.f64, ndim=1),
    turbulent_dissipation: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result_array: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """
    Vectorized Kolmogorov length kernel.

    Arguments:
        - kinematic_viscosity : Input array of kinematic viscosity values.
        - turbulent_dissipation : Input array of turbulent dissipation values.
        - result_array : Output array for Kolmogorov length results.

    Returns:
        - None. Results are written to result_array.
    """
    for index in range(result_array.shape[0]):
        result_array[index] = fget_kolmogorov_length(
            kinematic_viscosity[index], turbulent_dissipation[index]
        )


@ti.kernel
def kget_kolmogorov_velocity(
    kinematic_viscosity: ti.types.ndarray(dtype=ti.f64, ndim=1),
    turbulent_dissipation: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result_array: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """
    Vectorized Kolmogorov velocity kernel.

    Arguments:
        - kinematic_viscosity : Input array of kinematic viscosity values.
        - turbulent_dissipation : Input array of turbulent dissipation values.
        - result_array : Output array for Kolmogorov velocity results.

    Returns:
        - None. Results are written to result_array.
    """
    for index in range(result_array.shape[0]):
        result_array[index] = fget_kolmogorov_velocity(
            kinematic_viscosity[index], turbulent_dissipation[index]
        )


@register("get_kolmogorov_time", backend="taichi")
def ti_get_kolmogorov_time(kinematic_viscosity, turbulent_dissipation):
    """
    Taichi wrapper for Kolmogorov time.

    Arguments:
        - kinematic_viscosity : NumPy array of kinematic viscosity values.
        - turbulent_dissipation : NumPy array of turbulent dissipation values.

    Returns:
        - Kolmogorov time as a NumPy array or scalar if input is scalar.
    """
    if not (
        isinstance(kinematic_viscosity, np.ndarray)
        and isinstance(turbulent_dissipation, np.ndarray)
    ):
        raise TypeError("Taichi backend expects NumPy arrays for both inputs.")
    kinematic_viscosity_array = np.atleast_1d(kinematic_viscosity)
    turbulent_dissipation_array = np.atleast_1d(turbulent_dissipation)
    n_values = kinematic_viscosity_array.size
    kinematic_viscosity_field = ti.ndarray(dtype=ti.f64, shape=n_values)
    turbulent_dissipation_field = ti.ndarray(dtype=ti.f64, shape=n_values)
    result_field = ti.ndarray(dtype=ti.f64, shape=n_values)
    kinematic_viscosity_field.from_numpy(kinematic_viscosity_array)
    turbulent_dissipation_field.from_numpy(turbulent_dissipation_array)
    kget_kolmogorov_time(
        kinematic_viscosity_field,
        turbulent_dissipation_field,
        result_field,
    )
    result_array = result_field.to_numpy()
    return result_array.item() if result_array.size == 1 else result_array


@register("get_kolmogorov_length", backend="taichi")
def ti_get_kolmogorov_length(kinematic_viscosity, turbulent_dissipation):
    """
    Taichi wrapper for Kolmogorov length.

    Arguments:
        - kinematic_viscosity : NumPy array of kinematic viscosity values.
        - turbulent_dissipation : NumPy array of turbulent dissipation values.

    Returns:
        - Kolmogorov length as a NumPy array or scalar if input is scalar.
    """
    if not (
        isinstance(kinematic_viscosity, np.ndarray)
        and isinstance(turbulent_dissipation, np.ndarray)
    ):
        raise TypeError("Taichi backend expects NumPy arrays for both inputs.")
    kinematic_viscosity_array = np.atleast_1d(kinematic_viscosity)
    turbulent_dissipation_array = np.atleast_1d(turbulent_dissipation)
    n_values = kinematic_viscosity_array.size
    kinematic_viscosity_field = ti.ndarray(dtype=ti.f64, shape=n_values)
    turbulent_dissipation_field = ti.ndarray(dtype=ti.f64, shape=n_values)
    result_field = ti.ndarray(dtype=ti.f64, shape=n_values)
    kinematic_viscosity_field.from_numpy(kinematic_viscosity_array)
    turbulent_dissipation_field.from_numpy(turbulent_dissipation_array)
    kget_kolmogorov_length(
        kinematic_viscosity_field,
        turbulent_dissipation_field,
        result_field,
    )
    result_array = result_field.to_numpy()
    return result_array.item() if result_array.size == 1 else result_array


@register("get_kolmogorov_velocity", backend="taichi")
def ti_get_kolmogorov_velocity(kinematic_viscosity, turbulent_dissipation):
    """
    Taichi wrapper for Kolmogorov velocity.

    Arguments:
        - kinematic_viscosity : NumPy array of kinematic viscosity values.
        - turbulent_dissipation : NumPy array of turbulent dissipation values.

    Returns:
        - Kolmogorov velocity as a NumPy array or scalar if input is scalar.
    """
    if not (
        isinstance(kinematic_viscosity, np.ndarray)
        and isinstance(turbulent_dissipation, np.ndarray)
    ):
        raise TypeError("Taichi backend expects NumPy arrays for both inputs.")
    kinematic_viscosity_array = np.atleast_1d(kinematic_viscosity)
    turbulent_dissipation_array = np.atleast_1d(turbulent_dissipation)
    n_values = kinematic_viscosity_array.size
    kinematic_viscosity_field = ti.ndarray(dtype=ti.f64, shape=n_values)
    turbulent_dissipation_field = ti.ndarray(dtype=ti.f64, shape=n_values)
    result_field = ti.ndarray(dtype=ti.f64, shape=n_values)
    kinematic_viscosity_field.from_numpy(kinematic_viscosity_array)
    turbulent_dissipation_field.from_numpy(turbulent_dissipation_array)
    kget_kolmogorov_velocity(
        kinematic_viscosity_field,
        turbulent_dissipation_field,
        result_field,
    )
    result_array = result_field.to_numpy()
    return result_array.item() if result_array.size == 1 else result_array
